# Databricks notebook source
# MAGIC %md
# MAGIC # Conversion Control Tower — Conversion Runner
# MAGIC
# MAGIC This notebook is executed by the Conversion Control Tower app to convert
# MAGIC source workloads (SQL Server, SSIS, Informatica) to Databricks.
# MAGIC
# MAGIC **Parameters:**
# MAGIC - `job_id`: Unique job identifier
# MAGIC - `source_path`: UC Volume path to source file
# MAGIC - `catalog`: Target Unity Catalog
# MAGIC - `schema`: Target schema
# MAGIC - `output_path`: UC Volume path for outputs
# MAGIC - `ai_model`: FMAPI model to use for conversion
# MAGIC - `source_type`: Source type (ssis, sql_script, stored_proc, informatica_pc)
# MAGIC - `output_format`: Target output format (pyspark, dlt_sdp, dbt)

# COMMAND ----------

# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Parameters passed from the app
dbutils.widgets.text("job_id", "")
dbutils.widgets.text("source_path", "")
dbutils.widgets.text("catalog", "dev_conversion_tracker")
dbutils.widgets.text("schema", "conversion_tracker")
dbutils.widgets.text("output_path", "")
dbutils.widgets.text("ai_model", "databricks-claude-haiku-4-5")
dbutils.widgets.text("source_type", "auto")
dbutils.widgets.text("output_format", "pyspark")

job_id = dbutils.widgets.get("job_id")
source_path = dbutils.widgets.get("source_path")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
output_path = dbutils.widgets.get("output_path")
ai_model = dbutils.widgets.get("ai_model")
source_type = dbutils.widgets.get("source_type")
output_format = dbutils.widgets.get("output_format")

print(f"Job ID: {job_id}")
print(f"Source: {source_path}")
print(f"Output: {output_path}")
print(f"Model: {ai_model}")
print(f"Source Type: {source_type}")
print(f"Output Format: {output_format}")

# COMMAND ----------

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# Get workspace host and token for FMAPI
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Initialize OpenAI client for FMAPI
client = OpenAI(
    api_key=token,
    base_url=f"https://{host}/serving-endpoints",
)

print(f"FMAPI client initialized for {host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def update_job_status(job_id: str, status: str, error_message: str = None, quality_score: float = None, output_path: str = None):
    """Update job status in tracking table."""
    updates = [f"status = '{status}'", "updated_at = current_timestamp()"]

    if error_message:
        safe_msg = error_message.replace("'", "''")[:500]
        updates.append(f"error_message = '{safe_msg}'")
    if quality_score is not None:
        updates.append(f"quality_score = {quality_score}")
    if output_path:
        updates.append(f"output_path = '{output_path}'")
    if status in ('completed', 'failed'):
        updates.append("completed_at = current_timestamp()")

    update_sql = f"""
    UPDATE {catalog}.conversion_tracker.conversion_jobs
    SET {', '.join(updates)}
    WHERE job_id = '{job_id}'
    """
    try:
        spark.sql(update_sql)
        print(f"Updated job {job_id} status to: {status}")
    except Exception as e:
        print(f"Warning: Could not update job status: {e}")


def emit_status_event(job_id: str, event_type: str, event_data: dict):
    """Emit a status event for real-time tracking."""
    event_id = str(uuid.uuid4())
    event_json = json.dumps(event_data).replace("'", "''")

    insert_sql = f"""
    INSERT INTO {catalog}.conversion_tracker.status_events
    (event_id, job_id, event_type, event_data, created_at)
    VALUES ('{event_id}', '{job_id}', '{event_type}', '{event_json}', current_timestamp())
    """
    try:
        spark.sql(insert_sql)
    except Exception as e:
        print(f"Warning: Could not emit status event: {e}")


def save_validation_result(job_id: str, check_name: str, passed: bool, message: str, category: str, expected: str = None, actual: str = None, severity: str = "error"):
    """Save a validation result."""
    validation_id = str(uuid.uuid4())
    safe_message = message.replace("'", "''")[:500]

    insert_sql = f"""
    INSERT INTO {catalog}.conversion_tracker.validation_results
    (validation_id, job_id, check_name, passed, expected, actual, message, severity, category, created_at)
    VALUES (
        '{validation_id}',
        '{job_id}',
        '{check_name}',
        {str(passed).lower()},
        {f"'{expected}'" if expected else 'NULL'},
        {f"'{actual}'" if actual else 'NULL'},
        '{safe_message}',
        '{severity}',
        '{category}',
        current_timestamp()
    )
    """
    try:
        spark.sql(insert_sql)
    except Exception as e:
        print(f"Warning: Could not save validation result: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read and Detect Source

# COMMAND ----------

try:
    update_job_status(job_id, "parsing")
    emit_status_event(job_id, "status_change", {"status": "parsing", "message": "Reading source file"})

    # Read source file from UC Volume
    print(f"Reading source file: {source_path}")

    try:
        with open(source_path, "r") as f:
            source_content = f.read()
        print(f"Read {len(source_content)} characters via direct file access")
    except Exception as e:
        # Fallback to dbutils
        source_content = dbutils.fs.head(source_path.replace("/Volumes", "dbfs:/Volumes"), 2000000)
        print(f"Read {len(source_content)} characters via dbutils")

    # Auto-detect source type if set to "auto" or if it looks wrong
    _ext = Path(source_path).suffix.lower()

    if source_type == "auto":
        if _ext == ".dtsx":
            source_type = "ssis"
        elif _ext == ".sql":
            source_type = "sql_script"
        elif _ext == ".xml":
            if "<POWERMART" in source_content[:2000] or "<REPOSITORY" in source_content[:2000]:
                source_type = "informatica_pc"
            elif "<DTS:Executable" in source_content[:2000]:
                source_type = "ssis"
            else:
                source_type = "ssis"  # Default XML to SSIS
        else:
            source_type = "sql_script"  # Default fallback
        print(f"Auto-detected source type: {source_type}")

    # Correct mismatches (e.g., user selected SSIS but uploaded .sql)
    elif source_type == "ssis" and _ext == ".sql":
        source_type = "sql_script"
        print(f"Auto-corrected source type from ssis to sql_script based on .sql extension")
    elif source_type == "ssis" and _ext == ".xml":
        if "<POWERMART" in source_content[:2000] or "<REPOSITORY" in source_content[:2000]:
            source_type = "informatica_pc"
            print(f"Auto-corrected source type from ssis to informatica_pc (PowerCenter XML detected)")

    # For Informatica XML, try to extract basic structure metadata
    xml_summary = ""
    if source_type == "informatica_pc":
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(source_content)
            mappings = root.findall(".//{*}MAPPING") or root.findall(".//MAPPING")
            workflows = root.findall(".//{*}WORKFLOW") or root.findall(".//WORKFLOW")
            sources = root.findall(".//{*}SOURCE") or root.findall(".//SOURCE")
            targets = root.findall(".//{*}TARGET") or root.findall(".//TARGET")
            xml_summary = (
                f"Informatica PowerCenter export with {len(mappings)} mapping(s), "
                f"{len(workflows)} workflow(s), {len(sources)} source(s), {len(targets)} target(s)"
            )
            print(f"  Pre-scan: {xml_summary}")
        except Exception:
            print("  Could not pre-scan XML structure")

    print(f"Source type: {source_type}")
    print(f"Output format: {output_format}")
    parsing_success = True

    emit_status_event(job_id, "progress", {
        "progress": 25,
        "stage": "parsing",
        "message": f"Parsed source ({source_type}): {len(source_content)} chars"
    })

except Exception as e:
    parsing_success = False
    error_msg = str(e)
    print(f"Parsing failed: {error_msg}")
    update_job_status(job_id, "failed", error_message=f"Parsing failed: {error_msg}")
    emit_status_event(job_id, "error", {"error": error_msg, "stage": "parsing"})
    dbutils.notebook.exit(json.dumps({"status": "failed", "error": error_msg}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Build Dynamic System Prompt

# COMMAND ----------

# =============================================================================
# System prompts by SOURCE TYPE
# =============================================================================

SSIS_SYSTEM_PROMPT = """You are an expert at converting SSIS (SQL Server Integration Services) packages to Databricks.
Convert the provided SSIS package XML to a Databricks workflow with notebooks.

Follow Databricks best practices:
- Use Delta Lake for all tables
- Use Unity Catalog three-level namespace (catalog.schema.table)
- Map SSIS data flows to DataFrame operations
- Convert T-SQL expressions to Spark SQL
- Map SSIS connection managers to Databricks connection configs
- Map Execute SQL Tasks to spark.sql() cells
- Map Data Flow Tasks to PySpark DataFrame pipelines
"""

SQL_SYSTEM_PROMPT = """You are an expert at converting SQL Server stored procedures and scripts to Databricks.
Convert the provided SQL to Databricks-compatible code.

Conversion rules:
- Replace T-SQL syntax with Spark SQL
- Use Delta Lake tables (catalog.schema.table)
- Convert stored procedure parameters to widgets
- Replace cursors with set-based operations
- Convert temp tables to views or CTEs where possible
- GETDATE() -> current_timestamp()
- ISNULL(x, default) -> COALESCE(x, default)
- TOP N -> LIMIT N
- IDENTITY columns -> monotonically_increasing_id() or row_number()
"""

INFORMATICA_SYSTEM_PROMPT = """You are an expert at converting Informatica PowerCenter workflows and mappings to Databricks.
Convert the provided Informatica PowerCenter export to a Databricks workflow with notebooks.

## Informatica PowerCenter to Spark Mappings

### Expression Function Conversions
- IIF(cond, true, false) -> CASE WHEN cond THEN true ELSE false END
- DECODE(val, match1, res1, ..., default) -> CASE WHEN val = match1 THEN res1 ... ELSE default END
- IS_SPACES(x) -> TRIM(x) = ''
- TO_DATE(x, fmt) -> TO_DATE(x, fmt) (format string syntax differs)
- TO_CHAR(date, fmt) -> DATE_FORMAT(date, fmt)
- TO_INTEGER(x) -> CAST(x AS INT)
- SYSDATE -> CURRENT_TIMESTAMP()
- :LKP.xxx(args) -> LEFT JOIN
- $$param -> widget reference or config lookup

### Transformation to Spark Patterns
- Source Qualifier -> spark.sql("SELECT ... FROM ...")
- Expression -> withColumn() or CTE expressions
- Filter -> .filter() or WHERE clause
- Joiner -> .join() with join type
- Lookup -> LEFT JOIN (connected) or correlated subquery (unconnected)
- Aggregator -> .groupBy().agg()
- Router -> Multiple .filter() branches
- Rank -> Window + row_number() / rank()
- Sequence Generator -> monotonically_increasing_id()
- Sorter -> .orderBy()

### Code Structure
- Cell 1: Parameters (dbutils.widgets)
- Cell 2: Schema/database selection
- Cells 3-N: One cell per transformation in topological order
- Final cell: Target write (INSERT INTO / MERGE INTO / CREATE TABLE AS SELECT)

Follow Databricks best practices:
- Use Delta Lake for all tables
- Use Unity Catalog three-level namespace (catalog.schema.table)
- Generate one notebook per Informatica mapping
- Preserve original transformation names as comments for traceability
"""

# =============================================================================
# Output format modifiers
# =============================================================================

PYSPARK_OUTPUT_MODIFIER = """
## Output Format: PySpark Notebooks
Generate standard PySpark notebook code:
- Use spark.read / spark.sql for reading
- Use df.write.format("delta").mode("overwrite").saveAsTable() for writing
- Standard DataFrame API (filter, select, join, groupBy, etc.)
- Use dbutils.widgets for parameters
"""

DLT_SDP_OUTPUT_MODIFIER = """
## Output Format: Spark Declarative Pipelines (DLT/SDP)
Generate Databricks DLT / Spark Declarative Pipeline code:
- Use @dlt.table decorator for table definitions
- Use dlt.read() or dlt.read_stream() for reading upstream tables
- Use spark.readStream.format("cloudFiles") for Auto Loader ingestion
- Define expectations with @dlt.expect or @dlt.expect_all
- Use CREATE OR REFRESH STREAMING TABLE or CREATE OR REFRESH MATERIALIZED VIEW syntax
- Follow bronze/silver/gold medallion architecture where appropriate
- Import dlt module: `import dlt`
"""

DBT_OUTPUT_MODIFIER = """
## Output Format: dbt Models
Generate dbt model SQL files:
- Use {{ ref('model_name') }} for referencing other models
- Use {{ source('source_name', 'table_name') }} for source tables
- Use {{ config(materialized='table') }} or {{ config(materialized='incremental') }}
- Include schema.yml definitions for documentation
- Follow dbt naming conventions (stg_, int_, fct_, dim_)
- Use Jinja macros for reusable logic
"""

# =============================================================================
# JSON output schema instruction (common to all)
# =============================================================================

JSON_OUTPUT_INSTRUCTION = """
Your response must be valid JSON with this structure:
{
  "workflow": {
    "name": "workflow_name",
    "description": "Brief description",
    "tasks": [
      {
        "task_key": "unique_key",
        "description": "What this task does",
        "depends_on": ["optional_dependency_keys"],
        "notebook_path": "/path/to/notebook"
      }
    ]
  },
  "notebooks": [
    {
      "name": "notebook_name.py",
      "description": "What this notebook does",
      "code": "# Full code here..."
    }
  ],
  "quality_notes": {
    "score": 0.85,
    "warnings": ["Any conversion warnings"],
    "manual_review": ["Items needing manual review"]
  }
}

For simple SQL conversions with no workflow, omit the "workflow" key.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Call FMAPI for Conversion

# COMMAND ----------

if parsing_success:
    try:
        update_job_status(job_id, "converting")
        emit_status_event(job_id, "status_change", {"status": "converting", "message": f"Converting via {ai_model}"})

        # Select system prompt based on source_type
        if source_type == "ssis":
            base_prompt = SSIS_SYSTEM_PROMPT
        elif source_type == "informatica_pc":
            base_prompt = INFORMATICA_SYSTEM_PROMPT
        else:
            base_prompt = SQL_SYSTEM_PROMPT

        # Add output format modifier
        if output_format == "dlt_sdp":
            format_modifier = DLT_SDP_OUTPUT_MODIFIER
        elif output_format == "dbt":
            format_modifier = DBT_OUTPUT_MODIFIER
        else:
            format_modifier = PYSPARK_OUTPUT_MODIFIER

        # Assemble full system prompt
        system_prompt = base_prompt + format_modifier + JSON_OUTPUT_INSTRUCTION

        # Build user prompt based on source type
        if source_type == "ssis":
            user_prompt = f"Convert this SSIS package to Databricks:\n\n```xml\n{source_content}\n```"
        elif source_type == "informatica_pc":
            context_note = f"\n\nPre-scan summary: {xml_summary}" if xml_summary else ""
            user_prompt = (
                f"Convert this Informatica PowerCenter export to Databricks.{context_note}\n\n"
                "Important:\n"
                "- Generate one notebook per mapping\n"
                "- Follow topological order for transformations\n"
                "- Convert $$parameters to dbutils.widgets\n"
                "- Convert :LKP references to LEFT JOINs\n"
                "- Variable ports (V flag) are intermediate — inline or use CTEs\n"
                "- Preserve original transformation names as comments\n\n"
                f"```xml\n{source_content}\n```"
            )
        elif source_type == "stored_proc":
            user_prompt = f"Convert this SQL Server stored procedure to Databricks:\n\n```sql\n{source_content}\n```"
        else:
            user_prompt = f"Convert this SQL Server code to Databricks:\n\n```sql\n{source_content}\n```"

        print(f"Calling {ai_model} for conversion...")
        print(f"  System prompt: {len(system_prompt)} chars")
        print(f"  User prompt: {len(user_prompt)} chars")

        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=16000,
        )

        # Normalize response content (handles different model response formats)
        raw_content = response.choices[0].message.content

        def _extract_text_from_block(block):
            """Extract text from a content block (handles GPT-OSS typed blocks)."""
            if isinstance(block, str):
                return block
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    return None
                if "text" in block:
                    return block["text"]
                if "content" in block:
                    return str(block["content"])
                return None
            block_type = getattr(block, 'type', None)
            if block_type == "reasoning":
                return None
            if hasattr(block, 'text'):
                return str(block.text)
            if hasattr(block, 'content'):
                return str(block.content)
            return str(block)

        def normalize_content(content):
            """Normalize content to string - handles different model response formats."""
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if hasattr(content, '__iter__'):
                text_parts = [_extract_text_from_block(b) for b in content]
                text_parts = [p for p in text_parts if p]
                if text_parts:
                    return "\n".join(text_parts)
                return str(content)
            return str(content)

        result_text = normalize_content(raw_content)
        if not isinstance(result_text, str):
            result_text = str(result_text)

        print(f"Got response: {len(result_text)} characters (type: {type(raw_content).__name__})")

        emit_status_event(job_id, "progress", {
            "progress": 60,
            "stage": "converting",
            "message": f"AI conversion complete ({len(result_text)} chars)"
        })

        conversion_success = True

    except Exception as e:
        conversion_success = False
        error_msg = str(e)
        print(f"Conversion failed: {error_msg}")
        update_job_status(job_id, "failed", error_message=f"Conversion failed: {error_msg}")
        emit_status_event(job_id, "error", {"error": error_msg, "stage": "converting"})
        dbutils.notebook.exit(json.dumps({"status": "failed", "error": error_msg}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Parse JSON Response

# COMMAND ----------

if parsing_success and conversion_success:
    try:
        # Handle markdown code blocks
        parse_text = result_text
        if "```json" in parse_text:
            parse_text = parse_text.split("```json")[1].split("```")[0]
        elif "```" in parse_text:
            parse_text = parse_text.split("```")[1].split("```")[0]

        result = json.loads(parse_text.strip())
        print("Successfully parsed JSON response")

        if "workflow" in result:
            print(f"  Workflow: {result['workflow'].get('name', 'unnamed')}")
            print(f"  Tasks: {len(result['workflow'].get('tasks', []))}")
        print(f"  Notebooks: {len(result.get('notebooks', []))}")
        if "quality_notes" in result:
            print(f"  Quality score: {result['quality_notes'].get('score', 'N/A')}")

    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        print(f"Warning: Could not parse JSON response: {type(e).__name__}: {e}")
        _safe_text = result_text if isinstance(result_text, str) else str(result_text)
        result = {
            "notebooks": [{
                "name": "conversion_output.py",
                "description": "AI conversion output (unparsed)",
                "code": f"# AI Response (manual parsing needed):\n'''\n{_safe_text}\n'''"
            }],
            "quality_notes": {
                "score": 0.5,
                "warnings": ["AI response was not valid JSON - manual review required"],
                "manual_review": ["Full response needs parsing"]
            }
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Write Outputs

# COMMAND ----------

if parsing_success and conversion_success:
    try:
        output_base = output_path or f"/Volumes/{catalog}/conversion_tracker/files/outputs/{job_id}"

        # Create output directories
        os.makedirs(output_base, exist_ok=True)
        notebooks_dir = f"{output_base}/notebooks"
        os.makedirs(notebooks_dir, exist_ok=True)

        converted_files = []

        # Write workflow if present
        if "workflow" in result:
            workflow_path = f"{output_base}/workflow.json"
            with open(workflow_path, "w") as f:
                json.dump(result["workflow"], f, indent=2)
            converted_files.append({"type": "workflow", "path": workflow_path})
            print(f"Wrote workflow to {workflow_path}")

        # Write notebooks
        for notebook in result.get("notebooks", []):
            nb_name = notebook.get("name", "notebook.py")
            if not nb_name.endswith(".py") and not nb_name.endswith(".sql"):
                nb_name += ".py"
            nb_path = f"{notebooks_dir}/{nb_name}"
            nb_code = notebook.get("code", "# Empty notebook")

            # Add Databricks notebook header
            header = f"""# Databricks notebook source
# MAGIC %md
# MAGIC # {notebook.get("description", "Converted notebook")}
# MAGIC
# MAGIC Generated by Conversion Control Tower
# MAGIC - Source type: {source_type}
# MAGIC - Output format: {output_format}
# MAGIC - AI model: {ai_model}
# MAGIC - Job ID: {job_id}

"""
            full_code = header + nb_code

            with open(nb_path, "w") as f:
                f.write(full_code)
            converted_files.append({"type": "notebook", "path": nb_path})
            print(f"Wrote notebook to {nb_path}")

        # Write quality report
        if "quality_notes" in result:
            report_path = f"{output_base}/quality_report.json"
            with open(report_path, "w") as f:
                json.dump(result["quality_notes"], f, indent=2)
            print(f"Wrote quality report to {report_path}")

        # Write conversion metadata
        metadata = {
            "job_id": job_id,
            "source_type": source_type,
            "output_format": output_format,
            "ai_model": ai_model,
            "source_path": source_path,
            "files_generated": len(converted_files),
            "converted_at": datetime.now().isoformat(),
        }
        meta_path = f"{output_base}/conversion_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        update_job_status(job_id, "converting", output_path=output_base)

        emit_status_event(job_id, "progress", {
            "progress": 80,
            "stage": "writing_outputs",
            "message": f"Wrote {len(converted_files)} files to {output_base}"
        })

        print(f"\nWrote {len(converted_files)} output files")

    except Exception as e:
        error_msg = str(e)
        print(f"Failed to write outputs: {error_msg}")
        update_job_status(job_id, "failed", error_message=f"Output write failed: {error_msg}")
        emit_status_event(job_id, "error", {"error": error_msg, "stage": "writing_outputs"})
        dbutils.notebook.exit(json.dumps({"status": "failed", "error": error_msg}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Validate Output

# COMMAND ----------

if parsing_success and conversion_success:
    try:
        update_job_status(job_id, "validating")
        emit_status_event(job_id, "status_change", {"status": "validating", "message": "Validating conversion output"})

        validation_results = []

        # Check 1: Output files exist
        try:
            output_files = os.listdir(output_base)
            passed = len(output_files) > 0
            save_validation_result(
                job_id=job_id,
                check_name="output_files_exist",
                passed=passed,
                expected=">0",
                actual=str(len(output_files)),
                message="Output files generated successfully" if passed else "No output files generated",
                category="completeness"
            )
            validation_results.append({"name": "output_files_exist", "passed": passed})
        except Exception as e:
            save_validation_result(job_id, "output_files_exist", False, str(e), "completeness")
            validation_results.append({"name": "output_files_exist", "passed": False})

        # Check 2: Valid JSON in response
        json_valid = isinstance(result, dict) and "notebooks" in result
        save_validation_result(
            job_id=job_id,
            check_name="valid_json_response",
            passed=json_valid,
            message="AI returned valid structured JSON" if json_valid else "AI response was not valid JSON",
            category="completeness",
            severity="warning" if not json_valid else "info"
        )
        validation_results.append({"name": "valid_json_response", "passed": json_valid})

        # Check 3: Notebooks have valid Python syntax
        try:
            nb_dir = f"{output_base}/notebooks"
            if os.path.isdir(nb_dir):
                nb_files = [f for f in os.listdir(nb_dir) if f.endswith(".py")]
                syntax_valid = True
                syntax_errors = []
                for nb_file in nb_files:
                    nb_path = f"{nb_dir}/{nb_file}"
                    with open(nb_path, "r") as f:
                        nb_content = f.read()
                    try:
                        compile(nb_content, nb_file, 'exec')
                    except SyntaxError as se:
                        syntax_valid = False
                        syntax_errors.append(f"{nb_file}: {se}")

                save_validation_result(
                    job_id=job_id,
                    check_name="notebooks_syntax_valid",
                    passed=syntax_valid,
                    message="All notebooks have valid Python syntax" if syntax_valid else f"Syntax errors: {'; '.join(syntax_errors[:3])}",
                    category="logic"
                )
                validation_results.append({"name": "notebooks_syntax_valid", "passed": syntax_valid})
        except Exception as e:
            save_validation_result(job_id, "notebooks_syntax_valid", False, str(e), "logic")
            validation_results.append({"name": "notebooks_syntax_valid", "passed": False})

        # Check 4: Workflow JSON valid (if present)
        if "workflow" in result:
            has_tasks = len(result.get("workflow", {}).get("tasks", [])) > 0
            save_validation_result(
                job_id=job_id,
                check_name="workflow_json_valid",
                passed=has_tasks,
                expected=">0 tasks",
                actual=str(len(result.get("workflow", {}).get("tasks", []))),
                message="Workflow has tasks defined" if has_tasks else "Workflow has no tasks",
                category="logic"
            )
            validation_results.append({"name": "workflow_json_valid", "passed": has_tasks})

        # Check 5: Output format compliance
        format_ok = True
        if output_format == "dlt_sdp":
            # Check if any notebook mentions dlt or streaming table
            for nb in result.get("notebooks", []):
                code = nb.get("code", "")
                if "dlt" not in code.lower() and "streaming" not in code.lower() and "materialized" not in code.lower():
                    format_ok = False
                    break
        elif output_format == "dbt":
            for nb in result.get("notebooks", []):
                code = nb.get("code", "")
                if "ref(" not in code and "source(" not in code and "config(" not in code:
                    format_ok = False
                    break

        save_validation_result(
            job_id=job_id,
            check_name="output_format_compliance",
            passed=format_ok,
            expected=output_format,
            message=f"Output follows {output_format} patterns" if format_ok else f"Output may not follow {output_format} patterns",
            category="best_practices",
            severity="warning" if not format_ok else "info"
        )
        validation_results.append({"name": "output_format_compliance", "passed": format_ok})

        # Calculate quality score
        passed_count = sum(1 for r in validation_results if r["passed"])
        total_count = len(validation_results)
        quality_score = passed_count / total_count if total_count > 0 else 0.0

        # Also factor in AI's own quality score if available
        ai_score = result.get("quality_notes", {}).get("score")
        if ai_score is not None:
            # Blend validation score with AI self-assessment (60/40 weight)
            quality_score = 0.6 * quality_score + 0.4 * float(ai_score)

        emit_status_event(job_id, "progress", {
            "progress": 95,
            "stage": "validating",
            "message": f"Validation complete: {passed_count}/{total_count} checks passed"
        })

        print(f"Validation: {passed_count}/{total_count} checks passed, score: {quality_score:.2f}")

    except Exception as e:
        error_msg = str(e)
        print(f"Validation failed: {error_msg}")
        quality_score = 0.5  # Default if validation errors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Build Quality Report and Finalize

# COMMAND ----------

if parsing_success and conversion_success:
    # Build quality report for the app to consume
    quality_checks = []
    recommendations = []

    # Fetch validation results from table
    try:
        validation_df = spark.sql(f"""
            SELECT check_name, passed, expected, actual, message, severity, category
            FROM {catalog}.conversion_tracker.validation_results
            WHERE job_id = '{job_id}'
            ORDER BY created_at
        """)

        for row in validation_df.collect():
            quality_checks.append({
                "check_id": str(uuid.uuid4()),
                "check_name": row.check_name.replace("_", " ").title(),
                "category": row.category or "general",
                "passed": row.passed,
                "severity": row.severity or ("info" if row.passed else "error"),
                "message": row.message or "",
                "details": f"Expected: {row.expected}, Actual: {row.actual}" if row.expected else None,
                "suggestion": None
            })

            if not row.passed:
                if "syntax" in row.check_name.lower():
                    recommendations.append("Review generated code for syntax errors and fix manually if needed")
                elif "workflow" in row.check_name.lower():
                    recommendations.append("Verify workflow JSON structure matches Databricks Jobs API schema")
                elif "format" in row.check_name.lower():
                    recommendations.append(f"Review generated code to ensure it follows {output_format} patterns")
                elif "output" in row.check_name.lower():
                    recommendations.append("Check source file format and retry conversion")
    except Exception as e:
        print(f"Warning: Could not fetch validation results: {e}")

    # Add AI's own warnings and manual review items
    ai_notes = result.get("quality_notes", {})
    for warning in ai_notes.get("warnings", []):
        quality_checks.append({
            "check_id": str(uuid.uuid4()),
            "check_name": "AI Warning",
            "category": "ai_assessment",
            "passed": False,
            "severity": "warning",
            "message": warning,
            "details": None,
            "suggestion": None
        })
    for review_item in ai_notes.get("manual_review", []):
        quality_checks.append({
            "check_id": str(uuid.uuid4()),
            "check_name": "Manual Review Needed",
            "category": "ai_assessment",
            "passed": False,
            "severity": "warning",
            "message": review_item,
            "details": None,
            "suggestion": "Review this item manually before deploying"
        })
        recommendations.append(review_item)

    # Default checks if none found
    if not quality_checks:
        quality_checks = [{
            "check_id": str(uuid.uuid4()),
            "check_name": "Conversion Complete",
            "category": "completeness",
            "passed": True,
            "severity": "info",
            "message": "Conversion process completed successfully",
            "details": None,
            "suggestion": None
        }]

    if not recommendations:
        recommendations.append("Review generated code before deploying to production")

    # Build summary
    passed_checks = sum(1 for c in quality_checks if c["passed"])
    if quality_score >= 0.8:
        summary = f"Conversion completed with high quality ({passed_checks}/{len(quality_checks)} checks passed)"
    elif quality_score >= 0.5:
        summary = f"Conversion completed with some issues ({len(quality_checks) - passed_checks} checks need attention)"
    else:
        summary = f"Conversion has significant issues ({len(quality_checks) - passed_checks} checks failed)"

    quality_report = {
        "overall_score": quality_score * 100,
        "checks": quality_checks,
        "summary": summary,
        "recommendations": list(set(recommendations)),
        "generated_at": datetime.now().isoformat()
    }

    # Final status update
    update_job_status(job_id, "completed", quality_score=quality_score, output_path=output_base)
    emit_status_event(job_id, "status_change", {
        "status": "completed",
        "message": summary,
        "quality_score": quality_score
    })

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Job ID: {job_id}")
    print(f"Source Type: {source_type}")
    print(f"Output Format: {output_format}")
    print(f"AI Model: {ai_model}")
    print(f"Quality Score: {quality_score:.0%}")
    print(f"Output Path: {output_base}")
    print(f"Files Generated: {len(converted_files)}")
    print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Return Result to Control Tower

# COMMAND ----------

# Return structured result that the Control Tower can parse
notebook_result = {
    "job_id": job_id,
    "status": "completed",
    "quality_score": quality_score,
    "output_path": output_base,
    "quality_report": quality_report,
    "files_generated": len(converted_files),
    "source_type": source_type,
    "output_format": output_format,
}

dbutils.notebook.exit(json.dumps(notebook_result))
