# Databricks notebook source
# MAGIC %md
# MAGIC # Original Output Simulator
# MAGIC
# MAGIC Simulates the expected output from original source logic.
# MAGIC Uses an LLM to extract schemas from the source file, generates synthetic data
# MAGIC with Faker, and produces what the original code WOULD have produced.
# MAGIC
# MAGIC **Parameters:**
# MAGIC - `conversion_job_id`: Original conversion job ID
# MAGIC - `source_file_path`: UC Volume path to uploaded source file (.dtsx, .sql, .xml)
# MAGIC - `output_path`: UC Volume path for expected output
# MAGIC - `source_data_output_path`: UC Volume path to save synthetic source data (for converted runner)

# COMMAND ----------

# MAGIC %pip install openai faker
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Get parameters
dbutils.widgets.text("conversion_job_id", "")
dbutils.widgets.text("source_file_path", "")
dbutils.widgets.text("output_path", "")
dbutils.widgets.text("source_data_output_path", "")
dbutils.widgets.text("validation_table", "")

conversion_job_id = dbutils.widgets.get("conversion_job_id")
source_file_path = dbutils.widgets.get("source_file_path")
output_path = dbutils.widgets.get("output_path")
source_data_output_path = dbutils.widgets.get("source_data_output_path")
validation_table = dbutils.widgets.get("validation_table").strip()

print(f"Conversion Job ID: {conversion_job_id}")
print(f"Source File Path: {source_file_path}")
print(f"Output Path: {output_path}")
print(f"Source Data Output Path: {source_data_output_path}")
print(f"Validation Table: {validation_table or '(none — will generate synthetic data)'}")

# Validate required parameters
import json as _json
_missing = [p for p, v in [("source_file_path", source_file_path), ("output_path", output_path), ("source_data_output_path", source_data_output_path)] if not v]
if _missing:
    dbutils.notebook.exit(_json.dumps({"status": "error", "message": f"Required parameters are empty: {', '.join(_missing)}"}))

# COMMAND ----------

import json
import os
from pathlib import Path
from datetime import datetime, date, timedelta
from decimal import Decimal
import random
from faker import Faker
from openai import OpenAI

fake = Faker()
Faker.seed(42)
random.seed(42)

# Get workspace host and token for FMAPI
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=token,
    base_url=f"https://{host}/serving-endpoints",
)

print(f"FMAPI client initialized for {host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fast Path: Real UC Table (skip synthetic generation)
# MAGIC
# MAGIC When a `validation_table` is provided (e.g. a federated table or Lakeflow Connect replica),
# MAGIC we read real data from it instead of generating synthetic Faker data. This gives an apples-to-apples
# MAGIC comparison: the same real source data is used as both the expected output and input to the converted code.

# COMMAND ----------

import pandas as pd

use_real_table = bool(validation_table)

if use_real_table:
    print(f"=== REAL TABLE MODE: reading from {validation_table} ===")
    print("Skipping LLM schema extraction and Faker data generation.")

    # Read the UC table via Spark SQL (needed for federated tables)
    try:
        real_df = spark.sql(f"SELECT * FROM {validation_table} LIMIT 5000")
        real_pdf = real_df.toPandas()
        print(f"Read {len(real_pdf)} rows, {len(real_pdf.columns)} columns from {validation_table}")
    except Exception as e:
        print(f"Error reading from {validation_table}: {e}")
        print("Falling back to synthetic data generation.")
        use_real_table = False

if use_real_table:
    import os

    # Extract table name from the fully qualified path
    table_short_name = validation_table.split(".")[-1]

    # Save source data for the converted runner
    os.makedirs(source_data_output_path, exist_ok=True)
    source_parquet = f"{source_data_output_path}/{table_short_name}.parquet"
    real_pdf.to_parquet(source_parquet, index=False, engine="pyarrow")
    print(f"Saved source data ({len(real_pdf)} rows) to {source_parquet}")

    # Save expected output (same data — the baseline for comparison)
    os.makedirs(output_path, exist_ok=True)
    expected_file = f"{output_path}/expected_output.parquet"
    real_pdf.to_parquet(expected_file, index=False, engine="pyarrow")
    print(f"Saved expected output ({len(real_pdf)} rows) to {expected_file}")

    # Save schemas JSON for reference
    schemas = {
        "tables": [{
            "table_name": table_short_name,
            "role": "source",
            "columns": [{"name": c, "type": str(real_pdf[c].dtype)} for c in real_pdf.columns]
        }],
        "validation_table": validation_table,
        "mode": "real_table",
    }
    import json as _schemas_json
    with open(f"{source_data_output_path}/schemas.json", "w") as f:
        _schemas_json.dump(schemas, f, indent=2)

    # Save completion metadata
    completion_metadata = {
        "conversion_job_id": conversion_job_id,
        "mode": "real_table",
        "validation_table": validation_table,
        "row_count": len(real_pdf),
        "column_count": len(real_pdf.columns),
        "columns": list(real_pdf.columns),
        "completed_at": datetime.now().isoformat(),
    }
    with open(f"{output_path}/simulation_metadata.json", "w") as f:
        _schemas_json.dump(completion_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("REAL TABLE SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Source: {validation_table}")
    print(f"Rows: {len(real_pdf)}, Columns: {len(real_pdf.columns)}")
    print(f"Source Data: {source_parquet}")
    print(f"Expected Output: {expected_file}")
    print(f"{'='*60}")

    # Exit early — skip all synthetic generation steps below
    dbutils.notebook.exit(_schemas_json.dumps({"status": "success", "mode": "real_table", "rows": len(real_pdf)}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read Source File and Detect Type

# COMMAND ----------

source_ext = Path(source_file_path).suffix.lower()
print(f"Source file extension: {source_ext}")

try:
    with open(source_file_path, "r") as f:
        source_content = f.read()
    print(f"Read {len(source_content)} characters via direct file access")
except Exception:
    source_content = dbutils.fs.head(source_file_path.replace("/Volumes", "dbfs:/Volumes"), 2000000)
    print(f"Read {len(source_content)} characters via dbutils")

# Detect source type
if source_ext == ".dtsx":
    source_type = "ssis"
elif source_ext == ".xml":
    if "<POWERMART" in source_content[:2000] or "<REPOSITORY" in source_content[:2000]:
        source_type = "informatica_pc"
    else:
        source_type = "ssis"
elif source_ext == ".sql":
    source_type = "sql_script"
else:
    source_type = "sql_script"

print(f"Detected source type: {source_type}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Extract Schemas via LLM

# COMMAND ----------

# Use LLM to extract table schemas from the source code
schema_prompt = f"""Analyze the following {source_type} source code and extract ALL table schemas involved.

IMPORTANT RULES:
- For SSIS (.dtsx): Extract EVERY column from OLE DB Source, OLE DB Destination, Flat File,
  Derived Column transforms, and column mappings. The destination table should include ALL columns
  that are mapped/written, not just status flags.
- For SQL: Extract all SELECT columns, INSERT INTO targets, and CREATE TABLE definitions.
- For Informatica: Extract all source/target definitions with ALL their ports/columns.
- Include EVERY column you can find — do not summarize or truncate the column list.

For each table, identify:
- Table name
- Role: "source" (read from), "destination" (written to), or "lookup" (used for lookups/joins)
- ALL columns with their data types (list every single column, do not skip any)

Return valid JSON with this structure:
{{
  "tables": [
    {{
      "table_name": "table_name",
      "role": "source|destination|lookup",
      "columns": [
        {{"name": "col_name", "type": "string|int|bigint|decimal|double|date|timestamp|boolean", "precision": null, "scale": null, "nullable": true}}
      ]
    }}
  ]
}}

Source code (first 50000 chars):
```
{source_content[:50000]}
```
"""

print("Extracting schemas via LLM...")

try:
    schema_response = client.chat.completions.create(
        model="databricks-claude-haiku-4-5",
        messages=[
            {"role": "system", "content": "You are a database schema extraction expert. You MUST list every single column in each table — never summarize or truncate. Return only valid JSON, no markdown."},
            {"role": "user", "content": schema_prompt},
        ],
        temperature=0.0,
        max_tokens=8000,
    )

    schema_text = schema_response.choices[0].message.content
    if isinstance(schema_text, list):
        schema_text = next((b.text if hasattr(b, 'text') else str(b) for b in schema_text if getattr(b, 'type', '') != 'reasoning'), str(schema_text))

    # Parse JSON from response
    if "```json" in schema_text:
        schema_text = schema_text.split("```json")[1].split("```")[0]
    elif "```" in schema_text:
        schema_text = schema_text.split("```")[1].split("```")[0]

    extracted_schemas = json.loads(schema_text.strip())
    tables = extracted_schemas.get("tables", [])
    print(f"Extracted {len(tables)} tables from source")
    for t in tables:
        print(f"  {t['role']}: {t['table_name']} ({len(t['columns'])} columns)")

except Exception as e:
    print(f"Warning: LLM schema extraction failed: {e}")
    # Fallback: generic schema
    tables = [
        {
            "table_name": "source_data",
            "role": "source",
            "columns": [
                {"name": "id", "type": "int", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "value", "type": "decimal", "precision": 10, "scale": 2, "nullable": True},
                {"name": "category", "type": "string", "nullable": True},
                {"name": "created_date", "type": "timestamp", "nullable": True},
                {"name": "status", "type": "string", "nullable": True},
            ]
        },
        {
            "table_name": "output_data",
            "role": "destination",
            "columns": [
                {"name": "id", "type": "int", "nullable": False},
                {"name": "name", "type": "string", "nullable": True},
                {"name": "value", "type": "decimal", "precision": 10, "scale": 2, "nullable": True},
                {"name": "category", "type": "string", "nullable": True},
                {"name": "created_date", "type": "timestamp", "nullable": True},
                {"name": "status", "type": "string", "nullable": True},
            ]
        }
    ]
    print("Using fallback generic schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Generate Synthetic Data with Faker

# COMMAND ----------

def generate_column_value(col_def: dict, row_idx: int) -> object:
    """Generate a synthetic value for a column based on its type and name."""
    col_name = col_def["name"].lower()
    col_type = col_def.get("type", "string").lower()
    nullable = col_def.get("nullable", True)

    # Occasionally return None for nullable columns (10% chance)
    if nullable and random.random() < 0.1 and row_idx > 0:
        return None

    # Name-based heuristics for realistic data
    if "email" in col_name:
        return fake.email()
    if "phone" in col_name:
        return fake.phone_number()[:20]
    if col_name in ("first_name", "firstname"):
        return fake.first_name()
    if col_name in ("last_name", "lastname"):
        return fake.last_name()
    if col_name in ("name", "full_name", "fullname", "customer_name"):
        return fake.name()
    if "address" in col_name:
        return fake.address().replace("\n", ", ")
    if "city" in col_name:
        return fake.city()
    if "state" in col_name:
        return fake.state_abbr()
    if "zip" in col_name or "postal" in col_name:
        return fake.zipcode()
    if "country" in col_name:
        return fake.country_code()
    if "status" in col_name:
        return random.choice(["active", "inactive", "pending", "completed", "cancelled"])
    if "category" in col_name or "type" in col_name:
        return random.choice(["A", "B", "C", "D", "E"])
    if "description" in col_name or "comment" in col_name:
        return fake.sentence()

    # Type-based generation
    if col_type in ("int", "integer", "smallint"):
        if "id" in col_name:
            return row_idx + 1
        return random.randint(0, 10000)
    if col_type in ("bigint",):
        if "id" in col_name:
            return row_idx + 1
        return random.randint(0, 1000000)
    if col_type in ("decimal", "numeric", "number", "money"):
        precision = col_def.get("precision") or 10
        scale = col_def.get("scale") or 2
        max_val = 10 ** (precision - scale) - 1
        val = round(random.uniform(0, min(max_val, 100000)), scale)
        return float(val)
    if col_type in ("double", "float", "real"):
        return round(random.uniform(0, 10000), 4)
    if col_type in ("date",):
        return fake.date_between(start_date="-2y", end_date="today").isoformat()
    if col_type in ("timestamp", "datetime", "date/time"):
        return fake.date_time_between(start_date="-2y", end_date="now").isoformat()
    if col_type in ("boolean", "bool"):
        return random.choice([True, False])

    # Default: string
    if "id" in col_name and col_type == "string":
        return f"ID-{row_idx + 1:06d}"
    return fake.word() if random.random() > 0.3 else fake.sentence()[:50]


def generate_table_data(table_def: dict, row_count: int = 500) -> dict:
    """Generate synthetic data for a table definition."""
    columns = table_def.get("columns", [])
    data = {col["name"]: [] for col in columns}

    for i in range(row_count):
        for col in columns:
            data[col["name"]].append(generate_column_value(col, i))

    return data


# Generate data for source and lookup tables
source_tables = [t for t in tables if t["role"] in ("source", "lookup")]
source_data = {}

for table_def in source_tables:
    row_count = 500 if table_def["role"] == "source" else 50
    data = generate_table_data(table_def, row_count=row_count)
    source_data[table_def["table_name"]] = data
    print(f"  Generated {table_def['table_name']}: {row_count} rows, {len(table_def['columns'])} columns")

if not source_data:
    # If no source tables found, use first table regardless of role
    if tables:
        t = tables[0]
        data = generate_table_data(t, row_count=500)
        source_data[t["table_name"]] = data
        print(f"  Generated fallback: {t['table_name']}: 500 rows")

print(f"\nGenerated synthetic data for {len(source_data)} tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Synthetic Source Data (for Converted Runner)

# COMMAND ----------

# Write parquet via pandas+pyarrow directly — bypasses spark.createDataFrame entirely
import pandas as pd

os.makedirs(source_data_output_path, exist_ok=True)

for table_name, data_dict in source_data.items():
    pdf = pd.DataFrame(data_dict)
    parquet_path = f"{source_data_output_path}/{table_name}.parquet"
    pdf.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"  Saved {table_name} ({len(pdf)} rows) to {parquet_path}")

# Save schemas JSON for reference
schemas_json = json.dumps({"tables": tables}, indent=2)
schemas_path = f"{source_data_output_path}/schemas.json"
with open(schemas_path, "w") as f:
    f.write(schemas_json)
print(f"Saved schemas to {schemas_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Simulate Expected Output

# COMMAND ----------

# For simulation, we pass source data through as the "expected" output.
# The LLM conversion should produce equivalent results when run against the same source data.
# If destination tables are well-defined, use their schema; otherwise passthrough the source.

dest_tables = [t for t in tables if t["role"] == "destination"]
first_source = list(source_data.values())[0] if source_data else None
first_source_name = list(source_data.keys())[0] if source_data else None
source_col_count = len(first_source) if first_source else 0

use_destination_schema = False
if dest_tables and source_data:
    dest_table = dest_tables[0]
    dest_col_count = len(dest_table.get("columns", []))

    # Only use destination schema if it's reasonably complete.
    # If the LLM extracted <30% of the source columns for the destination,
    # it likely missed columns — fall back to source passthrough.
    if source_col_count > 0 and dest_col_count >= max(3, source_col_count * 0.3):
        use_destination_schema = True
        print(f"Destination schema looks complete: {dest_col_count} cols (source has {source_col_count})")
    else:
        print(f"Warning: Destination schema looks incomplete ({dest_col_count} cols vs {source_col_count} source cols)")
        print(f"  Falling back to source data passthrough for a meaningful comparison")

if use_destination_schema and source_data:
    # Generate expected output matching destination schema
    row_count = len(first_source[list(first_source.keys())[0]])
    expected_output = {}

    for col in dest_table["columns"]:
        col_name = col["name"]
        # If the column exists in source, use it; otherwise generate new data
        if col_name in first_source:
            expected_output[col_name] = first_source[col_name]
        else:
            expected_output[col_name] = [generate_column_value(col, i) for i in range(row_count)]

    print(f"Generated expected output matching destination schema: {dest_table['table_name']}")
    print(f"  Columns: {list(expected_output.keys())}")
    print(f"  Rows: {row_count}")

elif source_data:
    # Passthrough: expected output = source data (most reliable baseline comparison)
    expected_output = first_source
    row_count = len(first_source[list(first_source.keys())[0]])
    print(f"Using source table '{first_source_name}' as expected output (passthrough)")
    print(f"  Columns: {list(expected_output.keys())}")
    print(f"  Rows: {row_count}")

else:
    expected_output = None
    print("Warning: No data available for expected output")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save Expected Output

# COMMAND ----------

os.makedirs(output_path, exist_ok=True)

if expected_output:
    # Write expected output via pandas+pyarrow directly — bypasses spark.createDataFrame entirely
    exp_pdf = pd.DataFrame(expected_output)
    output_file = f"{output_path}/expected_output.parquet"
    exp_pdf.to_parquet(output_file, index=False, engine="pyarrow")
    print(f"Saved expected output to: {output_file}")
    print(f"  Rows: {len(exp_pdf)}")
    print(f"  Columns: {list(exp_pdf.columns)}")
    print(exp_pdf.head(10).to_string())
else:
    print("Warning: No expected output generated")

# COMMAND ----------

# Save metadata
completion_metadata = {
    "conversion_job_id": conversion_job_id,
    "source_type": source_type,
    "simulation_completed_at": datetime.now().isoformat(),
    "source_tables_used": list(source_data.keys()),
    "output_rows": len(expected_output[list(expected_output.keys())[0]]) if expected_output else 0,
    "output_columns": list(expected_output.keys()) if expected_output else [],
    "schemas_extracted": {
        "tables": len(tables),
        "source_tables": len([t for t in tables if t["role"] == "source"]),
        "destination_tables": len([t for t in tables if t["role"] == "destination"]),
        "lookup_tables": len([t for t in tables if t["role"] == "lookup"]),
    }
}

metadata_path = f"{output_path}/simulation_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(completion_metadata, f, indent=2)
print(f"Saved metadata to: {metadata_path}")

# COMMAND ----------

print(f"\n{'='*60}")
print("ORIGINAL OUTPUT SIMULATION COMPLETE")
print(f"{'='*60}")
print(f"Conversion Job ID: {conversion_job_id}")
print(f"Source Type: {source_type}")
print(f"Synthetic Data: {source_data_output_path}")
print(f"Expected Output: {output_path}/expected_output.parquet")
print(f"{'='*60}")
