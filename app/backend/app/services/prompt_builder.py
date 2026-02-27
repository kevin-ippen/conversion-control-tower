"""
Prompt Builder adapter.

Bridges the rich prompt engine (src/conversion/prompt_engine.py) and
configuration (config/conversion_config.yaml) to the app backend converter.

Replaces the hardcoded SSIS_SYSTEM_PROMPT and SQL_SYSTEM_PROMPT with
config-driven, example-aware prompts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Ensure src/ is importable from the app backend
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from src.config_loader import ConfigLoader, ConversionConfig
    from src.conversion.prompt_engine import PromptEngine, ConversionPrompt
    _HAS_SRC = True
except ImportError as e:
    logger.warning(f"Could not import src modules: {e}. Using fallback prompts.")
    _HAS_SRC = False


# T-SQL specific mappings (inspired by BrickMod conversion_prompts.yaml)
TSQL_CONVERSION_RULES = """
## T-SQL to Spark SQL Mappings

### Function Conversions
- `CONVERT(type, expr[, style])` -> `CAST(expr AS type)`
- `TRY_CONVERT/TRY_CAST` -> `TRY_CAST(expr AS type)`
- `ISNULL(a, b)` -> `COALESCE(a, b)`
- `GETDATE()` / `SYSDATETIME()` -> `CURRENT_TIMESTAMP()`
- `DATEADD(unit, n, date)` -> `date_add(date, n)` for days, `timestampadd(unit, n, date)` otherwise
- `DATEDIFF(unit, start, end)` -> `timestampdiff(unit, start, end)` or `datediff(end, start)` for days
- `FORMAT(ts, 'pat')` -> `date_format(ts, 'pat')`
- `IIF(cond, a, b)` -> `CASE WHEN cond THEN a ELSE b END`
- `TOP (n)` -> append `LIMIT n`

### Syntax Conversions
- `CROSS/OUTER APPLY` -> `LATERAL VIEW + explode()/posexplode()/inline()`
- Brackets `[schema].[table]` -> backticks or dots
- Inline table variables (`DECLARE @t TABLE(...)`) -> temp views

### Data Type Mappings
- `VARCHAR/NVARCHAR/TEXT` -> `STRING`
- `DATETIME/DATETIME2/SMALLDATETIME` -> `TIMESTAMP`
- `MONEY/SMALLMONEY` -> `DECIMAL(19,4)`
- `VARBINARY/IMAGE` -> `BINARY`
- `BIT` -> `BOOLEAN`
- `INT/BIGINT/SMALLINT/TINYINT` -> same names in Spark

### Stored Procedure Conversions
- Convert `@parameter` to `dbutils.widgets.get('parameter')`
- Replace `SET NOCOUNT ON` (not needed in Databricks)
- Convert `BEGIN...END` blocks to Python control flow
- Convert `RAISERROR` / `THROW` to Python `raise Exception()`
- Convert `TRY/CATCH` to Python `try/except`
- Convert temp tables (`#temp`) to CTEs or temp views
- Convert cursors to set-based DataFrame operations
- Convert `EXEC sp_executesql` to `spark.sql()`
"""


# JSON output schema for SSIS conversions
SSIS_OUTPUT_SCHEMA = """\
Your response must be valid JSON with this exact structure:
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
      "code": "# PySpark code here..."
    }
  ],
  "quality_notes": {
    "score": 0.85,
    "warnings": ["Any conversion warnings"],
    "manual_review": ["Items needing manual review"]
  }
}"""

# JSON output schema for SQL conversions
SQL_OUTPUT_SCHEMA = """\
Your response must be valid JSON with this exact structure:
{
  "notebooks": [
    {
      "name": "converted_sql.py",
      "description": "What this notebook does",
      "code": "# PySpark/Spark SQL code here..."
    }
  ],
  "quality_notes": {
    "score": 0.9,
    "warnings": ["Any conversion warnings"],
    "manual_review": ["Items needing manual review"]
  }
}"""


# Informatica PowerCenter function → Spark SQL mappings (from POV doc)
INFORMATICA_CONVERSION_RULES = """
## Informatica PowerCenter to Spark SQL Mappings

### Expression Function Conversions
- `IIF(cond, true, false)` -> `CASE WHEN cond THEN true ELSE false END` (nested IIF chains are common)
- `DECODE(val, match1, res1, ..., default)` -> `CASE WHEN val = match1 THEN res1 ... ELSE default END`
- `IS_SPACES(x)` -> `TRIM(x) = ''`
- `LTRIM/RTRIM(x, chars)` -> `LTRIM(x, chars)` / `RTRIM(x, chars)` (Spark supports trim chars natively)
- `TO_DATE(x, fmt)` -> `TO_DATE(x, fmt)` (format string syntax differs: Informatica vs Java)
- `TO_CHAR(date, fmt)` -> `DATE_FORMAT(date, fmt)`
- `TO_INTEGER(x)` -> `CAST(x AS INT)`
- `TO_DECIMAL(x, s)` -> `CAST(x AS DECIMAL(p,s))`
- `SYSDATE` / `SYSTIMESTAMP` -> `CURRENT_TIMESTAMP()`
- `ADD_TO_DATE(date, fmt, amt)` -> `date + INTERVAL amt fmt` or `DATE_ADD(date, amt)`
- `DATE_DIFF(d1, d2, fmt)` -> `DATEDIFF(d1, d2)` (day-level only; others need manual calc)
- `LPAD/RPAD(x, n, c)` -> `LPAD(x, n, c)` / `RPAD(x, n, c)` (direct match)
- `INSTR(x, search, start, occ)` -> `LOCATE(search, x, start)` (Spark LOCATE lacks occurrence)
- `SUBSTR(x, start, len)` -> `SUBSTR(x, start, len)` (1-indexed in both)
- `REG_EXTRACT(x, pat, grp)` -> `REGEXP_EXTRACT(x, pat, grp)`
- `REG_REPLACE(x, pat, rep)` -> `REGEXP_REPLACE(x, pat, rep)`
- `:LKP.xxx(args)` -> LEFT JOIN (requires structural conversion)
- `ERROR(msg)` -> `ASSERT_TRUE(FALSE, msg)` or custom UDF
- `ABORT(msg)` -> `dbutils.notebook.exit(msg)`
- `$$param` -> `${param}` widget reference or config lookup
- `$PMFolderName`, etc. -> Hardcode or parameterize (built-in session variables)

### Transformation → Spark Patterns
- **Source Qualifier** -> `spark.sql("CREATE OR REPLACE TEMP VIEW ... AS SELECT ... FROM ... WHERE <source_filter>")`
  - If `Sql Query` override is set, use it verbatim (after expression normalization)
  - If `Source Filter` is set, append as WHERE clause
  - If `User Defined Join` is set, use in FROM for multi-source SQs
- **Expression** -> `withColumn()` calls or CTE expressions
  - Variable ports (V flag) are intermediate — inline or pre-compute as CTEs, NOT in output SELECT
  - Output ports (O flag) are the actual output columns
- **Filter** -> `.filter()` or `WHERE` clause using FILTERCONDITION
- **Joiner** -> `.join()` using Join Condition, Join Type (Normal=INNER, Master Outer=LEFT, Detail Outer=RIGHT, Full=FULL)
- **Lookup** -> LEFT JOIN (connected) or correlated subquery (unconnected `:LKP.xxx()`)
  - Multi-match policy: "Use First Value"/"Use Last Value" → ROW_NUMBER + QUALIFY
- **Router** -> Multiple `.filter()` branches per output group
  - DEFAULT group: records not matched by any named group
- **Aggregator** -> `.groupBy().agg()` using GroupBy-flagged ports
  - If Sorted Input is checked, verify sort key matches group key
- **Rank** -> `Window` + `row_number()` / `rank()` / `dense_rank()`
  - Top/Bottom setting determines ASC/DESC
- **Union** -> `.union()` across input groups
- **Normalizer** -> `STACK(N, col1, ..., colN)` where N = Occurs value
- **Sequence Generator** -> `monotonically_increasing_id()` or `row_number()` window
- **Sorter** -> `.orderBy()` using sort key fields and direction

### Data Type Mappings (Informatica → Spark)
- `String/Varchar/Nvarchar/Text` -> `STRING`
- `Integer/Small Integer` -> `INT` / `SMALLINT`
- `Bigint` -> `BIGINT`
- `Double/Float/Real` -> `DOUBLE` / `FLOAT`
- `Decimal(p,s)` -> `DECIMAL(p,s)` (watch for `decimal(28,0)` edge cases)
- `Date/Time` -> `DATE`
- `Date/Time (with time)` -> `TIMESTAMP`
- `Binary` -> `BINARY`

### Workflow → Databricks Workflow Patterns
- Sequential workflow links → task dependencies
- Decision tasks → conditional task logic
- Parallel paths → fan-out from parent task
- Session → one notebook per mapping execution
- Pre-session SQL → separate setup cell or notebook
- Post-session SQL → separate cleanup cell or notebook

### Code Generation Structure
- Cell 1: Parameters (dbutils.widgets from mapping parameters/variables)
- Cell 2: Schema/database selection
- Cells 3-N: One cell per transformation in topological order
  - Comment header with component name, type, original transformation name
  - SQL wrapped in `spark.sql(\"\"\"...\"\"\")`
  - Register as temp view
- Cell N+1: Target write (INSERT INTO / MERGE INTO / CREATE TABLE AS SELECT)
- Cell N+2: Post-session SQL (if any)

### Common Pitfalls to Avoid
- Session-level overrides (SESSTRANSFORMATIONINST) can change mapping behavior — always check
- Reusable transformations may reference shared objects — resolve references
- Variable ports execute in port order (top to bottom); later variables can reference earlier ones
- Unconnected Lookups (`:LKP.xxx()` in expressions) need correlated LEFT JOIN conversion
"""

# JSON output schema for Informatica conversions
INFORMATICA_OUTPUT_SCHEMA = """\
Your response must be valid JSON with this exact structure:
{
  "workflow": {
    "name": "workflow_name",
    "description": "Brief description of the converted workflow",
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
      "name": "mapping_name.py",
      "description": "Converted from Informatica mapping: mapping_name",
      "code": "# PySpark code here, one cell per transformation in topological order..."
    }
  ],
  "quality_notes": {
    "score": 0.85,
    "warnings": ["Any conversion warnings"],
    "manual_review": ["Items needing manual review"],
    "unconverted": ["Any Informatica elements not converted (e.g., :LKP residue, ERROR() calls)"]
  }
}"""


class PromptBuilder:
    """Builds conversion prompts using config and examples."""

    def __init__(self, config_dir: Optional[Path] = None):
        self._config: Optional[ConversionConfig] = None
        self._engine: Optional[Any] = None
        self._config_dir = config_dir or _project_root / "config"

        if _HAS_SRC:
            try:
                loader = ConfigLoader(self._config_dir)
                self._config = loader.load_config()
                self._engine = PromptEngine(self._config)
                logger.info(
                    f"PromptBuilder initialized with config '{self._config.config_name}', "
                    f"{len(self._config.examples)} examples"
                )
            except Exception as e:
                logger.warning(f"Failed to load conversion config: {e}")

    def build_system_prompt(
        self,
        source_type: str,
        output_format: str = "pyspark",
        features: Optional[Dict[str, bool]] = None,
    ) -> str:
        """Build system prompt for a conversion.

        Args:
            source_type: ssis, sql_script, stored_proc
            output_format: pyspark, dlt, dbt
            features: Optional feature flags from UI checkboxes

        Returns:
            Complete system prompt string
        """
        parts = []

        # Core identity
        parts.append(
            "You are an expert data engineer specializing in migrating "
            "SQL Server, SSIS, and Informatica PowerCenter workloads to Databricks."
        )
        parts.append("")

        # Config-driven global instructions
        if self._config and self._config.global_instructions:
            parts.append("## Global Conversion Rules")
            parts.append(self._config.global_instructions)
            parts.append("")

        # Source-type-specific conversion rules
        if source_type in ("sql_script", "stored_proc", "ssis"):
            parts.append(TSQL_CONVERSION_RULES)
            parts.append("")
        elif source_type == "informatica_pc":
            parts.append(INFORMATICA_CONVERSION_RULES)
            parts.append("")

        # Config-driven data type mappings
        if self._config and self._config.data_type_mappings:
            parts.append("## Data Type Mappings")
            for source_type_name, target_type in self._config.data_type_mappings.items():
                parts.append(f"- `{source_type_name}` -> `{target_type}`")
            parts.append("")

        # Output format specific instructions
        if output_format == "dlt":
            parts.append("## Output Format: Databricks DLT/SDP")
            parts.append("- Generate DLT pipeline code using `CREATE OR REFRESH STREAMING TABLE` syntax")
            parts.append("- Use `@dlt.table` decorators for Python DLT")
            parts.append("- Map data quality checks to DLT expectations")
            parts.append("- Use `dlt.read()` or `dlt.read_stream()` for reading")
            parts.append("")
        elif output_format == "dbt":
            parts.append("## Output Format: dbt")
            parts.append("- Generate dbt model SQL files with `{{ config() }}`, `{{ ref() }}`, `{{ source() }}` macros")
            parts.append("- Use `dbt-databricks` adapter configuration patterns")
            parts.append("- Include `sources.yml` definition")
            parts.append("- Use staging/marts layering pattern")
            parts.append("")
        else:
            parts.append("## Output Format: PySpark Notebooks")
            parts.append("- Use Delta Lake for all tables")
            parts.append("- Use Unity Catalog three-level namespace (catalog.schema.table)")
            parts.append("- Use spark.read.format('delta') for reading")
            parts.append("- Use df.write.format('delta').mode('overwrite').saveAsTable() for writing")
            parts.append("- Convert T-SQL to Spark SQL where possible")
            parts.append("- Map SSIS data flows to PySpark DataFrame operations")
            parts.append("")

        # Feature-flag driven instructions
        if features:
            feature_instructions = self._build_feature_instructions(features)
            if feature_instructions:
                parts.append("## Feature Requirements")
                parts.append(feature_instructions)
                parts.append("")

        # Component-specific instructions from config
        if source_type == "ssis" and self._config:
            for comp_type, instructions in self._config.component_instructions.items():
                parts.append(f"### {comp_type} Handling")
                parts.append(instructions)
                parts.append("")

        # Quality requirements
        parts.append("## Quality Requirements")
        parts.append("- Generated code must be syntactically valid Python")
        parts.append("- No hardcoded credentials - use `dbutils.secrets.get(scope, key)`")
        parts.append("- Use Unity Catalog three-part naming for all table references")
        parts.append("- Use `dbutils.widgets` for configurable parameters")
        parts.append("- Include error handling (try/except) for production readiness")
        parts.append("- Add logging/print statements for row counts and metrics")
        if self._config:
            parts.append(f"- Target Databricks Runtime: {self._config.runtime_version}")
        parts.append("")

        # JSON output schema
        if source_type == "ssis":
            parts.append(SSIS_OUTPUT_SCHEMA)
        elif source_type == "informatica_pc":
            parts.append(INFORMATICA_OUTPUT_SCHEMA)
        else:
            parts.append(SQL_OUTPUT_SCHEMA)

        return "\n".join(parts)

    def build_user_prompt(
        self,
        source_content: str,
        source_type: str,
        conversion_instructions: str = "",
    ) -> str:
        """Build user prompt for a conversion.

        Args:
            source_content: The source code/XML to convert
            source_type: ssis, sql_script, stored_proc
            conversion_instructions: Additional user-provided instructions

        Returns:
            User prompt string
        """
        if source_type == "informatica_pc":
            prompt = (
                "Convert this Informatica PowerCenter export to Databricks.\n\n"
                "The structured context below was extracted from the PowerCenter XML "
                "and describes the mappings, transformations, connectors, and workflows.\n\n"
                f"{source_content}\n\n"
                "Important:\n"
                "- Generate one notebook per mapping\n"
                "- Follow topological order for transformations\n"
                "- Convert $$parameters to dbutils.widgets\n"
                "- Convert :LKP references to LEFT JOINs\n"
                "- Use ROW_NUMBER + QUALIFY for multi-match lookup policies\n"
                "- Variable ports (V flag) are intermediate — inline or use CTEs\n"
                "- Preserve original transformation names as comments for traceability"
            )
        elif source_type == "ssis":
            prompt = f"Convert this SSIS package to Databricks:\n\n```xml\n{source_content}\n```"
        elif source_type == "stored_proc":
            prompt = (
                f"Convert this SQL Server stored procedure to Databricks:\n\n"
                f"```sql\n{source_content}\n```\n\n"
                "Important:\n"
                "- Convert @parameters to dbutils.widgets\n"
                "- Replace cursors with set-based operations\n"
                "- Convert temp tables to CTEs or views\n"
                "- Convert TRY/CATCH to try/except\n"
                "- Remove SET NOCOUNT ON and similar T-SQL directives"
            )
        else:
            prompt = f"Convert this SQL Server code to Databricks:\n\n```sql\n{source_content}\n```"

        if conversion_instructions:
            prompt += f"\n\n## Additional Instructions\n{conversion_instructions}"

        return prompt

    def _build_feature_instructions(self, features: Dict[str, bool]) -> str:
        """Build prompt instructions based on feature flags."""
        instructions = []

        if features.get("delta_optimize"):
            instructions.append("- Add `OPTIMIZE` statements after large writes")
        if features.get("liquid_clustering"):
            instructions.append("- Use `CLUSTER BY` for liquid clustering on frequently filtered columns")
        if features.get("error_quarantine"):
            instructions.append(
                "- Route invalid/error records to a quarantine table "
                "(catalog.schema.tablename_quarantine)"
            )
        if features.get("error_logging"):
            instructions.append("- Log errors to a structured error table with timestamp, error_type, message, record_id")
        if features.get("dq_expectations"):
            instructions.append("- Add data quality checks (null checks, type validation, range validation)")
        if features.get("row_count_logging"):
            instructions.append(
                "- Log row counts before and after each transformation "
                "using print() statements"
            )
        if features.get("timing_metrics"):
            instructions.append("- Add timing metrics using `import time; start = time.time()` around major operations")
        if features.get("auto_partition"):
            instructions.append("- Partition large tables by date columns where appropriate")

        return "\n".join(instructions)
