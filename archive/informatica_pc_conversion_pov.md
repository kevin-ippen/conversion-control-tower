# Informatica PowerCenter → Databricks: LLM-Agentic Conversion POV

## Context

This document provides guidance for Claude Code agentic workflows that parse, normalize, modularize, validate, and simulate Informatica PowerCenter XML exports for automated conversion to Databricks notebooks (Python + SQL / PySpark). It is intended as a system-level reference that can be injected into agent prompts or used as a planning doc for multi-step orchestration.

---

## 1. Parsing PowerCenter XML

PowerCenter repository exports follow a well-defined XML schema rooted in `<POWERMART>`. Understanding the hierarchy is critical before any transformation logic runs.

### Document Structure

```
POWERMART
└── REPOSITORY
    └── FOLDER
        ├── SOURCE (table/file definitions)
        ├── TARGET (table/file definitions)
        ├── MAPPING
        │   ├── TRANSFORMATION (SQ, EXP, FIL, JNR, RTR, AGG, RNK, LKP, UNI, NRM, SEQ, SRT...)
        │   │   ├── TRANSFORMFIELD (ports with I/O/V/R flags, expressions, defaults)
        │   │   └── TABLEATTRIBUTE (transformation-level config: join type, filter condition, etc.)
        │   ├── CONNECTOR (wiring between transformation ports)
        │   └── MAPPINGVARIABLE / MAPPING_PARAMETER
        ├── MAPPLET (reusable sub-graphs, same internal structure as MAPPING)
        ├── SESSION
        │   ├── SESSIONEXTENSION (reader/writer configs)
        │   ├── ATTRIBUTE (pre/post SQL, commit intervals, etc.)
        │   └── SESSTRANSFORMATIONINST (per-transformation session overrides)
        └── WORKFLOW
            ├── TASK (sessions, commands, decisions, timers)
            └── WORKFLOWLINK (DAG edges between tasks)
```

### Parsing Strategy

1. **Use an XML parser that preserves attribute order and CDATA** — `lxml.etree` is preferred over `xml.etree.ElementTree` for XPath support and robustness with large files.
2. **Build an in-memory graph model early.** Each `TRANSFORMATION` is a node; each `CONNECTOR` is a directed edge (FROMINSTANCE/FROMFIELD → TOINSTANCE/TOFIELD). This graph is the backbone of everything downstream.
3. **Handle multi-file exports.** Customers often export one XML per folder, or one massive file. Your parser should accept both a single file and a directory of files, merging into a unified repository model.
4. **Extract metadata aggressively.** Every `TABLEATTRIBUTE` and `TRANSFORMFIELD` attribute matters. Key attributes by transformation type:

| Transformation Type | Critical Attributes |
|---|---|
| Source Qualifier (SQ) | `Sql Query`, `Source Filter`, `Number Of Sorted Ports`, `User Defined Join` |
| Expression (EXP) | `EXPRESSION` on each TRANSFORMFIELD |
| Filter (FIL) | `FILTERCONDITION` TABLEATTRIBUTE |
| Joiner (JNR) | `Join Type`, `Join Condition` (TABLEATTRIBUTE), master/detail flag on fields |
| Lookup (LKP) | `Lookup table name`, `Lookup Source Filter`, `Lookup policy on multiple match`, `Connection Information`, lookup condition fields |
| Router (RTR) | `Group Filter Condition` per output group |
| Aggregator (AGG) | `GroupBy` flag on TRANSFORMFIELD |
| Rank (RNK) | `Top/Bottom`, `Number of Ranks`, rank port `R` flag |
| Union (UNI) | Group/port mapping across input groups |
| Normalizer (NRM) | `Level`, `Occurs`, `Key Type` on fields |
| Sequence Generator (SEQ) | `Start Value`, `End Value`, `Increment By`, `Current Value` |
| Sorter (SRT) | sort key fields, `Sorter Direction` |

---

## 2. Normalization

Raw parsed XML is verbose and inconsistent. Normalize before conversion.

### Expression Normalization

PowerCenter expressions use a proprietary function library. Map these to Spark SQL equivalents:

| Informatica Function | Spark SQL Equivalent | Notes |
|---|---|---|
| `IIF(cond, true, false)` | `CASE WHEN cond THEN true ELSE false END` | Nested IIF chains are common |
| `DECODE(val, match1, res1, ..., default)` | `CASE WHEN val = match1 THEN res1 ... ELSE default END` | |
| `IS_SPACES(x)` | `TRIM(x) = ''` | |
| `LTRIM/RTRIM(x, chars)` | `LTRIM(x, chars)` / `RTRIM(x, chars)` | Spark supports trim chars natively |
| `TO_DATE(x, fmt)` | `TO_DATE(x, fmt)` | Format string syntax differs (Informatica vs Java) |
| `TO_CHAR(date, fmt)` | `DATE_FORMAT(date, fmt)` | |
| `TO_INTEGER(x)` | `CAST(x AS INT)` | |
| `TO_DECIMAL(x, s)` | `CAST(x AS DECIMAL(p,s))` | |
| `SYSDATE` | `CURRENT_TIMESTAMP()` | |
| `SYSTIMESTAMP` | `CURRENT_TIMESTAMP()` | |
| `ADD_TO_DATE(date, fmt, amt)` | `date + INTERVAL amt fmt` or `DATE_ADD(date, amt)` | Depends on granularity |
| `DATE_DIFF(d1, d2, fmt)` | `DATEDIFF(d1, d2)` | Only day-level; others need manual calc |
| `LPAD/RPAD(x, n, c)` | `LPAD(x, n, c)` / `RPAD(x, n, c)` | Direct match |
| `INSTR(x, search, start, occ)` | `LOCATE(search, x, start)` | Spark LOCATE doesn't support occurrence |
| `SUBSTR(x, start, len)` | `SUBSTR(x, start, len)` | Direct match (1-indexed in both) |
| `REG_EXTRACT(x, pat, grp)` | `REGEXP_EXTRACT(x, pat, grp)` | |
| `REG_REPLACE(x, pat, rep)` | `REGEXP_REPLACE(x, pat, rep)` | |
| `LOOKUP(...)` / `:LKP.xxx(args)` | LEFT JOIN (see unconnected lookup pattern) | Requires structural conversion |
| `ERROR(msg)` | Custom UDF or `ASSERT_TRUE(FALSE, msg)` | Semantics differ |
| `ABORT(msg)` | `dbutils.notebook.exit(msg)` | Workflow-level |
| `$$param` | `${param}` widget reference or config lookup | |
| `$PMFolderName`, etc. | Hardcode or parameterize | Built-in session variables |

### Port Normalization

- Ports flagged `V` (variable) are intermediate calculations — they should NOT appear in output SELECT lists but their expressions must be inlined or pre-computed as CTEs.
- Ports flagged `O` (output) are the actual output columns.
- Ports flagged `R` (rank) identify the ranking column in Rank transformations.
- The `GroupBy` attribute on Aggregator ports determines GROUP BY columns vs aggregate expressions.

### Naming Normalization

- Strip common prefixes: `m_`, `map_`, `exp_`, `fil_`, `lkp_`, `jnr_`, `rtr_`, `agg_`, `rnk_`, `srt_`, `sq_`, `SQ_`
- Lowercase and sanitize for Delta table/view names
- Preserve original names as comments in generated code for traceability

---

## 3. Modularization

### Graph Traversal Strategy

The connector graph defines the execution plan. Walk it **topologically** (sources → targets):

1. Start from Source Qualifier nodes (they have no upstream transformation inputs, only SOURCE connections).
2. For each node in topological order, generate the corresponding SQL/PySpark block.
3. Each transformation outputs either a temp view (`createOrReplaceTempView`) or a DataFrame variable.

### Code Generation Modules

Structure the output notebook with clear cell boundaries:

```
Cell 1: Parameters (dbutils.widgets from mapping parameters/variables)
Cell 2: Schema/database selection (USE schema)
Cell 3-N: One cell per transformation in topological order
  - Comment header: component name, type, original transformation name
  - SQL statement wrapped in spark.sql("""...""")
  - Register as temp view
Cell N+1: Target write (INSERT INTO / MERGE INTO / CREATE TABLE AS SELECT)
Cell N+2: Post-session SQL (if any)
```

### When to Split Notebooks

- If a mapping has independent branches (no shared intermediate state), consider splitting into parallel notebooks orchestrated by a parent.
- If a mapping exceeds ~50 transformation steps, consider breaking into sub-notebooks with temp table handoffs.
- Mapplets should become reusable Python functions or shared notebooks invoked via `%run` or `dbutils.notebook.run()`.

---

## 4. Validation

### Static Validation (Pre-Execution)

Run these checks on generated code before it ever hits a cluster:

1. **Column lineage continuity** — every column referenced in a downstream transformation must exist in an upstream SELECT or view. Walk the graph and verify.
2. **Expression syntax** — parse all generated SQL expressions with `sqlparse` or Spark's `spark.sql(f"EXPLAIN {query}")` to catch syntax errors.
3. **Join condition completeness** — every Joiner/Lookup must produce a valid ON clause. Flag any with empty conditions.
4. **Aggregate/GroupBy consistency** — non-aggregated columns in SELECT must appear in GROUP BY.
5. **Type compatibility** — cross-reference source/target DDL types with expression output types. Flag potential truncation or overflow.
6. **Unresolved references** — search for `$$` (unconverted parameters), `:LKP.` (unconverted lookups), or any Informatica-specific syntax that leaked through.
7. **Widget completeness** — every `dbutils.widgets.get("X")` should have a corresponding `dbutils.widgets.text("X", default)` in the init cell.

### Semantic Validation (Post-Execution)

1. **Row count reconciliation** — run source count vs target count for each mapping. Account for filters and aggregations.
2. **Sample data diff** — compare N random rows between legacy output and Databricks output on key columns.
3. **Null handling parity** — Informatica's NULL handling differs from Spark SQL in some edge cases (e.g., NULL in string concatenation). Test explicitly.
4. **Sort stability** — Rank and Sorter conversions may produce different row ordering for ties. Validate that tie-breaking is deterministic.

---

## 5. Simulation / Dry-Run

### Mock Execution Strategy

For large migrations, you don't want to run every converted notebook against production data to verify correctness. Instead:

1. **Schema-only mode** — create empty Delta tables matching source schemas. Run notebooks to verify they parse and compile without data errors. Use `LIMIT 0` subqueries or `CREATE TABLE ... AS SELECT * FROM source WHERE 1=0`.

2. **Sample data mode** — extract 1000 representative rows per source table (including edge cases: NULLs, boundary dates, max-length strings, special characters). Load into test tables. Run notebooks and compare outputs against expected results generated from the original Informatica execution.

3. **Expression unit testing** — for complex expression transformations, generate a test harness:
   ```python
   # For each expression port, generate:
   test_df = spark.sql("""
     SELECT 
       <original_input_columns>,
       <converted_expression> AS actual_result,
       <expected_value> AS expected_result
     FROM test_data
   """)
   assert test_df.filter("actual_result != expected_result").count() == 0
   ```

4. **DAG simulation** — for workflow conversions, validate the Airflow DAG structure:
   - Parse the generated Python DAG file
   - Verify task dependencies match the original workflow links
   - Check that conditional logic (decision tasks) maps correctly to Airflow branching operators

---

## 6. Transformation-Specific Conversion Patterns

### Source Qualifier → SQL View

```python
# Component: SQ_Employees, Type: SOURCE
SQ_Employees = spark.sql("""
  CREATE OR REPLACE TEMPORARY VIEW SQ_Employees AS
  SELECT SRC.*, row_number() over (order by 1) AS source_record_id
  FROM (
    SELECT col1, col2, ...
    FROM schema.table
    WHERE <source_filter>
  ) SRC
""")
```

Key considerations:
- If `Sql Query` is overridden, use it verbatim (after expression normalization).
- If `Source Filter` is set, append as WHERE clause.
- If `User Defined Join` is set, use it in the FROM clause for multi-source SQs.
- Always add `source_record_id` via `row_number()` for downstream ordering consistency.

### Router → Multiple Temp Views

Each output group becomes its own temp view with the group's filter condition as a WHERE clause. The DEFAULT group gets all records not matched by any named group (use `NOT (cond1 OR cond2 OR ...)` or handle via EXCEPT).

### Unconnected Lookup → Correlated LEFT JOINs

Each `:LKP.lookup_name(arg1, arg2)` call in an expression becomes a LEFT JOIN. Multiple LKP calls in the same expression = multiple LEFT JOINs. Use CTEs or subqueries with ROW_NUMBER + QUALIFY to handle multi-match policies.

### Normalizer → STACK

```sql
SELECT 
  base_cols...,
  STACK(N, col1, col2, ..., colN) AS (normalized_col)
FROM upstream_view
```

Where N = the `Occurs` value from the Normalizer definition.

---

## 7. Workflow → Airflow DAG Patterns

```python
from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime

with DAG('wf_<workflow_name>', start_date=datetime(2024,1,1), schedule_interval=None) as dag:
    start = DummyOperator(task_id='start')
    
    # One task per session in the workflow
    s_task_1 = DatabricksRunNowOperator(
        task_id='s_<session_name>',
        databricks_conn_id='databricks_default',
        job_id='<job_id>',  # or notebook_task with notebook_path
        notebook_params={...}
    )
    
    start >> s_task_1 >> ...
```

Map workflow link types:
- Sequential links → `>>` operator
- Decision tasks → `BranchPythonOperator`
- Parallel paths → fan-out from parent task
- Timer/Event waits → `TimeDeltaSensor` or `ExternalTaskSensor`

---

## 8. Agent Orchestration Tips

When building multi-step agentic workflows in Databricks for this conversion:

1. **Step 1 — Inventory**: Parse all XMLs, build a catalog of mappings/workflows with complexity scores (count of transformations, lookup depth, router branches).
2. **Step 2 — Prioritize**: Sort by complexity. Convert simple mappings first to build confidence and catch config issues early.
3. **Step 3 — Convert in batches**: Group related mappings (same source/target tables) and convert together so temp view naming doesn't collide.
4. **Step 4 — Validate incrementally**: Run static validation after each batch. Don't wait until the end.
5. **Step 5 — Human-in-the-loop**: Flag any mapping with unconverted elements (`:LKP` residue, `ERROR()` calls, stored procedure references) for manual review. Generate a clear report of what was and wasn't converted per mapping.

### Complexity Scoring Heuristic

```
score = (
    num_transformations * 1
    + num_lookups * 2
    + num_unconnected_lookups * 3
    + num_router_groups * 2
    + num_expression_ports_with_nested_IIF * 1.5
    + (1 if has_mapplet_reference else 0) * 5
    + num_mapping_variables * 0.5
)
# Low: < 10, Medium: 10-30, High: 30+
```

---

## 9. Common Pitfalls

- **Session-level overrides**: A session can override connection info, SQL overrides, and even transformation properties. Always check `SESSTRANSFORMATIONINST` for overrides that change the mapping's behavior.
- **Reusable transformations**: These reference shared objects. The XML may include them inline or reference them by name. Ensure your parser resolves these references.
- **Multi-group Lookup**: The `Lookup policy on multiple match` setting is frequently overlooked. If set to "Use First Value" or "Use Last Value", you need ROW_NUMBER + QUALIFY in the converted join.
- **Variable ports execution order**: In Expression transformations, variable ports execute in port order (top to bottom). Later variables can reference earlier ones. Respect this ordering when inlining expressions.
- **Aggregator sorted input**: If `Sorted Input` is checked, the aggregator expects pre-sorted data and processes groups in sequence. In Spark SQL with GROUP BY, this doesn't matter — but validate that the sort key matches the group key.
- **Data type mismatches**: Informatica's `decimal(28,0)` maps differently than you'd expect in some edge cases. Always generate explicit CASTs in the output.
