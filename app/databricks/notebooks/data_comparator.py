# Databricks notebook source
# MAGIC %md
# MAGIC # Output Data Comparator
# MAGIC
# MAGIC Compares expected output (from original simulator) with actual output (from converted code).
# MAGIC Generates a detailed validation report and updates the conversion job quality metrics.
# MAGIC
# MAGIC **Parameters:**
# MAGIC - `conversion_job_id`: Original conversion job ID
# MAGIC - `expected_path`: UC Volume path to expected output
# MAGIC - `actual_path`: UC Volume path to actual output
# MAGIC - `output_path`: UC Volume path for comparison report

# COMMAND ----------

# Get parameters
dbutils.widgets.text("conversion_job_id", "")
dbutils.widgets.text("expected_path", "")
dbutils.widgets.text("actual_path", "")
dbutils.widgets.text("output_path", "")
dbutils.widgets.text("catalog", "dev_conversion_tracker")
dbutils.widgets.text("pass_threshold", "0.95")

conversion_job_id = dbutils.widgets.get("conversion_job_id")
expected_path = dbutils.widgets.get("expected_path")
actual_path = dbutils.widgets.get("actual_path")
output_path = dbutils.widgets.get("output_path")
catalog = dbutils.widgets.get("catalog")
pass_threshold = float(dbutils.widgets.get("pass_threshold"))

print(f"Conversion Job ID: {conversion_job_id}")
print(f"Expected Path: {expected_path}")
print(f"Actual Path: {actual_path}")
print(f"Output Path: {output_path}")
print(f"Pass Threshold: {pass_threshold}")

# COMMAND ----------

import json
import os
import glob
import uuid
import pandas as pd
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load expected output via pandas (bypass Spark)
expected_df = None
try:
    expected_files = glob.glob(f"{expected_path}/*.parquet")
    if expected_files:
        expected_df = pd.read_parquet(expected_files[0], engine="pyarrow")
        print(f"Loaded expected output: {len(expected_df)} rows, {len(expected_df.columns)} columns")
    else:
        print("Warning: No expected output parquet files found")
except Exception as e:
    print(f"Warning: Could not load expected output: {e}")

# Load actual output via pandas (bypass Spark)
actual_df = None
try:
    actual_files = glob.glob(f"{actual_path}/*.parquet")
    if actual_files:
        actual_df = pd.read_parquet(actual_files[0], engine="pyarrow")
        print(f"Loaded actual output: {len(actual_df)} rows, {len(actual_df.columns)} columns")
    else:
        print("Warning: No actual output parquet files found")
except Exception as e:
    print(f"Warning: Could not load actual output: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Comparison

# COMMAND ----------

# Inline data comparison logic (replaces src.validation.data_validator)

class ComparisonResult:
    """Container for comparison results."""
    def __init__(self):
        self.overall_pass = False
        self.summary = ""
        self.detailed_summary = ""
        self.schema_match = False
        self.row_count_match = False
        self.row_count_expected = 0
        self.row_count_actual = 0
        self.match_rate = 0.0
        self.total_cells = 0
        self.matching_cells = 0
        self.missing_columns = []
        self.extra_columns = []
        self.type_mismatches = []
        self.mismatched_columns = {}
        self.mismatched_rows = []

    def to_json(self):
        return json.dumps({
            "overall_pass": self.overall_pass,
            "summary": self.summary,
            "detailed_summary": self.detailed_summary,
            "schema_match": self.schema_match,
            "row_count_match": self.row_count_match,
            "row_count_expected": self.row_count_expected,
            "row_count_actual": self.row_count_actual,
            "match_rate": self.match_rate,
            "total_cells": self.total_cells,
            "matching_cells": self.matching_cells,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "type_mismatches": self.type_mismatches,
            "mismatched_columns": self.mismatched_columns,
            "mismatched_rows": self.mismatched_rows[:100],
        }, indent=2)


def _values_match(expected, actual, float_tolerance=0.0001, case_sensitive=False):
    """Compare two values with tolerance for floats and optional case insensitivity."""
    # Both None
    if expected is None and actual is None:
        return True
    # One is None
    if expected is None or actual is None:
        # Treat NaN as None equivalent
        import math
        if expected is None and isinstance(actual, float) and math.isnan(actual):
            return True
        if actual is None and isinstance(expected, float) and math.isnan(expected):
            return True
        return False
    # Both numeric
    try:
        e_float = float(expected)
        a_float = float(actual)
        import math
        if math.isnan(e_float) and math.isnan(a_float):
            return True
        if math.isnan(e_float) or math.isnan(a_float):
            return False
        return abs(e_float - a_float) <= float_tolerance
    except (ValueError, TypeError):
        pass
    # String comparison
    e_str = str(expected)
    a_str = str(actual)
    if not case_sensitive:
        return e_str.lower() == a_str.lower()
    return e_str == a_str


def compare_dataframes(expected_pdf, actual_pdf, float_tolerance=0.0001, case_sensitive=False, max_mismatches=100):
    """Compare two pandas DataFrames and return a ComparisonResult."""
    result = ComparisonResult()

    expected_cols = set(expected_pdf.columns)
    actual_cols = set(actual_pdf.columns)

    result.missing_columns = sorted(expected_cols - actual_cols)
    result.extra_columns = sorted(actual_cols - expected_cols)
    common_cols = sorted(expected_cols & actual_cols)

    result.schema_match = len(result.missing_columns) == 0 and len(result.extra_columns) == 0

    # Check types for common columns
    for col in common_cols:
        e_dtype = str(expected_pdf[col].dtype)
        a_dtype = str(actual_pdf[col].dtype)
        if e_dtype != a_dtype:
            result.type_mismatches.append({"column": col, "expected": e_dtype, "actual": a_dtype})

    # Row count
    result.row_count_expected = len(expected_pdf)
    result.row_count_actual = len(actual_pdf)
    result.row_count_match = result.row_count_expected == result.row_count_actual

    # Cell-level comparison on common columns up to min row count
    min_rows = min(result.row_count_expected, result.row_count_actual)
    result.total_cells = min_rows * len(common_cols) if common_cols else 0
    result.matching_cells = 0
    result.mismatched_columns = {col: 0 for col in common_cols}

    for col in common_cols:
        for i in range(min_rows):
            e_val = expected_pdf[col].iloc[i]
            a_val = actual_pdf[col].iloc[i]

            if _values_match(e_val, a_val, float_tolerance, case_sensitive):
                result.matching_cells += 1
            else:
                result.mismatched_columns[col] = result.mismatched_columns.get(col, 0) + 1
                if len(result.mismatched_rows) < max_mismatches:
                    result.mismatched_rows.append({
                        "row": i,
                        "column": col,
                        "expected": str(e_val)[:100],
                        "actual": str(a_val)[:100],
                    })

    # Account for missing row cells
    if result.row_count_expected > result.row_count_actual:
        extra_cells = (result.row_count_expected - result.row_count_actual) * len(common_cols)
        result.total_cells += extra_cells

    # Calculate match rate
    result.match_rate = result.matching_cells / result.total_cells if result.total_cells > 0 else 0.0

    # Overall pass
    result.overall_pass = result.match_rate >= pass_threshold and result.schema_match

    # Build summary
    if result.overall_pass:
        result.summary = f"PASSED: {result.match_rate:.1%} match rate ({result.matching_cells}/{result.total_cells} cells)"
    else:
        issues = []
        if not result.schema_match:
            issues.append(f"schema mismatch (missing: {result.missing_columns}, extra: {result.extra_columns})")
        if result.match_rate < pass_threshold:
            issues.append(f"match rate {result.match_rate:.1%} < threshold {pass_threshold:.0%}")
        if not result.row_count_match:
            issues.append(f"row count mismatch ({result.row_count_expected} vs {result.row_count_actual})")
        result.summary = f"FAILED: {'; '.join(issues)}"

    # Detailed summary
    lines = [
        "=" * 60,
        "DATA COMPARISON REPORT",
        "=" * 60,
        f"Schema Match:     {'YES' if result.schema_match else 'NO'}",
        f"Row Count Match:  {'YES' if result.row_count_match else 'NO'} ({result.row_count_expected} expected, {result.row_count_actual} actual)",
        f"Match Rate:       {result.match_rate:.2%} ({result.matching_cells}/{result.total_cells} cells)",
        f"Overall:          {'PASS' if result.overall_pass else 'FAIL'}",
        "",
    ]
    if result.missing_columns:
        lines.append(f"Missing Columns: {result.missing_columns}")
    if result.extra_columns:
        lines.append(f"Extra Columns:   {result.extra_columns}")
    if result.type_mismatches:
        lines.append(f"Type Mismatches: {len(result.type_mismatches)}")
        for tm in result.type_mismatches[:10]:
            lines.append(f"  {tm['column']}: {tm['expected']} -> {tm['actual']}")

    mismatched = {k: v for k, v in result.mismatched_columns.items() if v > 0}
    if mismatched:
        lines.append(f"\nColumns with mismatches:")
        for col, count in sorted(mismatched.items(), key=lambda x: -x[1]):
            lines.append(f"  {col}: {count} mismatches")

    if result.mismatched_rows:
        lines.append(f"\nSample mismatches (first {min(10, len(result.mismatched_rows))}):")
        for m in result.mismatched_rows[:10]:
            lines.append(f"  Row {m['row']}, {m['column']}: expected='{m['expected']}' actual='{m['actual']}'")

    lines.append("=" * 60)
    result.detailed_summary = "\n".join(lines)

    return result


# Run comparison
if expected_df is not None and actual_df is not None:
    result = compare_dataframes(
        expected_df, actual_df,
        float_tolerance=0.0001,
        case_sensitive=False,
        max_mismatches=100
    )
    print(result.detailed_summary)
else:
    result = ComparisonResult()
    result.overall_pass = False
    missing_parts = []
    if expected_df is None:
        missing_parts.append("expected output")
    if actual_df is None:
        missing_parts.append("actual output")
    result.summary = f"Could not compare - missing {', '.join(missing_parts)}"
    result.detailed_summary = result.summary
    print(f"\nCOMPARISON FAILED: {result.summary}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Comparison Report

# COMMAND ----------

os.makedirs(output_path, exist_ok=True)

# Build report in the format the Control Tower app expects:
# { "expected": {...}, "actual": {...}, "comparison": {...} }

def _build_data_source(pdf, label):
    """Build a DataSource dict from a pandas DataFrame."""
    if pdf is None:
        return None
    columns = [{"name": str(c), "type": str(pdf[c].dtype)} for c in pdf.columns]
    sample_rows = []
    for _, row in pdf.head(10).iterrows():
        sample_rows.append([
            None if pd.isna(v) else (
                int(v) if isinstance(v, (int,)) else
                float(v) if isinstance(v, (float,)) else
                str(v)
            )
            for v in row
        ])
    return {
        "table": label,
        "row_count": len(pdf),
        "columns": columns,
        "sample_rows": sample_rows,
    }


# Build the mismatched columns list (column names with >0 mismatches)
mismatched_col_names = sorted(
    [col for col, count in result.mismatched_columns.items() if count > 0],
    key=lambda c: -result.mismatched_columns.get(c, 0)
)

api_report = {
    "expected": _build_data_source(expected_df, "expected_output"),
    "actual": _build_data_source(actual_df, "actual_output"),
    "comparison": {
        "row_count_match": result.row_count_match,
        "schema_match": result.schema_match,
        "sample_match_rate": round(result.match_rate * 100, 2),  # API expects percentage 0-100
        "mismatched_columns": mismatched_col_names[:10],
        "summary": result.summary,
    },
    # Also include the detailed results for debugging
    "details": {
        "overall_pass": result.overall_pass,
        "match_rate": result.match_rate,
        "total_cells": result.total_cells,
        "matching_cells": result.matching_cells,
        "missing_columns": result.missing_columns,
        "extra_columns": result.extra_columns,
        "type_mismatches": result.type_mismatches,
        "mismatched_rows": result.mismatched_rows[:50],
    }
}

report_path = f"{output_path}/comparison_report.json"
with open(report_path, "w") as f:
    json.dump(api_report, f, indent=2, default=str)
print(f"Saved comparison report to: {report_path}")

# Save human-readable summary
summary_path = f"{output_path}/comparison_summary.txt"
with open(summary_path, "w") as f:
    f.write(result.detailed_summary)
print(f"Saved comparison summary to: {summary_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Conversion Job Quality Score

# COMMAND ----------

# Only update Delta tables if we had actual data to compare
_has_comparison_data = expected_df is not None and actual_df is not None

if _has_comparison_data:
    data_quality_score = result.match_rate if result.overall_pass else result.match_rate * 0.8

    # Update the conversion job with data validation results
    try:
        update_sql = f"""
        UPDATE {catalog}.conversion_tracker.conversion_jobs
        SET
            quality_score = COALESCE(quality_score, 0) * 0.6 + {data_quality_score} * 0.4,
            updated_at = current_timestamp()
        WHERE job_id = '{conversion_job_id}'
        """
        spark.sql(update_sql)
        print(f"Updated job quality score with data validation component")
    except Exception as e:
        print(f"Warning: Could not update job quality score: {e}")

    # Save individual validation results
    try:
        # Schema validation
        validation_id = str(uuid.uuid4())
        schema_actual = f'{{"missing": {len(result.missing_columns)}, "extra": {len(result.extra_columns)}, "type_mismatches": {len(result.type_mismatches)}}}'
        schema_message = "Schema matches" if result.schema_match else result.summary[:200].replace("'", "''")
        schema_severity = "info" if result.schema_match else "error"

        spark.sql(f"""
        INSERT INTO {catalog}.conversion_tracker.validation_results
        (validation_id, job_id, check_name, passed, expected, actual, message, severity, category, created_at)
        VALUES (
            '{validation_id}', '{conversion_job_id}', 'data_schema_match',
            {str(result.schema_match).lower()}, 'Schema should match', '{schema_actual}',
            '{schema_message}', '{schema_severity}', 'data_validation', current_timestamp()
        )
        """)

        # Row count validation
        validation_id = str(uuid.uuid4())
        row_message = "Row count matches" if result.row_count_match else "Row count mismatch"
        row_severity = "info" if result.row_count_match else "warning"

        spark.sql(f"""
        INSERT INTO {catalog}.conversion_tracker.validation_results
        (validation_id, job_id, check_name, passed, expected, actual, message, severity, category, created_at)
        VALUES (
            '{validation_id}', '{conversion_job_id}', 'data_row_count_match',
            {str(result.row_count_match).lower()}, '{result.row_count_expected}', '{result.row_count_actual}',
            '{row_message}', '{row_severity}', 'data_validation', current_timestamp()
        )
        """)

        # Data match rate
        validation_id = str(uuid.uuid4())
        rate_passed = result.match_rate >= pass_threshold

        spark.sql(f"""
        INSERT INTO {catalog}.conversion_tracker.validation_results
        (validation_id, job_id, check_name, passed, expected, actual, message, severity, category, created_at)
        VALUES (
            '{validation_id}', '{conversion_job_id}', 'data_match_rate',
            {str(rate_passed).lower()}, '>= {pass_threshold:.0%}', '{result.match_rate:.2%}',
            'Data match rate: {result.match_rate:.2%}', '{"info" if rate_passed else "error"}', 'data_validation', current_timestamp()
        )
        """)

        print(f"Saved validation results to tracking table")

    except Exception as e:
        print(f"Warning: Could not save validation results: {e}")
else:
    print("Skipping Delta table updates - no comparison data available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Comparison Results

# COMMAND ----------

print(f"{'Metric':<25} {'Value':<20} {'Status'}")
print("-" * 55)
for row in [
    ("Overall Pass", str(result.overall_pass), "PASS" if result.overall_pass else "FAIL"),
    ("Match Rate", f"{result.match_rate:.2%}", "PASS" if result.match_rate >= pass_threshold else "FAIL"),
    ("Schema Match", str(result.schema_match), "PASS" if result.schema_match else "FAIL"),
    ("Row Count Match", str(result.row_count_match), "PASS" if result.row_count_match else "FAIL"),
    ("Expected Rows", str(result.row_count_expected), "-"),
    ("Actual Rows", str(result.row_count_actual), "-"),
    ("Total Cells", str(result.total_cells), "-"),
    ("Matching Cells", str(result.matching_cells), "-"),
]:
    print(f"{row[0]:<25} {row[1]:<20} {row[2]}")

# COMMAND ----------

if result.mismatched_columns:
    mismatched = {k: v for k, v in result.mismatched_columns.items() if v > 0}
    if mismatched:
        print("\nColumns with mismatches:")
        for col, count in sorted(mismatched.items(), key=lambda x: -x[1]):
            print(f"  {col}: {count}")

# COMMAND ----------

if result.mismatched_rows:
    print(f"\nSample mismatches (showing first {min(10, len(result.mismatched_rows))}):")
    for m in result.mismatched_rows[:10]:
        print(f"  Row {m.get('row')}, {m.get('column')}: expected={m.get('expected')} actual={m.get('actual')}")

# COMMAND ----------

print(f"\n{'='*60}")
print("DATA COMPARISON COMPLETE")
print(f"{'='*60}")
print(f"Conversion Job ID: {conversion_job_id}")
print(f"Overall Pass: {result.overall_pass}")
print(f"Match Rate: {result.match_rate:.2%}")
print(f"Report Path: {output_path}/comparison_report.json")
print(f"{'='*60}")

if not result.overall_pass:
    print("\nData validation FAILED - converted output differs from expected")
else:
    print("\nData validation PASSED")
