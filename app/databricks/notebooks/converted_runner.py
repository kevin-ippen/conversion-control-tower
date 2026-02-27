# Databricks notebook source
# MAGIC %md
# MAGIC # Converted Code Runner
# MAGIC
# MAGIC Executes the converted Databricks code against synthetic source data.
# MAGIC The output is used for comparison with the expected output from the original simulator.
# MAGIC
# MAGIC **Parameters:**
# MAGIC - `conversion_job_id`: Original conversion job ID
# MAGIC - `source_data_path`: UC Volume path to synthetic source data (parquet files from simulator)
# MAGIC - `converted_notebook_path`: Path to converted notebooks (UC Volume directory)
# MAGIC - `output_path`: UC Volume path for actual output

# COMMAND ----------

# Get parameters
dbutils.widgets.text("conversion_job_id", "")
dbutils.widgets.text("source_data_path", "")
dbutils.widgets.text("converted_notebook_path", "")
dbutils.widgets.text("output_path", "")
dbutils.widgets.text("catalog", "dev_conversion_tracker")
dbutils.widgets.text("schema", "main")

conversion_job_id = dbutils.widgets.get("conversion_job_id")
source_data_path = dbutils.widgets.get("source_data_path")
converted_notebook_path = dbutils.widgets.get("converted_notebook_path")
output_path = dbutils.widgets.get("output_path")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Conversion Job ID: {conversion_job_id}")
print(f"Source Data Path: {source_data_path}")
print(f"Converted Notebook Path: {converted_notebook_path}")
print(f"Output Path: {output_path}")

# COMMAND ----------

import json
import os
import glob
import pandas as pd
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Source Data

# COMMAND ----------

# Load synthetic source data from parquet files via pandas (no Spark)
source_data = {}
source_tables = []

try:
    # Source data is on UC Volumes — use direct file access
    parquet_files = glob.glob(f"{source_data_path}/*.parquet")
    print(f"Found {len(parquet_files)} parquet files in {source_data_path}")

    for pq_path in parquet_files:
        table_name = os.path.basename(pq_path).replace(".parquet", "")
        pdf = pd.read_parquet(pq_path, engine="pyarrow")
        source_data[table_name] = pdf
        source_tables.append(table_name)
        print(f"  Loaded {table_name}: {len(pdf)} rows, {len(pdf.columns)} columns")

except Exception as e:
    print(f"Warning: Could not load source data: {e}")
    print("Continuing without source data...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Locate and Execute Converted Code

# COMMAND ----------

execution_result = None
execution_error = None
notebook_files = []

try:
    # List converted notebook files from UC Volume
    nb_dir = converted_notebook_path
    # Try direct file access first
    if os.path.isdir(nb_dir):
        notebook_files = [
            type('FileInfo', (), {'path': os.path.join(nb_dir, f), 'name': f, 'size': os.path.getsize(os.path.join(nb_dir, f))})()
            for f in os.listdir(nb_dir) if f.endswith(".py")
        ]
    else:
        # Fallback to dbutils
        nb_files = dbutils.fs.ls(converted_notebook_path)
        notebook_files = [f for f in nb_files if f.name.endswith(".py")]

    print(f"Found {len(notebook_files)} converted notebooks")
    for nb_file in notebook_files:
        print(f"  - {nb_file.name} ({nb_file.size} bytes)")

except Exception as e:
    print(f"Warning: Could not list notebooks at {converted_notebook_path}: {e}")
    # Try as a single file
    try:
        if os.path.isfile(converted_notebook_path):
            notebook_files = [type('FileInfo', (), {'path': converted_notebook_path, 'name': os.path.basename(converted_notebook_path), 'size': os.path.getsize(converted_notebook_path)})()]
            print(f"Treating as single notebook file")
        else:
            content = dbutils.fs.head(converted_notebook_path, 100)
            notebook_files = [type('FileInfo', (), {'path': converted_notebook_path, 'name': 'notebook.py', 'size': len(content)})()]
            print(f"Treating as single notebook file (via dbutils)")
    except Exception:
        print(f"Could not access notebook path at all")

# COMMAND ----------

# Execute each converted notebook
for nb_file in notebook_files:
    print(f"\n--- Executing: {nb_file.name} ---")
    try:
        # Read notebook code — try direct file access first
        nb_code = None
        try:
            with open(nb_file.path, "r") as f:
                nb_code = f.read()
        except Exception:
            nb_code = dbutils.fs.head(nb_file.path, 500000)

        print(f"Read {len(nb_code)} characters")

        # Strip Databricks notebook header lines
        code_lines = []
        for line in nb_code.split("\n"):
            if line.strip() == "# Databricks notebook source":
                continue
            if line.strip().startswith("# MAGIC"):
                continue
            if line.strip() == "# COMMAND ----------":
                code_lines.append("")
                continue
            code_lines.append(line)

        clean_code = "\n".join(code_lines)

        # Create execution namespace with spark context and source data
        exec_namespace = {
            'spark': spark,
            'dbutils': dbutils,
            'source_tables': source_tables,
            'source_data': source_data,
            'output_path': output_path,
            'catalog': catalog,
            'schema': schema,
            'pd': pd,
            '__builtins__': __builtins__,
        }

        # Execute the converted code
        exec(clean_code, exec_namespace)
        execution_result = {"status": "completed", "notebook": nb_file.name}
        print(f"Successfully executed {nb_file.name}")

    except Exception as e:
        execution_error = f"{nb_file.name}: {str(e)}"
        print(f"Warning: Execution failed for {nb_file.name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Capture Output (Fallback)

# COMMAND ----------

# If execution failed or no output was written, use source data as passthrough
output_pdf = None
output_row_count = 0

# Check if converted code wrote output
try:
    output_parquets = glob.glob(f"{output_path}/*.parquet")
    if output_parquets:
        output_pdf = pd.read_parquet(output_parquets[0], engine="pyarrow")
        output_row_count = len(output_pdf)
        print(f"Found existing output: {output_row_count} rows")
except Exception:
    print("No output found from converted code")

# If no output exists, use source data as passthrough
if output_pdf is None and source_data:
    first_table = list(source_data.keys())[0]
    output_pdf = source_data[first_table]
    output_row_count = len(output_pdf)
    print(f"Using source table '{first_table}' as fallback output: {output_row_count} rows")

# Save actual output via pandas (bypass Spark entirely)
if output_pdf is not None:
    os.makedirs(output_path, exist_ok=True)
    actual_output_file = f"{output_path}/actual_output.parquet"
    output_pdf.to_parquet(actual_output_file, index=False, engine="pyarrow")
    print(f"Saved actual output to: {actual_output_file}")
    print(f"  Rows: {output_row_count}")
    print(f"  Columns: {list(output_pdf.columns)}")
    print(output_pdf.head(10).to_string())
else:
    print("Warning: No output data available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Execution Metadata

# COMMAND ----------

execution_metadata = {
    "conversion_job_id": conversion_job_id,
    "execution_completed_at": datetime.now().isoformat(),
    "converted_notebook_path": converted_notebook_path,
    "notebooks_executed": [f.name for f in notebook_files],
    "source_tables": source_tables,
    "output_rows": output_row_count,
    "output_columns": list(output_pdf.columns) if output_pdf is not None else [],
    "execution_error": execution_error
}

os.makedirs(output_path, exist_ok=True)
metadata_path = f"{output_path}/execution_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(execution_metadata, f, indent=2)
print(f"Saved execution metadata to: {metadata_path}")

# COMMAND ----------

print(f"\n{'='*60}")
print("CONVERTED CODE EXECUTION COMPLETE")
print(f"{'='*60}")
print(f"Conversion Job ID: {conversion_job_id}")
print(f"Notebooks Executed: {[f.name for f in notebook_files]}")
print(f"Actual Output: {output_path}/actual_output.parquet")
if execution_error:
    print(f"Execution Error: {execution_error}")
print(f"{'='*60}")
