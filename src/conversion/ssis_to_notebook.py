"""
SSIS Data Flow to PySpark Notebook Converter

Converts SSIS Data Flow pipelines to PySpark notebook code.
"""

import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from src.ssis.dtsx_parser import (
    SSISDataFlow,
    SSISDataFlowComponent,
    SSISDataFlowPath
)


@dataclass
class SparkCodeBlock:
    """Represents a block of generated Spark code."""
    component_name: str
    component_type: str
    code: str
    output_var: str
    input_vars: List[str]
    comments: List[str]


class SSISToNotebookConverter:
    """
    Converts SSIS Data Flow pipelines to PySpark notebooks.
    """

    def __init__(self, mappings_path: Optional[Path] = None):
        """Initialize converter with optional custom mappings."""
        if mappings_path is None:
            mappings_path = Path(__file__).parent.parent.parent / "config" / "ssis_component_mappings.yaml"

        self.mappings = {}
        if mappings_path.exists():
            with open(mappings_path) as f:
                self.mappings = yaml.safe_load(f)

        self.component_mappings = self.mappings.get("data_flow_components", {})
        self._var_counter = 0

    def convert_data_flow(self, data_flow: SSISDataFlow, task_name: str) -> str:
        """Convert a complete data flow to a PySpark notebook."""

        self._var_counter = 0

        # Build component dependency graph
        graph = self._build_component_graph(data_flow)

        # Generate code blocks in topological order
        code_blocks = []
        processed = set()

        for component in self._topological_sort(data_flow.components, graph):
            if component.name not in processed:
                block = self._convert_component(component, data_flow, graph)
                if block:
                    code_blocks.append(block)
                processed.add(component.name)

        # Assemble notebook
        return self._assemble_notebook(data_flow.name, task_name, code_blocks)

    def _build_component_graph(self, data_flow: SSISDataFlow) -> Dict[str, List[str]]:
        """Build adjacency list of component dependencies."""

        # Map ref_id to component name
        ref_to_name = {}
        for comp in data_flow.components:
            ref_to_name[comp.ref_id] = comp.name
            # Also map output refs
            for out in comp.output_columns:
                if "ref_id" in out:
                    ref_to_name[out["ref_id"]] = comp.name

        # Build graph from paths
        graph = {comp.name: [] for comp in data_flow.components}

        for path in data_flow.paths:
            # Find source and dest components
            source_comp = None
            dest_comp = None

            for comp in data_flow.components:
                if path.source_ref.startswith(comp.ref_id):
                    source_comp = comp.name
                if path.dest_ref.startswith(comp.ref_id):
                    dest_comp = comp.name

            if source_comp and dest_comp and source_comp != dest_comp:
                if source_comp not in graph[dest_comp]:
                    graph[dest_comp].append(source_comp)

        return graph

    def _topological_sort(
        self,
        components: List[SSISDataFlowComponent],
        graph: Dict[str, List[str]]
    ) -> List[SSISDataFlowComponent]:
        """Sort components in dependency order."""

        # Kahn's algorithm
        in_degree = {comp.name: len(graph.get(comp.name, [])) for comp in components}
        queue = [comp for comp in components if in_degree[comp.name] == 0]
        result = []

        while queue:
            comp = queue.pop(0)
            result.append(comp)

            # Find components that depend on this one
            for other in components:
                if comp.name in graph.get(other.name, []):
                    in_degree[other.name] -= 1
                    if in_degree[other.name] == 0:
                        queue.append(other)

        return result

    def _convert_component(
        self,
        component: SSISDataFlowComponent,
        data_flow: SSISDataFlow,
        graph: Dict[str, List[str]]
    ) -> Optional[SparkCodeBlock]:
        """Convert a single data flow component to Spark code."""

        comp_type = component.component_type
        mapping = self.component_mappings.get(comp_type, {})

        # Get input variable names
        input_vars = [self._get_var_name(dep) for dep in graph.get(component.name, [])]
        output_var = self._get_var_name(component.name)

        # Generate code based on component type
        if comp_type == "OLEDBSource":
            return self._convert_oledb_source(component, output_var)

        elif comp_type == "Lookup":
            return self._convert_lookup(component, input_vars, output_var)

        elif comp_type == "DerivedColumn":
            return self._convert_derived_column(component, input_vars, output_var)

        elif comp_type == "ConditionalSplit":
            return self._convert_conditional_split(component, input_vars, output_var)

        elif comp_type == "Aggregate":
            return self._convert_aggregate(component, input_vars, output_var)

        elif comp_type == "Sort":
            return self._convert_sort(component, input_vars, output_var)

        elif comp_type == "UnionAll":
            return self._convert_union(component, input_vars, output_var)

        elif comp_type == "Multicast":
            return self._convert_multicast(component, input_vars, output_var)

        elif comp_type == "RowCount":
            return self._convert_row_count(component, input_vars, output_var)

        elif comp_type == "OLEDBDestination":
            return self._convert_oledb_destination(component, input_vars, output_var)

        elif comp_type == "FlatFileDestination":
            return self._convert_flatfile_destination(component, input_vars, output_var)

        elif comp_type == "SlowlyChangingDimension":
            return self._convert_scd(component, input_vars, output_var)

        elif comp_type == "OLEDBCommand":
            return self._convert_oledb_command(component, input_vars, output_var)

        else:
            # Generic passthrough for unknown components
            return SparkCodeBlock(
                component_name=component.name,
                component_type=comp_type,
                code=f"# TODO: Implement conversion for {comp_type}\n{output_var} = {input_vars[0] if input_vars else 'spark.createDataFrame([])'}",
                output_var=output_var,
                input_vars=input_vars,
                comments=[f"Unknown component type: {comp_type}"]
            )

    def _convert_oledb_source(self, component: SSISDataFlowComponent, output_var: str) -> SparkCodeBlock:
        """Convert OLE DB Source to Spark read."""

        sql_command = component.properties.get("SqlCommand", "")

        # Clean up SQL for Spark
        sql_command = self._transform_sql(sql_command)

        code = f'''# Source: {component.name}
# Original SQL: {sql_command[:100]}...
{output_var} = spark.read.format("jdbc") \\
    .option("url", jdbc_url) \\
    .option("query", """
{sql_command}
    """) \\
    .load()

# Alternative using Unity Catalog:
# {output_var} = spark.table("catalog.schema.table")'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="OLEDBSource",
            code=code,
            output_var=output_var,
            input_vars=[],
            comments=[f"Source query from {component.name}"]
        )

    def _convert_lookup(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Lookup to Spark join."""

        lookup_sql = component.properties.get("SqlCommand", "")
        input_var = input_vars[0] if input_vars else "df"

        # Extract join keys from input columns
        join_keys = []
        for col in component.input_columns:
            join_keys.append(col.get("name", ""))

        code = f'''# Lookup: {component.name}
# Join keys: {join_keys}
lookup_df = spark.read.format("jdbc") \\
    .option("url", jdbc_url) \\
    .option("query", """{lookup_sql}""") \\
    .load()

# Use broadcast for small dimension tables
{output_var} = {input_var}.join(
    broadcast(lookup_df),
    on={json.dumps(join_keys) if join_keys else "['key']"},
    how="left"
)'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="Lookup",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=[f"Lookup join with {len(join_keys)} keys"]
        )

    def _convert_derived_column(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Derived Column to withColumn."""

        input_var = input_vars[0] if input_vars else "df"

        # Generate withColumn calls for each derived column
        with_columns = []
        for col in component.output_columns:
            col_name = col.get("name", "")
            expression = col.get("expression", "")

            if expression:
                # Convert SSIS expression to Spark
                spark_expr = self._convert_expression(expression)
                with_columns.append(f'    .withColumn("{col_name}", {spark_expr})')

        code = f'''# Derived Column: {component.name}
{output_var} = {input_var} \\
{chr(10).join(with_columns) if with_columns else "    # No derived columns defined"}'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="DerivedColumn",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=[f"Adding {len(with_columns)} derived columns"]
        )

    def _convert_conditional_split(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Conditional Split to multiple filtered DataFrames."""

        input_var = input_vars[0] if input_vars else "df"

        # Generate filter for each output
        filters = []
        for i, out in enumerate(component.output_columns):
            out_name = out.get("name", f"output_{i}")
            # Expression would be in a different structure - simplified here
            filters.append(f'{output_var}_{self._sanitize_var(out_name)} = {input_var}.filter("condition_{i}")')

        code = f'''# Conditional Split: {component.name}
# TODO: Update filter conditions based on SSIS expressions
{chr(10).join(filters) if filters else f"{output_var} = {input_var}  # No conditions defined"}'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="ConditionalSplit",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["Conditional split - update filter conditions"]
        )

    def _convert_aggregate(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Aggregate to groupBy/agg."""

        input_var = input_vars[0] if input_vars else "df"

        # Parse aggregation columns
        group_cols = []
        agg_exprs = []

        for col in component.output_columns:
            agg_type = col.get("aggregationType", "").lower()
            col_name = col.get("name", "")

            if agg_type == "groupby":
                group_cols.append(f'"{col_name}"')
            elif agg_type == "sum":
                agg_exprs.append(f'F.sum("{col_name}").alias("{col_name}")')
            elif agg_type == "count":
                agg_exprs.append(f'F.count("*").alias("{col_name}")')
            elif agg_type == "countdistinct":
                agg_exprs.append(f'F.countDistinct("{col_name}").alias("{col_name}")')
            elif agg_type == "avg":
                agg_exprs.append(f'F.avg("{col_name}").alias("{col_name}")')
            elif agg_type == "min":
                agg_exprs.append(f'F.min("{col_name}").alias("{col_name}")')
            elif agg_type == "max":
                agg_exprs.append(f'F.max("{col_name}").alias("{col_name}")')

        code = f'''# Aggregate: {component.name}
{output_var} = {input_var}.groupBy({", ".join(group_cols)}).agg(
    {("," + chr(10) + "    ").join(agg_exprs) if agg_exprs else "F.count('*').alias('count')"}
)'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="Aggregate",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=[f"Aggregation with {len(group_cols)} group columns"]
        )

    def _convert_sort(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Sort to orderBy."""

        input_var = input_vars[0] if input_vars else "df"

        code = f'''# Sort: {component.name}
# WARNING: Full sort is expensive in Spark - consider if really needed
{output_var} = {input_var}.orderBy("sort_column")  # TODO: Update sort columns'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="Sort",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["WARNING: Consider removing sort if not required"]
        )

    def _convert_union(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Union All to unionByName."""

        if len(input_vars) >= 2:
            union_chain = f"{input_vars[0]}"
            for var in input_vars[1:]:
                union_chain = f"{union_chain}.unionByName({var}, allowMissingColumns=True)"
            code = f'''# Union All: {component.name}
{output_var} = {union_chain}'''
        else:
            code = f'''# Union All: {component.name}
{output_var} = {input_vars[0] if input_vars else "spark.createDataFrame([])"}'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="UnionAll",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=[f"Union of {len(input_vars)} inputs"]
        )

    def _convert_multicast(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Multicast to cache."""

        input_var = input_vars[0] if input_vars else "df"

        code = f'''# Multicast: {component.name}
# Cache DataFrame for reuse in multiple downstream operations
{output_var} = {input_var}.cache()'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="Multicast",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["Cached for multiple downstream uses"]
        )

    def _convert_row_count(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Row Count to count operation."""

        input_var = input_vars[0] if input_vars else "df"

        code = f'''# Row Count: {component.name}
row_count_{self._sanitize_var(component.name)} = {input_var}.count()
print(f"Row count for {component.name}: {{row_count_{self._sanitize_var(component.name)}}}")
{output_var} = {input_var}  # Pass through'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="RowCount",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["Row count for metrics"]
        )

    def _convert_oledb_destination(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert OLE DB Destination to Delta write."""

        input_var = input_vars[0] if input_vars else "df"
        table_name = component.properties.get("OpenRowset", "target_table")

        # Clean up table name
        table_name = table_name.replace("[", "").replace("]", "")

        code = f'''# Destination: {component.name}
# Target table: {table_name}
{input_var}.write.format("delta") \\
    .mode("append") \\
    .option("mergeSchema", "true") \\
    .saveAsTable("catalog.schema.{table_name.replace(".", "_")}")

{output_var} = {input_var}  # Reference for downstream'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="OLEDBDestination",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=[f"Write to Delta table: {table_name}"]
        )

    def _convert_flatfile_destination(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Flat File Destination to CSV write."""

        input_var = input_vars[0] if input_vars else "df"

        code = f'''# Flat File Destination: {component.name}
{input_var}.write.format("csv") \\
    .option("header", "true") \\
    .mode("append") \\
    .save("/Volumes/catalog/schema/volume/output/")

{output_var} = {input_var}'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="FlatFileDestination",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["Write to CSV file"]
        )

    def _convert_scd(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert Slowly Changing Dimension to Delta MERGE."""

        input_var = input_vars[0] if input_vars else "df"

        code = f'''# Slowly Changing Dimension: {component.name}
from delta.tables import DeltaTable

# Get target dimension table
dim_table = DeltaTable.forName(spark, "catalog.schema.dim_table")

# SCD Type 2: Expire old records for changed historical attributes
dim_table.alias("t").merge(
    {input_var}.alias("s"),
    "t.business_key = s.business_key AND t.is_current = true"
).whenMatchedUpdate(
    condition="hash(t.type2_cols) != hash(s.type2_cols)",
    set={{
        "is_current": "false",
        "effective_end_date": "current_timestamp()",
        "modified_date": "current_timestamp()"
    }}
).execute()

# Insert new records (new and changed)
new_records = {input_var}.withColumn("effective_start_date", F.current_timestamp()) \\
    .withColumn("effective_end_date", F.lit("9999-12-31").cast("timestamp")) \\
    .withColumn("is_current", F.lit(True))

dim_table.alias("t").merge(
    new_records.alias("s"),
    "t.business_key = s.business_key AND t.is_current = true"
).whenNotMatched().insertAll().execute()

{output_var} = {input_var}'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="SlowlyChangingDimension",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["SCD Type 2 via Delta MERGE - update business_key and type2_cols"]
        )

    def _convert_oledb_command(
        self,
        component: SSISDataFlowComponent,
        input_vars: List[str],
        output_var: str
    ) -> SparkCodeBlock:
        """Convert OLE DB Command to batch operation."""

        input_var = input_vars[0] if input_vars else "df"

        code = f'''# OLE DB Command: {component.name}
# WARNING: Row-by-row updates are anti-pattern in Spark
# Consider using Delta MERGE for batch updates instead

# Original approach (avoid):
# for row in {input_var}.collect():
#     spark.sql(f"UPDATE ...")

# Better approach - batch MERGE:
# DeltaTable.forName(spark, "table").merge(...)

{output_var} = {input_var}'''

        return SparkCodeBlock(
            component_name=component.name,
            component_type="OLEDBCommand",
            code=code,
            output_var=output_var,
            input_vars=input_vars,
            comments=["WARNING: Convert to batch MERGE operation"]
        )

    def _transform_sql(self, sql: str) -> str:
        """Apply basic SQL transformations."""

        transformations = [
            ("GETDATE()", "current_timestamp()"),
            ("ISNULL(", "COALESCE("),
            ("NOLOCK", ""),
            ("WITH (NOLOCK)", ""),
            ("&gt;", ">"),
            ("&lt;", "<"),
            ("&amp;", "&"),
        ]

        result = sql
        for old, new in transformations:
            result = result.replace(old, new)

        return result

    def _convert_expression(self, expression: str) -> str:
        """Convert SSIS expression to Spark SQL expression."""

        # Basic conversions
        result = expression

        # Ternary to CASE WHEN
        if "?" in result and ":" in result:
            result = f'F.expr("CASE WHEN {result} END")  # TODO: Convert ternary'

        # GETDATE() to current_timestamp()
        result = result.replace("GETDATE()", "F.current_timestamp()")

        # Wrap in F.expr if it looks like SQL
        if any(kw in result.upper() for kw in ["CASE", "WHEN", "COALESCE", "CAST"]):
            result = f'F.expr("{result}")'
        elif "col(" not in result.lower() and "f." not in result.lower():
            result = f'F.expr("{result}")'

        return result

    def _get_var_name(self, name: str) -> str:
        """Generate a valid Python variable name."""

        self._var_counter += 1
        return f"df_{self._sanitize_var(name)}"

    def _sanitize_var(self, name: str) -> str:
        """Sanitize string for use as variable name."""

        result = name.lower()
        result = result.replace(" ", "_")
        result = result.replace("-", "_")
        result = result.replace(".", "_")

        # Remove invalid characters
        result = "".join(c for c in result if c.isalnum() or c == "_")

        return result

    def _assemble_notebook(
        self,
        data_flow_name: str,
        task_name: str,
        code_blocks: List[SparkCodeBlock]
    ) -> str:
        """Assemble code blocks into a complete notebook."""

        header = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # {task_name}
# MAGIC
# MAGIC Converted from SSIS Data Flow: **{data_flow_name}**
# MAGIC
# MAGIC ## Components
# MAGIC {chr(10).join(f"# MAGIC - {b.component_name} ({b.component_type})" for b in code_blocks)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when, broadcast, current_timestamp
from delta.tables import DeltaTable

# Configuration - update these values
jdbc_url = "jdbc:sqlserver://server:1433;database=db"  # TODO: Use secrets
catalog = "your_catalog"
schema = "your_schema"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline

'''

        # Add each code block as a cell
        cells = []
        for block in code_blocks:
            cell = f'''# COMMAND ----------

# MAGIC %md
# MAGIC ### {block.component_name}
# MAGIC Type: `{block.component_type}`
{"# MAGIC " + chr(10).join("# MAGIC " + c for c in block.comments) if block.comments else ""}

# COMMAND ----------

{block.code}
'''
            cells.append(cell)

        footer = '''
# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

# Print summary metrics
print("Pipeline execution complete")
'''

        return header + "\n".join(cells) + footer


# For JSON imports in lookup conversion
import json
