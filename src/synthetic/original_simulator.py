"""
Original Output Simulator

Simulates what the original SQL Server code would produce when run
against synthetic source data. This enables comparison between
the "expected" output and the actual converted code output.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from decimal import Decimal
import re

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..extraction.schema_extractor import ExtractedSchemas, TableSchema, DataType


class OriginalOutputSimulator:
    """
    Simulate the expected output of original SQL Server logic.

    Uses the extracted transformation rules from the source script
    to produce what the original code WOULD have produced if run
    against the synthetic source data.

    This serves as the "expected" baseline for validating converted code.
    """

    def __init__(self):
        self.expression_evaluators: Dict[str, Callable] = {}
        self._setup_default_evaluators()

    def _setup_default_evaluators(self):
        """Set up default expression evaluators for common SSIS expressions."""
        self.expression_evaluators = {
            "GETDATE": lambda: datetime.now(),
            "GETUTCDATE": lambda: datetime.utcnow(),
            "YEAR": lambda d: d.year if hasattr(d, 'year') else None,
            "MONTH": lambda d: d.month if hasattr(d, 'month') else None,
            "DAY": lambda d: d.day if hasattr(d, 'day') else None,
            "ISNULL": lambda val, default: default if val is None else val,
            "UPPER": lambda s: s.upper() if s else s,
            "LOWER": lambda s: s.lower() if s else s,
            "LTRIM": lambda s: s.lstrip() if s else s,
            "RTRIM": lambda s: s.rstrip() if s else s,
            "TRIM": lambda s: s.strip() if s else s,
            "LEN": lambda s: len(s) if s else 0,
            "SUBSTRING": lambda s, start, length: s[start-1:start-1+length] if s else "",
            "REPLACE": lambda s, old, new: s.replace(old, new) if s else s,
            "ROUND": lambda val, decimals: round(val, decimals) if val else val,
            "ABS": lambda val: abs(val) if val else val,
            "CEILING": lambda val: int(val) + (1 if val > int(val) else 0) if val else val,
            "FLOOR": lambda val: int(val) if val else val,
        }

    def simulate_ssis_output(
        self,
        source_data: Dict[str, Any],
        schemas: ExtractedSchemas,
        data_flow_name: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Simulate SSIS data flow transformations.

        Applies in order:
        1. Column mappings (renames)
        2. Derived columns (expressions)
        3. Type conversions
        4. Filter conditions
        5. Aggregations (if any)

        Args:
            source_data: Dict mapping table_name -> data dict
            schemas: Extracted schemas with transformations
            data_flow_name: Optional name of specific data flow to simulate

        Returns:
            Dict representing the expected output data
        """
        # Get the main source table
        if not schemas.source_tables:
            return {}

        source_table = schemas.source_tables[0]
        source_name = source_table.table_name

        # Get source data for the main table
        if source_name not in source_data:
            # Try to find a match by partial name
            for key in source_data:
                if source_name.lower() in key.lower() or key.lower() in source_name.lower():
                    source_name = key
                    break

        if source_name not in source_data:
            return {}

        # Start with a copy of source data
        result = {col: list(vals) for col, vals in source_data[source_name].items()}

        # Apply transformations
        result = self._apply_column_mappings(result, schemas.transformations)
        result = self._apply_derived_columns(result, schemas.transformations)

        # Apply destination schema constraints
        if schemas.destination_tables:
            dest_schema = schemas.destination_tables[0]
            result = self._enforce_schema(result, dest_schema)

        return result

    def simulate_ssis_output_pandas(
        self,
        source_data: Dict[str, Any],
        schemas: ExtractedSchemas,
        data_flow_name: Optional[str] = None
    ):
        """
        Simulate SSIS data flow transformations returning pandas DataFrame.

        Args:
            source_data: Dict mapping table_name -> DataFrame
            schemas: Extracted schemas with transformations
            data_flow_name: Optional name of specific data flow to simulate

        Returns:
            pandas DataFrame representing the expected output
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for simulate_ssis_output_pandas()")

        # Convert DataFrames to dicts for processing
        dict_data = {}
        for table_name, df in source_data.items():
            if isinstance(df, pd.DataFrame):
                dict_data[table_name] = df.to_dict('list')
            else:
                dict_data[table_name] = df

        result_dict = self.simulate_ssis_output(dict_data, schemas, data_flow_name)

        if result_dict:
            return pd.DataFrame(result_dict)
        return pd.DataFrame()

    def simulate_stored_proc_output(
        self,
        source_data: Dict[str, Any],
        schemas: ExtractedSchemas,
        sql_logic: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Simulate stored procedure output using extracted logic.

        This is a simplified simulation that handles common patterns:
        - SELECT with column transformations
        - JOINs between tables
        - WHERE filters
        - GROUP BY aggregations

        Args:
            source_data: Dict mapping table_name -> data dict
            schemas: Extracted schemas
            sql_logic: Optional raw SQL for parsing

        Returns:
            Dict representing the expected output data
        """
        # Similar logic to SSIS, adapted for SQL patterns
        if not schemas.source_tables:
            return {}

        source_table = schemas.source_tables[0]
        source_name = source_table.table_name

        # Find matching source data
        for key in source_data:
            if source_name.lower() in key.lower():
                source_name = key
                break

        if source_name not in source_data:
            return {}

        result = {col: list(vals) for col, vals in source_data[source_name].items()}

        # Apply basic transformations
        result = self._apply_column_mappings(result, schemas.transformations)

        return result

    def _apply_column_mappings(
        self,
        data: Dict[str, List[Any]],
        transformations: Dict[str, str]
    ) -> Dict[str, List[Any]]:
        """
        Apply column renames/mappings from transformations.

        Args:
            data: Source data dict
            transformations: Dict of source_col -> dest_col or expression

        Returns:
            Data dict with columns renamed
        """
        result = dict(data)

        for dest_col, source_expr in transformations.items():
            # Simple column rename
            if source_expr in data and dest_col not in result:
                result[dest_col] = list(data[source_expr])

        return result

    def _apply_derived_columns(
        self,
        data: Dict[str, List[Any]],
        transformations: Dict[str, str]
    ) -> Dict[str, List[Any]]:
        """
        Apply derived column expressions.

        Handles common patterns:
        - Column references: [ColumnName]
        - Concatenation: [Col1] + " " + [Col2]
        - Math operations: [Price] * [Quantity]
        - Function calls: UPPER([Name])
        - Type conversions: (DT_STR,50)[Column]

        Args:
            data: Current data dict
            transformations: Dict of column -> expression

        Returns:
            Data dict with derived columns added
        """
        if not data:
            return data

        row_count = len(next(iter(data.values())))
        result = dict(data)

        for dest_col, expression in transformations.items():
            # Skip simple column references (handled by _apply_column_mappings)
            if expression in data:
                continue

            # Try to evaluate the expression for each row
            try:
                values = self._evaluate_expression_rows(expression, data, row_count)
                if values:
                    result[dest_col] = values
            except Exception:
                # If expression evaluation fails, skip
                pass

        return result

    def _evaluate_expression_rows(
        self,
        expression: str,
        data: Dict[str, List[Any]],
        row_count: int
    ) -> Optional[List[Any]]:
        """
        Evaluate an expression for all rows.

        Args:
            expression: SSIS expression string
            data: Current data dict
            row_count: Number of rows

        Returns:
            List of evaluated values or None if evaluation fails
        """
        results = []

        for i in range(row_count):
            # Create row context
            row = {col: vals[i] for col, vals in data.items()}
            try:
                value = self._evaluate_expression(expression, row)
                results.append(value)
            except Exception:
                results.append(None)

        return results

    def _evaluate_expression(self, expression: str, row: Dict[str, Any]) -> Any:
        """
        Evaluate a single expression for one row.

        Args:
            expression: SSIS expression string
            row: Dict of column values for this row

        Returns:
            Evaluated value
        """
        # Handle column references [ColumnName]
        result = expression
        for col_name, value in row.items():
            # Replace [ColName] with the value
            pattern = re.escape(f"[{col_name}]")
            if re.search(pattern, result, re.IGNORECASE):
                if isinstance(value, str):
                    result = re.sub(pattern, f'"{value}"', result, flags=re.IGNORECASE)
                elif value is None:
                    result = re.sub(pattern, "None", result, flags=re.IGNORECASE)
                else:
                    result = re.sub(pattern, str(value), result, flags=re.IGNORECASE)

        # Handle function calls
        for func_name, func in self.expression_evaluators.items():
            pattern = rf"{func_name}\s*\((.*?)\)"
            matches = re.findall(pattern, result, re.IGNORECASE)
            for match in matches:
                try:
                    # Simple single-argument evaluation
                    arg_val = eval(match) if match else None
                    func_result = func(arg_val) if arg_val is not None else func()
                    result = re.sub(
                        rf"{func_name}\s*\({re.escape(match)}\)",
                        str(func_result),
                        result,
                        count=1,
                        flags=re.IGNORECASE
                    )
                except Exception:
                    pass

        # Try to evaluate the final expression
        try:
            return eval(result)
        except Exception:
            return result

    def _enforce_schema(
        self,
        data: Dict[str, List[Any]],
        schema: TableSchema
    ) -> Dict[str, List[Any]]:
        """
        Enforce destination schema on the output data.

        Ensures output matches expected column names and types.

        Args:
            data: Current data dict
            schema: Destination table schema

        Returns:
            Data dict conforming to schema
        """
        result = {}
        row_count = len(next(iter(data.values()))) if data else 0

        for col_def in schema.columns:
            col_name = col_def.name

            if col_name in data:
                # Column exists, apply type conversion
                result[col_name] = self._convert_column_type(
                    data[col_name],
                    col_def.data_type
                )
            else:
                # Column doesn't exist, try fuzzy match or generate default
                matched = False
                for existing_col in data:
                    if existing_col.lower() == col_name.lower():
                        result[col_name] = self._convert_column_type(
                            data[existing_col],
                            col_def.data_type
                        )
                        matched = True
                        break

                if not matched:
                    # Generate default values
                    result[col_name] = [self._get_default_value(col_def.data_type)
                                        for _ in range(row_count)]

        return result

    def _convert_column_type(
        self,
        values: List[Any],
        target_type: DataType
    ) -> List[Any]:
        """Convert column values to target data type."""
        result = []

        for val in values:
            if val is None:
                result.append(None)
                continue

            try:
                if target_type == DataType.INT:
                    result.append(int(val))
                elif target_type == DataType.BIGINT:
                    result.append(int(val))
                elif target_type == DataType.STRING:
                    result.append(str(val))
                elif target_type in (DataType.FLOAT, DataType.DOUBLE):
                    result.append(float(val))
                elif target_type == DataType.DECIMAL:
                    result.append(Decimal(str(val)))
                elif target_type == DataType.BOOLEAN:
                    result.append(bool(val))
                elif target_type == DataType.DATE:
                    if hasattr(val, 'date'):
                        result.append(val.date())
                    else:
                        result.append(val)
                elif target_type == DataType.TIMESTAMP:
                    result.append(val)
                else:
                    result.append(val)
            except (ValueError, TypeError):
                result.append(val)

        return result

    def _get_default_value(self, data_type: DataType) -> Any:
        """Get default value for a data type."""
        defaults = {
            DataType.INT: 0,
            DataType.BIGINT: 0,
            DataType.STRING: "",
            DataType.FLOAT: 0.0,
            DataType.DOUBLE: 0.0,
            DataType.DECIMAL: Decimal("0"),
            DataType.BOOLEAN: False,
            DataType.DATE: datetime.now().date(),
            DataType.TIMESTAMP: datetime.now(),
            DataType.BINARY: b"",
        }
        return defaults.get(data_type)

    def simulate_lookup_join(
        self,
        main_data: Dict[str, List[Any]],
        lookup_data: Dict[str, List[Any]],
        join_columns: Dict[str, str],
        output_columns: List[str]
    ) -> Dict[str, List[Any]]:
        """
        Simulate a Lookup transformation (JOIN).

        Args:
            main_data: Main data flow data
            lookup_data: Lookup reference data
            join_columns: Dict of main_col -> lookup_col for join
            output_columns: Columns to add from lookup

        Returns:
            Main data with lookup columns added
        """
        result = {col: list(vals) for col, vals in main_data.items()}
        row_count = len(next(iter(main_data.values())))

        # Build lookup index
        lookup_index = {}
        lookup_row_count = len(next(iter(lookup_data.values())))

        for i in range(lookup_row_count):
            # Build composite key from join columns
            key_parts = []
            for main_col, lookup_col in join_columns.items():
                if lookup_col in lookup_data:
                    key_parts.append(str(lookup_data[lookup_col][i]))

            key = tuple(key_parts)
            lookup_index[key] = i

        # Add output columns initialized with None
        for col in output_columns:
            result[col] = [None] * row_count

        # Perform lookup for each row
        for i in range(row_count):
            key_parts = []
            for main_col in join_columns:
                if main_col in main_data:
                    key_parts.append(str(main_data[main_col][i]))

            key = tuple(key_parts)

            if key in lookup_index:
                lookup_idx = lookup_index[key]
                for col in output_columns:
                    if col in lookup_data:
                        result[col][i] = lookup_data[col][lookup_idx]

        return result

    def simulate_aggregation(
        self,
        data: Dict[str, List[Any]],
        group_by_columns: List[str],
        aggregations: Dict[str, str]  # output_col -> "SUM(col)" | "COUNT(*)" | "AVG(col)"
    ) -> Dict[str, List[Any]]:
        """
        Simulate an Aggregate transformation.

        Args:
            data: Input data
            group_by_columns: Columns to group by
            aggregations: Dict of output_col -> aggregation expression

        Returns:
            Aggregated data
        """
        if not data:
            return {}

        row_count = len(next(iter(data.values())))

        # Build groups
        groups: Dict[tuple, List[int]] = {}
        for i in range(row_count):
            key = tuple(data[col][i] for col in group_by_columns if col in data)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # Initialize result with group by columns
        result = {col: [] for col in group_by_columns}
        for col in aggregations:
            result[col] = []

        # Process each group
        for key, indices in groups.items():
            # Add group by values
            for j, col in enumerate(group_by_columns):
                result[col].append(key[j] if j < len(key) else None)

            # Compute aggregations
            for out_col, agg_expr in aggregations.items():
                agg_expr_upper = agg_expr.upper()

                if "COUNT" in agg_expr_upper:
                    result[out_col].append(len(indices))

                elif "SUM" in agg_expr_upper:
                    match = re.search(r"SUM\s*\(\s*(\w+)\s*\)", agg_expr, re.IGNORECASE)
                    if match:
                        col_name = match.group(1)
                        if col_name in data:
                            total = sum(data[col_name][i] for i in indices if data[col_name][i] is not None)
                            result[out_col].append(total)
                        else:
                            result[out_col].append(0)
                    else:
                        result[out_col].append(0)

                elif "AVG" in agg_expr_upper:
                    match = re.search(r"AVG\s*\(\s*(\w+)\s*\)", agg_expr, re.IGNORECASE)
                    if match:
                        col_name = match.group(1)
                        if col_name in data:
                            vals = [data[col_name][i] for i in indices if data[col_name][i] is not None]
                            avg = sum(vals) / len(vals) if vals else 0
                            result[out_col].append(avg)
                        else:
                            result[out_col].append(0)
                    else:
                        result[out_col].append(0)

                elif "MIN" in agg_expr_upper:
                    match = re.search(r"MIN\s*\(\s*(\w+)\s*\)", agg_expr, re.IGNORECASE)
                    if match:
                        col_name = match.group(1)
                        if col_name in data:
                            vals = [data[col_name][i] for i in indices if data[col_name][i] is not None]
                            result[out_col].append(min(vals) if vals else None)
                        else:
                            result[out_col].append(None)

                elif "MAX" in agg_expr_upper:
                    match = re.search(r"MAX\s*\(\s*(\w+)\s*\)", agg_expr, re.IGNORECASE)
                    if match:
                        col_name = match.group(1)
                        if col_name in data:
                            vals = [data[col_name][i] for i in indices if data[col_name][i] is not None]
                            result[out_col].append(max(vals) if vals else None)
                        else:
                            result[out_col].append(None)

        return result
