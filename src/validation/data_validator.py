"""
Data Validator

Compares expected vs actual output data for validation testing.
Provides detailed comparison results including schema match, row counts,
value matches, and comprehensive diff reporting.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class DataComparisonResult:
    """Result of comparing expected vs actual data."""

    # Schema comparison
    schema_match: bool = False
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    type_mismatches: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # Row comparison
    row_count_expected: int = 0
    row_count_actual: int = 0
    row_count_match: bool = False

    # Value comparison
    total_cells: int = 0
    matching_cells: int = 0
    match_rate: float = 0.0

    # Detailed mismatches (sample)
    mismatched_rows: List[Dict] = field(default_factory=list)
    mismatched_columns: Dict[str, int] = field(default_factory=dict)

    # Summary
    overall_pass: bool = False
    pass_threshold: float = 0.95
    summary: str = ""
    detailed_summary: str = ""

    # Metadata
    comparison_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert tuple values to lists for JSON serialization
        result["type_mismatches"] = {k: list(v) for k, v in self.type_mismatches.items()}
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DataComparisonResult":
        """Create from JSON string."""
        data = json.loads(json_str)
        # Convert type_mismatches back to tuples
        data["type_mismatches"] = {k: tuple(v) for k, v in data.get("type_mismatches", {}).items()}
        return cls(**data)


class DataValidator:
    """
    Compare expected vs actual output data for validation.

    Features:
    - Schema comparison (columns, types)
    - Row count comparison
    - Cell-by-cell value comparison with configurable tolerance
    - Detailed mismatch reporting
    - Configurable pass threshold
    """

    def __init__(
        self,
        float_tolerance: float = 0.0001,
        string_case_sensitive: bool = False,
        null_equals_null: bool = True,
        pass_threshold: float = 0.95
    ):
        """
        Initialize the validator.

        Args:
            float_tolerance: Tolerance for float comparisons
            string_case_sensitive: Whether string comparisons are case-sensitive
            null_equals_null: Whether NULL == NULL
            pass_threshold: Minimum match rate to pass validation
        """
        self.float_tolerance = float_tolerance
        self.string_case_sensitive = string_case_sensitive
        self.null_equals_null = null_equals_null
        self.pass_threshold = pass_threshold

    def compare(
        self,
        expected: Dict[str, List[Any]],
        actual: Dict[str, List[Any]],
        key_columns: Optional[List[str]] = None,
        max_mismatches: int = 100
    ) -> DataComparisonResult:
        """
        Compare two data dictionaries and return detailed comparison result.

        Args:
            expected: Expected output (from original simulator)
            actual: Actual output (from converted code)
            key_columns: Columns to use for row matching (if None, compares by position)
            max_mismatches: Maximum number of mismatches to record in detail

        Returns:
            DataComparisonResult with detailed comparison info
        """
        result = DataComparisonResult(pass_threshold=self.pass_threshold)

        # Schema comparison
        expected_cols = set(expected.keys())
        actual_cols = set(actual.keys())

        result.missing_columns = list(expected_cols - actual_cols)
        result.extra_columns = list(actual_cols - expected_cols)
        common_cols = expected_cols & actual_cols

        # Type comparison for common columns
        result.type_mismatches = self._compare_types(expected, actual, common_cols)

        result.schema_match = (
            len(result.missing_columns) == 0 and
            len(result.type_mismatches) == 0
        )

        # Row count comparison
        result.row_count_expected = len(next(iter(expected.values()))) if expected else 0
        result.row_count_actual = len(next(iter(actual.values()))) if actual else 0
        result.row_count_match = result.row_count_expected == result.row_count_actual

        # Value comparison
        if key_columns:
            self._compare_by_key(expected, actual, common_cols, key_columns, result, max_mismatches)
        else:
            self._compare_by_position(expected, actual, common_cols, result, max_mismatches)

        # Calculate match rate
        if result.total_cells > 0:
            result.match_rate = result.matching_cells / result.total_cells
        else:
            result.match_rate = 1.0 if not expected and not actual else 0.0

        # Determine overall pass
        result.overall_pass = (
            result.schema_match and
            result.row_count_match and
            result.match_rate >= self.pass_threshold
        )

        # Generate summary
        result.summary = self._generate_summary(result)
        result.detailed_summary = self._generate_detailed_summary(result)

        return result

    def compare_pandas(
        self,
        expected_df,
        actual_df,
        key_columns: Optional[List[str]] = None,
        max_mismatches: int = 100
    ) -> DataComparisonResult:
        """
        Compare two pandas DataFrames.

        Args:
            expected_df: Expected output DataFrame
            actual_df: Actual output DataFrame
            key_columns: Columns to use for row matching
            max_mismatches: Maximum mismatches to record

        Returns:
            DataComparisonResult with detailed comparison info
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for compare_pandas()")

        # Convert to dicts
        expected = expected_df.to_dict('list') if isinstance(expected_df, pd.DataFrame) else expected_df
        actual = actual_df.to_dict('list') if isinstance(actual_df, pd.DataFrame) else actual_df

        return self.compare(expected, actual, key_columns, max_mismatches)

    def _compare_types(
        self,
        expected: Dict[str, List[Any]],
        actual: Dict[str, List[Any]],
        columns: set
    ) -> Dict[str, Tuple[str, str]]:
        """Compare column types between expected and actual."""
        mismatches = {}

        for col in columns:
            exp_type = self._infer_type(expected[col])
            act_type = self._infer_type(actual[col])

            if not self._types_compatible(exp_type, act_type):
                mismatches[col] = (exp_type, act_type)

        return mismatches

    def _infer_type(self, values: List[Any]) -> str:
        """Infer the data type from a list of values."""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return "null"

        sample = non_null[0]
        if isinstance(sample, bool):
            return "bool"
        if isinstance(sample, int):
            return "int"
        if isinstance(sample, float):
            return "float"
        if isinstance(sample, str):
            return "string"
        if hasattr(sample, 'date') and hasattr(sample, 'hour'):
            return "datetime"
        if hasattr(sample, 'year') and hasattr(sample, 'month'):
            return "date"
        return type(sample).__name__

    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible."""
        numeric_types = {"int", "float", "int64", "float64", "int32", "float32", "decimal"}
        datetime_types = {"datetime", "timestamp", "date"}
        string_types = {"string", "str", "object"}

        if type1 in numeric_types and type2 in numeric_types:
            return True
        if type1 in datetime_types and type2 in datetime_types:
            return True
        if type1 in string_types and type2 in string_types:
            return True
        return type1 == type2

    def _compare_by_position(
        self,
        expected: Dict[str, List[Any]],
        actual: Dict[str, List[Any]],
        columns: set,
        result: DataComparisonResult,
        max_mismatches: int
    ):
        """Compare data by row position."""
        min_rows = min(result.row_count_expected, result.row_count_actual)

        for col in columns:
            result.mismatched_columns[col] = 0

            for i in range(min_rows):
                result.total_cells += 1
                exp_val = expected[col][i] if i < len(expected[col]) else None
                act_val = actual[col][i] if i < len(actual[col]) else None

                if self._values_match(exp_val, act_val):
                    result.matching_cells += 1
                else:
                    result.mismatched_columns[col] += 1
                    if len(result.mismatched_rows) < max_mismatches:
                        result.mismatched_rows.append({
                            "row": i,
                            "column": col,
                            "expected": self._serialize_value(exp_val),
                            "actual": self._serialize_value(act_val)
                        })

    def _compare_by_key(
        self,
        expected: Dict[str, List[Any]],
        actual: Dict[str, List[Any]],
        columns: set,
        key_columns: List[str],
        result: DataComparisonResult,
        max_mismatches: int
    ):
        """Compare data by key columns (similar to a JOIN)."""
        # Build index for actual data
        actual_index = {}
        for i in range(result.row_count_actual):
            key = tuple(actual[k][i] for k in key_columns if k in actual)
            actual_index[key] = i

        # Compare each expected row
        non_key_cols = [c for c in columns if c not in key_columns]

        for col in non_key_cols:
            result.mismatched_columns[col] = 0

        for i in range(result.row_count_expected):
            key = tuple(expected[k][i] for k in key_columns if k in expected)

            if key not in actual_index:
                # Row not found in actual
                for col in non_key_cols:
                    result.total_cells += 1
                    result.mismatched_columns[col] = result.mismatched_columns.get(col, 0) + 1
                    if len(result.mismatched_rows) < max_mismatches:
                        result.mismatched_rows.append({
                            "row": i,
                            "column": col,
                            "expected": self._serialize_value(expected[col][i]),
                            "actual": "ROW_NOT_FOUND",
                            "key": str(key)
                        })
            else:
                actual_i = actual_index[key]
                for col in non_key_cols:
                    result.total_cells += 1
                    exp_val = expected[col][i]
                    act_val = actual[col][actual_i]

                    if self._values_match(exp_val, act_val):
                        result.matching_cells += 1
                    else:
                        result.mismatched_columns[col] = result.mismatched_columns.get(col, 0) + 1
                        if len(result.mismatched_rows) < max_mismatches:
                            result.mismatched_rows.append({
                                "row": i,
                                "column": col,
                                "expected": self._serialize_value(exp_val),
                                "actual": self._serialize_value(act_val),
                                "key": str(key)
                            })

    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Check if two values match."""
        # Handle nulls
        if expected is None and actual is None:
            return self.null_equals_null
        if expected is None or actual is None:
            return False

        # Handle NaN
        try:
            if str(expected) == 'nan' and str(actual) == 'nan':
                return self.null_equals_null
        except Exception:
            pass

        # Handle floats with tolerance
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                return abs(float(expected) - float(actual)) <= self.float_tolerance
            except (ValueError, TypeError):
                return False

        # Handle strings
        if isinstance(expected, str) and isinstance(actual, str):
            if self.string_case_sensitive:
                return expected == actual
            return expected.lower() == actual.lower()

        # Handle datetime comparison
        if hasattr(expected, 'isoformat') and hasattr(actual, 'isoformat'):
            return expected == actual

        # Direct comparison
        return expected == actual

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON output."""
        if value is None:
            return None
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.hex()
        return value

    def _generate_summary(self, result: DataComparisonResult) -> str:
        """Generate a short summary string."""
        parts = []

        if not result.schema_match:
            if result.missing_columns:
                parts.append(f"Missing columns: {', '.join(result.missing_columns[:3])}")
            if result.extra_columns:
                parts.append(f"Extra columns: {', '.join(result.extra_columns[:3])}")
            if result.type_mismatches:
                parts.append(f"Type mismatches: {len(result.type_mismatches)}")

        if not result.row_count_match:
            parts.append(f"Row count mismatch: {result.row_count_expected} vs {result.row_count_actual}")

        parts.append(f"Data match rate: {result.match_rate * 100:.1f}%")

        status = "PASS" if result.overall_pass else "FAIL"
        return f"[{status}] " + "; ".join(parts) if parts else f"[{status}] Perfect match"

    def _generate_detailed_summary(self, result: DataComparisonResult) -> str:
        """Generate a detailed summary string."""
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            "",
            f"Overall Result: {'PASS' if result.overall_pass else 'FAIL'}",
            f"Match Rate: {result.match_rate * 100:.2f}% (threshold: {result.pass_threshold * 100:.0f}%)",
            "",
            "SCHEMA COMPARISON",
            "-" * 40,
            f"Schema Match: {'Yes' if result.schema_match else 'No'}",
        ]

        if result.missing_columns:
            lines.append(f"Missing Columns: {', '.join(result.missing_columns)}")
        if result.extra_columns:
            lines.append(f"Extra Columns: {', '.join(result.extra_columns)}")
        if result.type_mismatches:
            lines.append("Type Mismatches:")
            for col, (exp_t, act_t) in result.type_mismatches.items():
                lines.append(f"  - {col}: expected {exp_t}, got {act_t}")

        lines.extend([
            "",
            "ROW COMPARISON",
            "-" * 40,
            f"Expected Rows: {result.row_count_expected}",
            f"Actual Rows: {result.row_count_actual}",
            f"Row Count Match: {'Yes' if result.row_count_match else 'No'}",
            "",
            "VALUE COMPARISON",
            "-" * 40,
            f"Total Cells Compared: {result.total_cells}",
            f"Matching Cells: {result.matching_cells}",
            f"Mismatched Cells: {result.total_cells - result.matching_cells}",
        ])

        if result.mismatched_columns:
            lines.append("")
            lines.append("Mismatches by Column:")
            for col, count in sorted(result.mismatched_columns.items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"  - {col}: {count} mismatches")

        if result.mismatched_rows:
            lines.extend([
                "",
                "SAMPLE MISMATCHES",
                "-" * 40,
            ])
            for mismatch in result.mismatched_rows[:10]:
                lines.append(f"  Row {mismatch['row']}, {mismatch['column']}:")
                lines.append(f"    Expected: {mismatch['expected']}")
                lines.append(f"    Actual: {mismatch['actual']}")

            if len(result.mismatched_rows) > 10:
                lines.append(f"  ... and {len(result.mismatched_rows) - 10} more mismatches")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


def validate_outputs(
    expected_path: str,
    actual_path: str,
    output_path: str,
    key_columns: Optional[List[str]] = None,
    pass_threshold: float = 0.95
) -> DataComparisonResult:
    """
    Validate outputs from file paths (convenience function).

    Args:
        expected_path: Path to expected output (parquet)
        actual_path: Path to actual output (parquet)
        output_path: Path to write comparison report
        key_columns: Columns for row matching
        pass_threshold: Minimum match rate to pass

    Returns:
        DataComparisonResult
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for validate_outputs()")

    expected_df = pd.read_parquet(expected_path)
    actual_df = pd.read_parquet(actual_path)

    validator = DataValidator(pass_threshold=pass_threshold)
    result = validator.compare_pandas(expected_df, actual_df, key_columns)

    # Write report
    with open(output_path, 'w') as f:
        f.write(result.to_json())

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_validator.py <expected_path> <actual_path> [output_path]")
        sys.exit(1)

    expected_path = sys.argv[1]
    actual_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "comparison_report.json"

    result = validate_outputs(expected_path, actual_path, output_path)
    print(result.detailed_summary)
    sys.exit(0 if result.overall_pass else 1)
