"""
Synthetic Data Generator

Generates type-matched, semantically-aware synthetic data for validation testing.
Uses extracted schemas to create realistic test data that exercises the
converted code with appropriate edge cases.
"""

import random
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import string

try:
    from faker import Faker
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..extraction.schema_extractor import TableSchema, ColumnDef, DataType


class SyntheticDataGenerator:
    """
    Generate type-matched synthetic data for validation testing.

    Features:
    - Type-aware data generation based on DataType
    - Semantic inference from column names (email, phone, etc.)
    - Foreign key relationship preservation
    - Configurable edge case injection (nulls, boundaries)
    - Deterministic output with seed control
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        random.seed(seed)

        if HAS_FAKER:
            self.faker = Faker()
            Faker.seed(seed)
        else:
            self.faker = None

    def generate_table(
        self,
        schema: TableSchema,
        row_count: int = 1000,
        related_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Any]]:
        """
        Generate synthetic data for a table schema.

        Args:
            schema: Table schema definition
            row_count: Number of rows to generate
            related_data: Dict of table_name -> data dict for FK lookups

        Returns:
            Dict mapping column names to lists of values
        """
        data = {}

        for col in schema.columns:
            # Check for foreign key reference
            if col.foreign_key_ref and related_data:
                ref_table, ref_col = col.foreign_key_ref.split(".")
                if ref_table in related_data and ref_col in related_data[ref_table]:
                    ref_values = related_data[ref_table][ref_col]
                    data[col.name] = [random.choice(ref_values) for _ in range(row_count)]
                    continue

            # Generate column data
            data[col.name] = self._generate_column(col, row_count)

        return data

    def generate_table_pandas(
        self,
        schema: TableSchema,
        row_count: int = 1000,
        related_data: Optional[Dict[str, Any]] = None
    ):
        """
        Generate synthetic data as a pandas DataFrame.

        Args:
            schema: Table schema definition
            row_count: Number of rows to generate
            related_data: Dict of table_name -> DataFrame for FK lookups

        Returns:
            pandas DataFrame with generated data
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for generate_table_pandas()")

        data = self.generate_table(schema, row_count, related_data)
        return pd.DataFrame(data)

    def _generate_column(self, col: ColumnDef, count: int) -> List[Any]:
        """
        Generate values for a single column based on type and semantics.

        Tries semantic inference first, falls back to type-based generation.
        """
        name_lower = col.name.lower()

        # Identity/Primary key columns
        if col.is_identity or col.is_primary_key:
            return list(range(1, count + 1))

        # Semantic inference from column name
        semantic_value = self._generate_by_semantic(name_lower, count)
        if semantic_value is not None:
            return semantic_value

        # Fall back to type-based generation
        return self._generate_by_type(col, count)

    def _generate_by_semantic(self, name_lower: str, count: int) -> Optional[List[Any]]:
        """Generate data based on semantic meaning inferred from column name."""

        # ID columns (foreign keys)
        if name_lower.endswith("_id") or (name_lower.endswith("id") and len(name_lower) > 2):
            return [random.randint(1, 100) for _ in range(count)]

        # Email
        if "email" in name_lower:
            if self.faker:
                return [self.faker.email() for _ in range(count)]
            return [f"user{i}@example.com" for i in range(count)]

        # Phone
        if "phone" in name_lower or "tel" in name_lower:
            if self.faker:
                return [self.faker.phone_number()[:15] for _ in range(count)]
            return [f"555-{random.randint(100,999)}-{random.randint(1000,9999)}" for _ in range(count)]

        # Names
        if "name" in name_lower:
            if "first" in name_lower:
                if self.faker:
                    return [self.faker.first_name() for _ in range(count)]
                return [f"FirstName{i}" for i in range(count)]
            if "last" in name_lower:
                if self.faker:
                    return [self.faker.last_name() for _ in range(count)]
                return [f"LastName{i}" for i in range(count)]
            if "company" in name_lower or "customer" in name_lower:
                if self.faker:
                    return [self.faker.company() for _ in range(count)]
                return [f"Company{i} Inc." for i in range(count)]
            if "product" in name_lower:
                products = ["Widget", "Gadget", "Tool", "Device", "Component", "Module"]
                return [f"{random.choice(products)} {random.randint(100, 999)}" for _ in range(count)]
            if self.faker:
                return [self.faker.name() for _ in range(count)]
            return [f"Name{i}" for i in range(count)]

        # Address fields
        if "address" in name_lower or "street" in name_lower:
            if self.faker:
                return [self.faker.street_address() for _ in range(count)]
            return [f"{random.randint(1,999)} Main St" for _ in range(count)]

        if "city" in name_lower:
            if self.faker:
                return [self.faker.city() for _ in range(count)]
            cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle"]
            return [random.choice(cities) for _ in range(count)]

        if "state" in name_lower:
            if self.faker:
                return [self.faker.state_abbr() for _ in range(count)]
            states = ["CA", "NY", "TX", "FL", "WA", "IL", "PA", "OH"]
            return [random.choice(states) for _ in range(count)]

        if "postal" in name_lower or "zip" in name_lower:
            if self.faker:
                return [self.faker.zipcode() for _ in range(count)]
            return [f"{random.randint(10000, 99999)}" for _ in range(count)]

        if "country" in name_lower:
            if self.faker:
                return [self.faker.country_code() for _ in range(count)]
            countries = ["US", "CA", "GB", "DE", "FR", "AU"]
            return [random.choice(countries) for _ in range(count)]

        # Financial fields
        if "price" in name_lower or "amount" in name_lower or "cost" in name_lower or "total" in name_lower:
            return [round(random.uniform(1.0, 1000.0), 2) for _ in range(count)]

        if "quantity" in name_lower or "qty" in name_lower or "count" in name_lower:
            return [random.randint(1, 100) for _ in range(count)]

        if "discount" in name_lower or "percent" in name_lower or "rate" in name_lower:
            return [round(random.uniform(0, 30), 1) for _ in range(count)]

        # Date fields
        if "date" in name_lower or "created" in name_lower or "modified" in name_lower or "updated" in name_lower:
            base = datetime(2024, 1, 1)
            return [base + timedelta(days=random.randint(0, 365)) for _ in range(count)]

        # Category/Segment fields
        if "category" in name_lower:
            categories = ["Electronics", "Clothing", "Food", "Home", "Sports", "Books"]
            return [random.choice(categories) for _ in range(count)]

        if "segment" in name_lower:
            segments = ["Premium", "Standard", "Basic", "Enterprise", "SMB"]
            return [random.choice(segments) for _ in range(count)]

        if "status" in name_lower:
            statuses = ["Active", "Inactive", "Pending", "Completed", "Cancelled"]
            return [random.choice(statuses) for _ in range(count)]

        if "type" in name_lower:
            types = ["Type A", "Type B", "Type C", "Standard", "Custom"]
            return [random.choice(types) for _ in range(count)]

        if "region" in name_lower:
            regions = ["North", "South", "East", "West", "Central"]
            return [random.choice(regions) for _ in range(count)]

        # Boolean-like fields
        if "is_" in name_lower or "has_" in name_lower or name_lower.startswith("is") or "flag" in name_lower:
            return [random.choice([True, False]) for _ in range(count)]

        if "current" in name_lower:
            # Mostly True for "IsCurrent" type fields
            return [random.random() > 0.2 for _ in range(count)]

        # Description/Notes
        if "description" in name_lower or "notes" in name_lower or "comment" in name_lower:
            if self.faker:
                return [self.faker.sentence() for _ in range(count)]
            return [f"Description for record {i}" for i in range(count)]

        return None

    def _generate_by_type(self, col: ColumnDef, count: int) -> List[Any]:
        """Generate data based purely on data type."""

        if col.data_type == DataType.INT:
            return [random.randint(1, 10000) for _ in range(count)]

        if col.data_type == DataType.BIGINT:
            return [random.randint(1, 1000000) for _ in range(count)]

        if col.data_type == DataType.STRING:
            if self.faker:
                return [self.faker.word() for _ in range(count)]
            return [self._random_string(10) for _ in range(count)]

        if col.data_type in (DataType.FLOAT, DataType.DOUBLE):
            return [round(random.uniform(0, 1000), 4) for _ in range(count)]

        if col.data_type == DataType.DECIMAL:
            precision = col.precision or 10
            scale = col.scale or 2
            max_val = 10 ** (precision - scale)
            return [round(Decimal(str(random.uniform(0, max_val))), scale) for _ in range(count)]

        if col.data_type == DataType.DATE:
            base = date(2024, 1, 1)
            return [base + timedelta(days=random.randint(0, 365)) for _ in range(count)]

        if col.data_type == DataType.TIMESTAMP:
            base = datetime(2024, 1, 1)
            return [base + timedelta(seconds=random.randint(0, 365 * 24 * 3600)) for _ in range(count)]

        if col.data_type == DataType.BOOLEAN:
            return [random.choice([True, False]) for _ in range(count)]

        if col.data_type == DataType.BINARY:
            return [bytes([random.randint(0, 255) for _ in range(16)]) for _ in range(count)]

        # Default to string
        return [self._random_string(10) for _ in range(count)]

    def _random_string(self, length: int) -> str:
        """Generate a random alphanumeric string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def add_edge_cases(
        self,
        data: Dict[str, List[Any]],
        schema: TableSchema,
        null_pct: float = 0.05,
        boundary_pct: float = 0.02
    ) -> Dict[str, List[Any]]:
        """
        Inject edge cases into generated data for thorough testing.

        Edge cases include:
        - NULL values (where nullable)
        - Empty strings
        - Boundary values (min/max int, very long strings)
        - Missing FK references

        Args:
            data: Generated data dictionary
            schema: Table schema
            null_pct: Percentage of values to set as NULL
            boundary_pct: Percentage of values to set as boundary values

        Returns:
            Data dictionary with edge cases injected
        """
        row_count = len(next(iter(data.values())))
        num_nulls = int(row_count * null_pct)
        num_boundaries = int(row_count * boundary_pct)

        for col in schema.columns:
            if col.name not in data:
                continue

            col_data = data[col.name]

            # Inject NULLs for nullable columns
            if col.nullable and num_nulls > 0 and not col.is_primary_key:
                null_indices = random.sample(range(row_count), min(num_nulls, row_count))
                for idx in null_indices:
                    col_data[idx] = None

            # Inject boundary values
            if num_boundaries > 0:
                boundary_indices = random.sample(range(row_count), min(num_boundaries, row_count))

                for idx in boundary_indices:
                    boundary_val = self._get_boundary_value(col)
                    if boundary_val is not None:
                        col_data[idx] = boundary_val

        return data

    def add_edge_cases_pandas(
        self,
        df,
        schema: TableSchema,
        null_pct: float = 0.05,
        boundary_pct: float = 0.02
    ):
        """
        Inject edge cases into a pandas DataFrame.

        Args:
            df: pandas DataFrame
            schema: Table schema
            null_pct: Percentage of values to set as NULL
            boundary_pct: Percentage of values to set as boundary values

        Returns:
            DataFrame with edge cases injected
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for add_edge_cases_pandas()")

        row_count = len(df)
        num_nulls = int(row_count * null_pct)
        num_boundaries = int(row_count * boundary_pct)

        for col in schema.columns:
            if col.name not in df.columns:
                continue

            # Inject NULLs for nullable columns
            if col.nullable and num_nulls > 0 and not col.is_primary_key:
                null_indices = random.sample(range(row_count), min(num_nulls, row_count))
                df.loc[null_indices, col.name] = None

            # Inject boundary values
            if num_boundaries > 0:
                boundary_indices = random.sample(range(row_count), min(num_boundaries, row_count))
                boundary_val = self._get_boundary_value(col)
                if boundary_val is not None:
                    df.loc[boundary_indices, col.name] = boundary_val

        return df

    def _get_boundary_value(self, col: ColumnDef) -> Any:
        """Get a boundary/edge case value for a column type."""

        if col.data_type == DataType.INT:
            return random.choice([0, -1, 2147483647, -2147483648])

        if col.data_type == DataType.BIGINT:
            return random.choice([0, -1, 9223372036854775807])

        if col.data_type == DataType.STRING:
            # Empty string or very long string
            return random.choice(["", "A" * 255, "   ", "NULL"])

        if col.data_type in (DataType.FLOAT, DataType.DOUBLE):
            return random.choice([0.0, -0.0, float('inf') * 0, 0.0000001])  # NaN not included

        if col.data_type == DataType.DECIMAL:
            return Decimal("0.00")

        if col.data_type == DataType.DATE:
            return random.choice([date(1900, 1, 1), date(2099, 12, 31)])

        if col.data_type == DataType.TIMESTAMP:
            return random.choice([datetime(1900, 1, 1), datetime(2099, 12, 31, 23, 59, 59)])

        return None

    def generate_related_tables(
        self,
        schemas: List[TableSchema],
        row_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Generate synthetic data for multiple related tables.

        Handles foreign key relationships by generating parent tables first.

        Args:
            schemas: List of table schemas
            row_counts: Optional dict of table_name -> row count

        Returns:
            Dict mapping table names to their data dicts
        """
        if row_counts is None:
            row_counts = {}

        generated = {}

        # Sort tables to generate parent tables first (simple heuristic)
        # Tables with "dim" in name or fewer FK refs come first
        def sort_key(schema):
            fk_count = sum(1 for c in schema.columns if c.foreign_key_ref)
            is_dim = "dim" in schema.table_name.lower()
            return (fk_count, not is_dim, schema.table_name)

        sorted_schemas = sorted(schemas, key=sort_key)

        for schema in sorted_schemas:
            count = row_counts.get(schema.table_name, 1000)
            data = self.generate_table(schema, count, generated)
            data = self.add_edge_cases(data, schema)
            generated[schema.table_name] = data

        return generated


if __name__ == "__main__":
    # Example usage
    from ..extraction.schema_extractor import SchemaExtractor, DataType

    # Create a sample schema
    schema = TableSchema(
        table_name="Sales",
        columns=[
            ColumnDef(name="SaleID", data_type=DataType.INT, is_primary_key=True, is_identity=True),
            ColumnDef(name="CustomerID", data_type=DataType.INT),
            ColumnDef(name="ProductID", data_type=DataType.INT),
            ColumnDef(name="SaleDate", data_type=DataType.TIMESTAMP),
            ColumnDef(name="Quantity", data_type=DataType.INT),
            ColumnDef(name="UnitPrice", data_type=DataType.DECIMAL, precision=10, scale=2),
            ColumnDef(name="DiscountPercent", data_type=DataType.FLOAT, nullable=True),
        ],
        role="source"
    )

    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_table(schema, row_count=100)
    data = generator.add_edge_cases(data, schema)

    print(f"Generated {len(data['SaleID'])} rows")
    for col in schema.columns:
        print(f"  {col.name}: {data[col.name][:3]}...")
