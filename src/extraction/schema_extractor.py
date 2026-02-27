"""
Schema Extractor for SSIS Packages and SQL Scripts

Extracts source and destination table schemas from uploaded scripts
to enable dynamic synthetic data generation.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path


class DataType(str, Enum):
    """Unified data type enum for schema definitions."""
    INT = "int"
    BIGINT = "bigint"
    STRING = "string"
    FLOAT = "float"
    DOUBLE = "double"
    DECIMAL = "decimal"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    BINARY = "binary"


@dataclass
class ColumnDef:
    """Definition of a single column."""
    name: str
    data_type: DataType
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    is_primary_key: bool = False
    is_identity: bool = False
    foreign_key_ref: Optional[str] = None  # table.column format

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "precision": self.precision,
            "scale": self.scale,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_identity": self.is_identity,
            "foreign_key_ref": self.foreign_key_ref,
        }


@dataclass
class TableSchema:
    """Schema definition for a table."""
    table_name: str
    columns: List[ColumnDef] = field(default_factory=list)
    role: str = "source"  # 'source', 'destination', 'lookup', 'staging'
    database: Optional[str] = None
    schema_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": [c.to_dict() for c in self.columns],
            "role": self.role,
            "database": self.database,
            "schema_name": self.schema_name,
        }


@dataclass
class ExtractedSchemas:
    """Container for all extracted schemas from a source file."""
    source_tables: List[TableSchema] = field(default_factory=list)
    destination_tables: List[TableSchema] = field(default_factory=list)
    lookup_tables: List[TableSchema] = field(default_factory=list)
    transformations: Dict[str, str] = field(default_factory=dict)  # source_col -> dest_col

    def to_json(self) -> str:
        return json.dumps({
            "source_tables": [t.to_dict() for t in self.source_tables],
            "destination_tables": [t.to_dict() for t in self.destination_tables],
            "lookup_tables": [t.to_dict() for t in self.lookup_tables],
            "transformations": self.transformations,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ExtractedSchemas":
        data = json.loads(json_str)

        def parse_column(col_data: Dict) -> ColumnDef:
            return ColumnDef(
                name=col_data["name"],
                data_type=DataType(col_data["data_type"]),
                precision=col_data.get("precision"),
                scale=col_data.get("scale"),
                nullable=col_data.get("nullable", True),
                is_primary_key=col_data.get("is_primary_key", False),
                is_identity=col_data.get("is_identity", False),
                foreign_key_ref=col_data.get("foreign_key_ref"),
            )

        def parse_table(table_data: Dict) -> TableSchema:
            return TableSchema(
                table_name=table_data["table_name"],
                columns=[parse_column(c) for c in table_data.get("columns", [])],
                role=table_data.get("role", "source"),
                database=table_data.get("database"),
                schema_name=table_data.get("schema_name"),
            )

        return cls(
            source_tables=[parse_table(t) for t in data.get("source_tables", [])],
            destination_tables=[parse_table(t) for t in data.get("destination_tables", [])],
            lookup_tables=[parse_table(t) for t in data.get("lookup_tables", [])],
            transformations=data.get("transformations", {}),
        )


class SchemaExtractor:
    """
    Extract table schemas from SSIS packages and SQL scripts.

    Used to generate type-matched synthetic data for validation testing.
    """

    # SQL Server to unified DataType mapping
    SQL_TYPE_MAP = {
        # Integer types
        "int": DataType.INT,
        "integer": DataType.INT,
        "smallint": DataType.INT,
        "tinyint": DataType.INT,
        "bigint": DataType.BIGINT,

        # String types
        "varchar": DataType.STRING,
        "nvarchar": DataType.STRING,
        "char": DataType.STRING,
        "nchar": DataType.STRING,
        "text": DataType.STRING,
        "ntext": DataType.STRING,

        # Numeric types
        "decimal": DataType.DECIMAL,
        "numeric": DataType.DECIMAL,
        "money": DataType.DECIMAL,
        "smallmoney": DataType.DECIMAL,
        "float": DataType.FLOAT,
        "real": DataType.FLOAT,

        # Date/time types
        "datetime": DataType.TIMESTAMP,
        "datetime2": DataType.TIMESTAMP,
        "smalldatetime": DataType.TIMESTAMP,
        "date": DataType.DATE,
        "time": DataType.STRING,  # Store as string for simplicity
        "datetimeoffset": DataType.TIMESTAMP,

        # Boolean
        "bit": DataType.BOOLEAN,

        # Binary
        "binary": DataType.BINARY,
        "varbinary": DataType.BINARY,
        "image": DataType.BINARY,

        # Other
        "uniqueidentifier": DataType.STRING,
        "xml": DataType.STRING,
    }

    # SSIS DTS data type codes to unified DataType
    # Supports both numeric codes (from VariableValue) and short names (from outputColumn)
    DTS_TYPE_MAP = {
        # Numeric codes
        "2": DataType.INT,        # DT_I2 (short)
        "3": DataType.INT,        # DT_I4 (int)
        "4": DataType.FLOAT,      # DT_R4 (float)
        "5": DataType.DOUBLE,     # DT_R8 (double)
        "6": DataType.DECIMAL,    # DT_CY (currency)
        "7": DataType.TIMESTAMP,  # DT_DATE
        "8": DataType.STRING,     # DT_BSTR
        "11": DataType.BOOLEAN,   # DT_BOOL
        "14": DataType.DECIMAL,   # DT_DECIMAL
        "16": DataType.INT,       # DT_I1 (byte)
        "17": DataType.INT,       # DT_UI1 (unsigned byte)
        "18": DataType.INT,       # DT_UI2 (unsigned short)
        "19": DataType.INT,       # DT_UI4 (unsigned int)
        "20": DataType.BIGINT,    # DT_I8 (long)
        "21": DataType.BIGINT,    # DT_UI8 (unsigned long)
        "72": DataType.STRING,    # DT_GUID
        "129": DataType.STRING,   # DT_STR (ANSI string)
        "130": DataType.STRING,   # DT_WSTR (Unicode string)
        "131": DataType.DECIMAL,  # DT_NUMERIC
        "133": DataType.DATE,     # DT_DBDATE
        "134": DataType.STRING,   # DT_DBTIME
        "135": DataType.TIMESTAMP,# DT_DBTIMESTAMP
        "141": DataType.STRING,   # DT_XML
        # Short string names (used in component outputColumn attributes)
        "i1": DataType.INT,
        "i2": DataType.INT,
        "i4": DataType.INT,
        "i8": DataType.BIGINT,
        "ui1": DataType.INT,
        "ui2": DataType.INT,
        "ui4": DataType.INT,
        "ui8": DataType.BIGINT,
        "r4": DataType.FLOAT,
        "r8": DataType.DOUBLE,
        "str": DataType.STRING,
        "wstr": DataType.STRING,
        "text": DataType.STRING,
        "ntext": DataType.STRING,
        "bool": DataType.BOOLEAN,
        "numeric": DataType.DECIMAL,
        "decimal": DataType.DECIMAL,
        "cy": DataType.DECIMAL,
        "date": DataType.DATE,
        "dbDate": DataType.DATE,
        "dbTime": DataType.STRING,
        "dbTimeStamp": DataType.TIMESTAMP,
        "dbTimeStamp2": DataType.TIMESTAMP,
        "guid": DataType.STRING,
        "bytes": DataType.BINARY,
        "image": DataType.BINARY,
        "xml": DataType.STRING,
    }

    def extract_from_ssis(self, dtsx_path: str) -> ExtractedSchemas:
        """
        Parse SSIS package to extract source/destination schemas.

        Extracts from:
        - OLE DB Source components → external columns (source tables)
        - OLE DB Destination components → external columns (destination tables)
        - Lookup components → reference tables
        - Derived Column expressions → transformations
        """
        from ..ssis.dtsx_parser import DTSXParser

        parser = DTSXParser(dtsx_path)
        package = parser.parse()

        schemas = ExtractedSchemas()

        # Process all data flow tasks
        for task in package.tasks:
            self._extract_from_task(task, schemas)

        return schemas

    def _extract_from_task(self, task, schemas: ExtractedSchemas):
        """Recursively extract schemas from a task and its children."""
        from ..ssis.dtsx_parser import SSISTask

        if task.data_flow:
            self._extract_from_data_flow(task.data_flow, schemas)

        # Process child tasks (for containers)
        for child in task.child_tasks:
            self._extract_from_task(child, schemas)

    def _extract_from_data_flow(self, data_flow, schemas: ExtractedSchemas):
        """Extract schemas from a data flow's components."""
        for component in data_flow.components:
            comp_type = component.component_type

            if comp_type == "OLEDBSource":
                table_schema = self._extract_source_schema(component)
                if table_schema:
                    schemas.source_tables.append(table_schema)

            elif comp_type == "OLEDBDestination":
                table_schema = self._extract_destination_schema(component)
                if table_schema:
                    schemas.destination_tables.append(table_schema)

            elif comp_type == "Lookup":
                table_schema = self._extract_lookup_schema(component)
                if table_schema:
                    schemas.lookup_tables.append(table_schema)

            elif comp_type == "DerivedColumn":
                transforms = self._extract_derived_transforms(component)
                schemas.transformations.update(transforms)

    def _extract_source_schema(self, component) -> Optional[TableSchema]:
        """Extract schema from an OLE DB Source component."""
        # Get table name from properties
        table_name = component.properties.get("OpenRowset", "")
        if not table_name:
            # Try to extract from SQL command
            sql_cmd = component.properties.get("SqlCommand", "")
            table_name = self._extract_table_from_sql(sql_cmd) or component.name

        # Clean up table name (remove brackets, schema prefix)
        table_name = self._clean_table_name(table_name)

        # Extract columns from output columns
        columns = []
        for col in component.output_columns:
            col_def = self._parse_column_def(col)
            if col_def:
                columns.append(col_def)

        if columns:
            return TableSchema(
                table_name=table_name,
                columns=columns,
                role="source"
            )
        return None

    def _extract_destination_schema(self, component) -> Optional[TableSchema]:
        """Extract schema from an OLE DB Destination component."""
        table_name = component.properties.get("OpenRowset", "")
        if not table_name:
            table_name = component.name

        table_name = self._clean_table_name(table_name)

        # For destination, use input columns (what's being written)
        columns = []
        for col in component.input_columns:
            # Input columns may have less type info, create basic def
            col_def = ColumnDef(
                name=col.get("name", ""),
                data_type=DataType.STRING,  # Default, will be refined
                nullable=True
            )
            columns.append(col_def)

        # Also check output columns for better type info
        for col in component.output_columns:
            col_def = self._parse_column_def(col)
            if col_def:
                # Update existing column with better type info
                for i, existing in enumerate(columns):
                    if existing.name == col_def.name:
                        columns[i] = col_def
                        break
                else:
                    columns.append(col_def)

        if columns:
            return TableSchema(
                table_name=table_name,
                columns=columns,
                role="destination"
            )
        return None

    def _extract_lookup_schema(self, component) -> Optional[TableSchema]:
        """Extract schema from a Lookup component."""
        # Try to get table from SQL command or reference
        sql_cmd = component.properties.get("SqlCommand", "")
        table_name = self._extract_table_from_sql(sql_cmd) or component.name
        table_name = self._clean_table_name(table_name)

        columns = []
        for col in component.output_columns:
            col_def = self._parse_column_def(col)
            if col_def:
                columns.append(col_def)

        if columns:
            return TableSchema(
                table_name=table_name,
                columns=columns,
                role="lookup"
            )
        return None

    def _extract_derived_transforms(self, component) -> Dict[str, str]:
        """Extract column transformations from a Derived Column component."""
        transforms = {}

        for col in component.output_columns:
            col_name = col.get("name", "")
            expression = col.get("expression", "")

            if expression and col_name:
                # Simple mapping: if expression is just a column reference
                # For complex expressions, store the expression
                transforms[col_name] = expression

        return transforms

    def _parse_column_def(self, col_data: Dict) -> Optional[ColumnDef]:
        """Parse column definition from SSIS column data."""
        name = col_data.get("name", "")
        if not name:
            return None

        # Get data type from DTS type code
        dts_type = col_data.get("data_type", "")
        data_type = self.DTS_TYPE_MAP.get(dts_type, DataType.STRING)

        # Extract precision/scale if available
        precision = None
        scale = None

        # Check for precision in properties (value may be None or string)
        if col_data.get("precision") is not None:
            precision = int(col_data["precision"])
        if col_data.get("scale") is not None:
            scale = int(col_data["scale"])

        # Infer if primary key from name convention
        is_pk = name.lower().endswith("id") and name.lower() in ["id", "saleid", "customerid", "productid"]

        return ColumnDef(
            name=name,
            data_type=data_type,
            precision=precision,
            scale=scale,
            nullable=True,
            is_primary_key=is_pk
        )

    def _clean_table_name(self, name: str) -> str:
        """Clean up table name by removing brackets and schema prefixes."""
        # Remove brackets
        name = name.replace("[", "").replace("]", "")

        # If has schema prefix (e.g., dbo.Sales), extract table name
        if "." in name:
            parts = name.split(".")
            return parts[-1]

        return name

    def _extract_table_from_sql(self, sql: str) -> Optional[str]:
        """Extract table name from a SQL query."""
        if not sql:
            return None

        # Pattern for FROM clause
        match = re.search(r'\bFROM\s+(\[?\w+\]?\.)?(\[?\w+\]?)', sql, re.IGNORECASE)
        if match:
            return match.group(2).replace("[", "").replace("]", "")

        return None

    def extract_from_sql(self, sql_path: str) -> ExtractedSchemas:
        """
        Parse SQL stored procedure to extract schemas.

        Extracts from:
        - SELECT FROM clauses → source tables
        - INSERT INTO targets → destination tables
        - CREATE TABLE statements → table definitions
        - Temp table definitions
        """
        with open(sql_path, 'r') as f:
            sql_content = f.read()

        schemas = ExtractedSchemas()

        # Extract CREATE TABLE statements
        create_tables = self._extract_create_tables(sql_content)
        for table_schema in create_tables:
            if table_schema.table_name.startswith("#"):
                table_schema.role = "staging"
                schemas.source_tables.append(table_schema)
            elif "fact" in table_schema.table_name.lower() or "dim" in table_schema.table_name.lower():
                table_schema.role = "destination"
                schemas.destination_tables.append(table_schema)
            else:
                schemas.source_tables.append(table_schema)

        # Extract table references from SELECT statements
        select_tables = self._extract_select_tables(sql_content)
        for table_name in select_tables:
            if not any(t.table_name == table_name for t in schemas.source_tables):
                schemas.source_tables.append(TableSchema(
                    table_name=table_name,
                    role="source"
                ))

        # Extract INSERT INTO targets
        insert_tables = self._extract_insert_tables(sql_content)
        for table_name in insert_tables:
            if not any(t.table_name == table_name for t in schemas.destination_tables):
                schemas.destination_tables.append(TableSchema(
                    table_name=table_name,
                    role="destination"
                ))

        return schemas

    def _extract_create_tables(self, sql: str) -> List[TableSchema]:
        """Extract schemas from CREATE TABLE statements."""
        tables = []

        # Pattern for CREATE TABLE with columns
        pattern = r'CREATE\s+TABLE\s+(\[?\w+\]?\.)?(\[?#?\w+\]?)\s*\((.*?)\)'

        for match in re.finditer(pattern, sql, re.IGNORECASE | re.DOTALL):
            table_name = match.group(2).replace("[", "").replace("]", "")
            columns_str = match.group(3)

            columns = self._parse_column_definitions(columns_str)

            if columns:
                tables.append(TableSchema(
                    table_name=table_name,
                    columns=columns,
                    role="source"
                ))

        return tables

    def _parse_column_definitions(self, columns_str: str) -> List[ColumnDef]:
        """Parse column definitions from CREATE TABLE column list."""
        columns = []

        # Split by comma but respect parentheses
        col_defs = self._split_column_defs(columns_str)

        for col_def in col_defs:
            col_def = col_def.strip()
            if not col_def or col_def.upper().startswith(("PRIMARY", "FOREIGN", "CONSTRAINT", "INDEX", "UNIQUE")):
                continue

            # Parse column name and type
            parts = col_def.split()
            if len(parts) < 2:
                continue

            col_name = parts[0].replace("[", "").replace("]", "")
            type_str = parts[1].replace("[", "").replace("]", "").lower()

            # Parse type and precision
            precision = None
            scale = None

            type_match = re.match(r'(\w+)(?:\((\d+)(?:,\s*(\d+))?\))?', type_str)
            if type_match:
                base_type = type_match.group(1)
                if type_match.group(2):
                    precision = int(type_match.group(2))
                if type_match.group(3):
                    scale = int(type_match.group(3))
            else:
                base_type = type_str

            data_type = self.SQL_TYPE_MAP.get(base_type, DataType.STRING)

            # Check for NOT NULL, PRIMARY KEY, IDENTITY
            col_upper = col_def.upper()
            nullable = "NOT NULL" not in col_upper
            is_pk = "PRIMARY KEY" in col_upper
            is_identity = "IDENTITY" in col_upper

            columns.append(ColumnDef(
                name=col_name,
                data_type=data_type,
                precision=precision,
                scale=scale,
                nullable=nullable,
                is_primary_key=is_pk,
                is_identity=is_identity
            ))

        return columns

    def _split_column_defs(self, columns_str: str) -> List[str]:
        """Split column definitions respecting parentheses."""
        result = []
        current = ""
        paren_depth = 0

        for char in columns_str:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                result.append(current)
                current = ""
                continue
            current += char

        if current.strip():
            result.append(current)

        return result

    def _extract_select_tables(self, sql: str) -> List[str]:
        """Extract table names from SELECT statements."""
        tables = []

        # Pattern for FROM and JOIN clauses
        pattern = r'\b(?:FROM|JOIN)\s+(\[?\w+\]?\.)?(\[?\w+\]?)\b'

        for match in re.finditer(pattern, sql, re.IGNORECASE):
            table_name = match.group(2).replace("[", "").replace("]", "")
            if table_name.lower() not in ["select", "where", "group", "order", "having"]:
                if table_name not in tables:
                    tables.append(table_name)

        return tables

    def _extract_insert_tables(self, sql: str) -> List[str]:
        """Extract table names from INSERT INTO statements."""
        tables = []

        pattern = r'\bINSERT\s+INTO\s+(\[?\w+\]?\.)?(\[?\w+\]?)\b'

        for match in re.finditer(pattern, sql, re.IGNORECASE):
            table_name = match.group(2).replace("[", "").replace("]", "")
            if table_name not in tables:
                tables.append(table_name)

        return tables

    def map_sql_server_type(self, sql_type: str) -> DataType:
        """Convert SQL Server type string to unified DataType."""
        base_type = sql_type.lower().split("(")[0].strip()
        return self.SQL_TYPE_MAP.get(base_type, DataType.STRING)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python schema_extractor.py <path_to_dtsx_or_sql>")
        sys.exit(1)

    extractor = SchemaExtractor()
    path = sys.argv[1]

    if path.endswith(".dtsx"):
        schemas = extractor.extract_from_ssis(path)
    else:
        schemas = extractor.extract_from_sql(path)

    print(schemas.to_json())
