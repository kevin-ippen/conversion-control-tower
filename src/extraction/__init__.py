"""Schema extraction from SSIS packages and SQL scripts."""

from .schema_extractor import (
    SchemaExtractor,
    DataType,
    ColumnDef,
    TableSchema,
    ExtractedSchemas,
)

__all__ = [
    "SchemaExtractor",
    "DataType",
    "ColumnDef",
    "TableSchema",
    "ExtractedSchemas",
]
