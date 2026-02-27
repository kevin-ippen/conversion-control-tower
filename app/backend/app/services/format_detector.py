"""
Auto-detect source file format based on filename and content.
"""

import re
import logging
from typing import Optional

from ..models.conversion import SourceType

logger = logging.getLogger(__name__)

# Patterns for detecting stored procedures in SQL files
STORED_PROC_PATTERNS = [
    re.compile(r'\bCREATE\s+(OR\s+ALTER\s+)?PROC(EDURE)?\b', re.IGNORECASE),
    re.compile(r'\bCREATE\s+(OR\s+ALTER\s+)?FUNCTION\b', re.IGNORECASE),
]

# Patterns for detecting Informatica PowerCenter XML
INFORMATICA_ROOT_TAGS = ['POWERMART', 'REPOSITORY']


def detect_format(filename: str, content_preview: Optional[str] = None) -> SourceType:
    """Detect the source format from filename and optional content preview.

    Args:
        filename: The source file name (e.g., "package.dtsx", "proc.sql")
        content_preview: First ~2000 chars of file content for deeper inspection

    Returns:
        Detected SourceType
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    # DTSX files are always SSIS
    if ext == 'dtsx':
        return SourceType.SSIS

    # XML files - check for Informatica PowerCenter
    if ext == 'xml' and content_preview:
        for tag in INFORMATICA_ROOT_TAGS:
            if f'<{tag}' in content_preview.upper():
                logger.info(f"Detected Informatica PowerCenter XML from <{tag}> root tag")
                return SourceType.INFORMATICA_PC
        # XML but not Informatica - could be SSIS without .dtsx extension
        if '<DTS:Executable' in content_preview or 'DTS:' in content_preview:
            return SourceType.SSIS

    # SQL files - check for stored procedures
    if ext == 'sql':
        if content_preview:
            for pattern in STORED_PROC_PATTERNS:
                if pattern.search(content_preview):
                    logger.info("Detected stored procedure from CREATE PROC/FUNCTION pattern")
                    return SourceType.STORED_PROC
        return SourceType.SQL_SCRIPT

    # TXT files - try to infer from content
    if ext == 'txt' and content_preview:
        # Check if it looks like SQL
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER']
        upper_preview = content_preview.upper()
        if any(kw in upper_preview for kw in sql_keywords):
            for pattern in STORED_PROC_PATTERNS:
                if pattern.search(content_preview):
                    return SourceType.STORED_PROC
            return SourceType.SQL_SCRIPT

    # Default to SQL script
    logger.info(f"Could not auto-detect format for {filename}, defaulting to sql_script")
    return SourceType.SQL_SCRIPT
