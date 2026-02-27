"""
Shared test fixtures for Conversion Control Tower.

Provides parsed SSIS packages and helper utilities reused across test modules.
"""

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ssis.dtsx_parser import DTSXParser, SSISPackage

# Path to sample SSIS package
SALES_DTSX = PROJECT_ROOT / "samples" / "ssis" / "SalesDataETL.dtsx"


@pytest.fixture(scope="session")
def sales_package() -> SSISPackage:
    """Parse the sample SalesDataETL SSIS package (session-scoped for speed)."""
    assert SALES_DTSX.exists(), f"Sample DTSX not found at {SALES_DTSX}"
    parser = DTSXParser(SALES_DTSX)
    return parser.parse()


@pytest.fixture(scope="session")
def sales_dtsx_path() -> Path:
    """Path to the Sales DTSX file."""
    return SALES_DTSX
