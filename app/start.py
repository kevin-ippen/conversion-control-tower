"""
Databricks App entry point for Conversion Control Tower.

This file is executed by the Databricks Apps runtime.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add backend and parent (for src/) to path
app_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(app_dir, "backend"))
sys.path.insert(0, os.path.dirname(app_dir))  # For src/ imports


def main():
    """Start the application server."""
    import uvicorn
    from app.main import app

    # Get port from environment (Databricks Apps sets this)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting Conversion Control Tower on {host}:{port}")
    logger.info(f"CATALOG: {os.environ.get('CATALOG', 'dev_conversion_tracker')}")
    logger.info(f"SOURCE_CATALOG: {os.environ.get('SOURCE_CATALOG', 'dev_conversion_tracker')}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
