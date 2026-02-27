"""
Conversion Control Tower - FastAPI Application

Conversion Control Tower — Conversion Control Tower application.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from .config import get_settings
from .routers import conversions, files, validation, workflows, analytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Catalog: {settings.catalog}")
    logger.info(f"Source Catalog: {settings.source_catalog}")
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title="Conversion Control Tower",
    description="Conversion Control Tower — migrate SQL Server, SSIS, and Informatica to Databricks",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(conversions.router, prefix="/api/conversions", tags=["conversions"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(validation.router, prefix="/api/validation", tags=["validation"])
app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "app": settings.app_name,
        "catalog": settings.catalog,
    }


@app.get("/api")
async def api_root():
    """API root - list available endpoints."""
    return {
        "message": "Conversion Control Tower API",
        "endpoints": {
            "conversions": "/api/conversions - Manage conversion jobs",
            "files": "/api/files - Upload and manage files",
            "validation": "/api/validation - Run and view validations",
            "workflows": "/api/workflows - Deploy and promote workflows",
            "analytics": "/api/analytics - View aggregate metrics",
        },
        "docs": "/docs",
    }


# Determine static files directory
# Try multiple possible locations for Databricks Apps
def find_static_dir():
    candidates = [
        Path(__file__).parent.parent / "static",  # Relative to main.py
        Path(os.getcwd()) / "backend" / "static",  # From app root
        Path("/app/backend/static"),  # Absolute in container
        Path(os.getcwd()) / "static",  # Direct in cwd
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "index.html").exists():
            return candidate
    return None


static_dir = find_static_dir()
if static_dir:
    # Mount static assets (CSS, JS)
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        logger.info(f"Serving assets from {assets_dir}")

    # Serve index.html for root
    @app.get("/", response_class=HTMLResponse)
    async def serve_root():
        index_path = static_dir / "index.html"
        return FileResponse(str(index_path))

    # Catch-all for SPA routes (must be AFTER API routes)
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api/") or full_path in ["health", "docs", "openapi.json", "redoc"]:
            return {"detail": "Not Found"}

        # Check if it's a static file request
        file_path = static_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        # Otherwise serve index.html for SPA routing
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"detail": "Not Found"}

    logger.info(f"Serving frontend from {static_dir}")
else:
    logger.info("No static folder found, running API-only mode")
