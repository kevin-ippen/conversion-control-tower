"""
API Router for file management (UC Volumes).
"""

import logging
from typing import List, Optional
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..auth import get_user_context, UserContext
from ..config import get_settings, Settings
from ..services.volume_manager import VolumeManager
from ..services.source_browser import SourceBrowser
from ..services.format_detector import detect_format
from ..models.conversion import ReferenceFile

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for reference files (replace with UC table in production)
_reference_files_store: dict[str, ReferenceFile] = {}


class FileInfo(BaseModel):
    """Information about a file in UC Volume."""
    name: str
    path: str
    size: int
    is_directory: bool
    modified_at: Optional[str] = None


class UploadResponse(BaseModel):
    """Response after file upload."""
    job_id: str
    file_path: str
    file_name: str
    file_size: int


class DirectoryListing(BaseModel):
    """Directory listing response."""
    path: str
    files: List[FileInfo]


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    job_id: Optional[str] = Form(default=None),
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Upload a source file to UC Volume.

    If job_id is not provided, a new one will be generated.
    """
    # Generate job_id if not provided
    if not job_id:
        job_id = str(uuid.uuid4())

    # Validate file type
    allowed_extensions = {".dtsx", ".sql", ".txt", ".xml"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Upload to UC Volume
    volume_manager = VolumeManager(user.workspace_client, settings)
    file_content = await file.read()

    upload_path = f"{settings.uploads_path}/{job_id}/{file.filename}"
    await volume_manager.write_file(upload_path, file_content)

    logger.info(f"Uploaded file {file.filename} to {upload_path}")

    return UploadResponse(
        job_id=job_id,
        file_path=upload_path,
        file_name=file.filename,
        file_size=len(file_content),
    )


# ===== Browse Source Locations =====

class BrowseRequest(BaseModel):
    """Request to browse a directory."""
    path: str


class BrowseResponse(BaseModel):
    """Directory listing from a source location."""
    path: str
    items: List[dict]


class DetectFormatRequest(BaseModel):
    """Request to auto-detect file format."""
    filename: str
    path: Optional[str] = None  # Full path for content preview


class DetectFormatResponse(BaseModel):
    """Auto-detected format result."""
    filename: str
    detected_format: str
    confidence: str  # "high" (extension-based) or "medium" (content-based)


class IngestRemoteRequest(BaseModel):
    """Request to copy a remote file into job upload path."""
    source_path: str  # Workspace or Volume path
    source_type: str  # "workspace" or "volume"
    job_id: Optional[str] = None


@router.post("/browse-workspace", response_model=BrowseResponse)
async def browse_workspace(
    request: BrowseRequest,
    user: UserContext = Depends(get_user_context),
):
    """Browse files in a Databricks Workspace directory."""
    browser = SourceBrowser(user.workspace_client)
    items = await browser.list_workspace(request.path)
    return BrowseResponse(
        path=request.path,
        items=[item.dict() for item in items],
    )


@router.post("/browse-volume", response_model=BrowseResponse)
async def browse_volume(
    request: BrowseRequest,
    user: UserContext = Depends(get_user_context),
):
    """Browse files in a UC Volume directory."""
    browser = SourceBrowser(user.workspace_client)
    items = await browser.list_volume(request.path)
    return BrowseResponse(
        path=request.path,
        items=[item.dict() for item in items],
    )


@router.post("/browse-repo", response_model=BrowseResponse)
async def browse_repo(
    request: BrowseRequest,
    user: UserContext = Depends(get_user_context),
):
    """Browse files in a Databricks Repo path.

    Path should be /Repos/<user>/<repo-name> or a sub-path.
    """
    browser = SourceBrowser(user.workspace_client)
    items = await browser.list_repo(request.path)
    return BrowseResponse(
        path=request.path,
        items=[item.dict() for item in items],
    )


@router.post("/detect-format", response_model=DetectFormatResponse)
async def detect_file_format(
    request: DetectFormatRequest,
    user: UserContext = Depends(get_user_context),
):
    """Auto-detect the source format of a file."""
    content_preview = None
    confidence = "high"

    # If a path is provided, read a preview for deeper detection
    if request.path:
        try:
            browser = SourceBrowser(user.workspace_client)
            content_preview = await browser.read_file_preview(request.path, max_bytes=2000)
            if content_preview:
                confidence = "high"
        except Exception:
            pass

    detected = detect_format(request.filename, content_preview)

    # Lower confidence for content-based detection vs extension-based
    ext = request.filename.rsplit('.', 1)[-1].lower() if '.' in request.filename else ''
    if ext in ('sql', 'txt', 'xml') and content_preview:
        confidence = "medium"
    elif ext == 'dtsx':
        confidence = "high"

    return DetectFormatResponse(
        filename=request.filename,
        detected_format=detected.value,
        confidence=confidence,
    )


@router.post("/ingest-remote", response_model=UploadResponse)
async def ingest_remote_file(
    request: IngestRemoteRequest,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Copy a file from Workspace/Volume into the job's upload path in UC Volumes.

    This ensures the converter always reads from a consistent UC Volume location.
    """
    job_id = request.job_id or str(uuid.uuid4())
    file_name = request.source_path.rsplit("/", 1)[-1]

    browser = SourceBrowser(user.workspace_client)

    # Read the full file content
    if request.source_type == "volume":
        # For volume files, use the volume manager directly
        volume_manager = VolumeManager(user.workspace_client, settings)
        content = await volume_manager.read_file(request.source_path)
    else:
        # For workspace files, read via workspace export API
        content_str = await browser.read_file_preview(request.source_path, max_bytes=10_000_000)
        content = content_str.encode("utf-8")

    # Write to the job's upload path
    volume_manager = VolumeManager(user.workspace_client, settings)
    upload_path = f"{settings.uploads_path}/{job_id}/{file_name}"
    await volume_manager.write_file(upload_path, content)

    logger.info(f"Ingested {request.source_path} to {upload_path}")

    return UploadResponse(
        job_id=job_id,
        file_path=upload_path,
        file_name=file_name,
        file_size=len(content),
    )


# ===== Reference Files =====

class ReferenceFileResponse(BaseModel):
    """Response after reference file upload."""
    file_id: str
    file_name: str
    file_path: str
    file_type: str


@router.post("/reference/upload", response_model=ReferenceFileResponse)
async def upload_reference_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(default=None),
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Upload a reference file (md, sql, txt) for conversion context."""
    # Validate file type
    allowed_extensions = {".md", ".sql", ".txt", ".py", ".json", ".yaml", ".yml"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    file_id = str(uuid.uuid4())
    volume_manager = VolumeManager(user.workspace_client, settings)
    file_content = await file.read()

    # Store in reference files directory
    upload_path = f"{settings.volume_path}/reference/{file_id}/{file.filename}"
    await volume_manager.write_file(upload_path, file_content)

    # Create reference file record
    ref_file = ReferenceFile(
        file_id=file_id,
        file_name=file.filename,
        file_path=upload_path,
        file_type=file_ext.lstrip("."),
        description=description,
    )
    _reference_files_store[file_id] = ref_file

    logger.info(f"Uploaded reference file {file.filename} with ID {file_id}")

    return ReferenceFileResponse(
        file_id=file_id,
        file_name=file.filename,
        file_path=upload_path,
        file_type=file_ext.lstrip("."),
    )


@router.get("/reference", response_model=List[ReferenceFile])
async def list_reference_files(
    user: UserContext = Depends(get_user_context),
):
    """List all uploaded reference files."""
    return list(_reference_files_store.values())


@router.get("/reference/{file_id}")
async def get_reference_file(
    file_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get content of a reference file."""
    if file_id not in _reference_files_store:
        raise HTTPException(status_code=404, detail=f"Reference file {file_id} not found")

    ref_file = _reference_files_store[file_id]
    volume_manager = VolumeManager(user.workspace_client, settings)

    try:
        content = await volume_manager.read_file(ref_file.file_path)
        return {
            "file_id": file_id,
            "file_name": ref_file.file_name,
            "file_type": ref_file.file_type,
            "content": content.decode("utf-8") if isinstance(content, bytes) else content,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@router.delete("/reference/{file_id}")
async def delete_reference_file(
    file_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Delete a reference file."""
    if file_id not in _reference_files_store:
        raise HTTPException(status_code=404, detail=f"Reference file {file_id} not found")

    ref_file = _reference_files_store[file_id]
    volume_manager = VolumeManager(user.workspace_client, settings)

    try:
        await volume_manager.delete_file(ref_file.file_path)
    except Exception:
        pass  # File might already be deleted

    del _reference_files_store[file_id]
    return {"status": "deleted", "file_id": file_id}


# ===== Code Compare =====
# NOTE: This must be defined BEFORE the catch-all /{job_id}/{path:path} route

class CodeCompareResponse(BaseModel):
    """Response for code comparison."""
    source_file: str
    source_content: str
    output_files: List[dict]  # [{name, content, type}]


@router.get("/{job_id}/compare")
async def get_code_compare(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get source and converted code for side-by-side comparison."""
    volume_manager = VolumeManager(user.workspace_client, settings)

    # Look up the job to get the actual source_path and output_path
    from .conversions import _get_job
    job = await _get_job(job_id, user, settings)

    # Get source file using the job's actual source_path
    source_content = ""
    source_file = ""
    debug_info = {}  # Collect debug info for troubleshooting
    try:
        if job and job.source_path:
            # Read source directly from the job's source_path
            source_file = Path(job.source_path).name
            debug_info["source_path"] = job.source_path
            debug_info["vm_auth_type"] = volume_manager.client.config.auth_type
            content = await volume_manager.read_file(job.source_path)
            source_content = content.decode("utf-8") if isinstance(content, bytes) else content
        else:
            # Fallback: try listing the uploads directory
            upload_files = await volume_manager.list_files(f"{settings.uploads_path}/{job_id}")
            if upload_files:
                source_file = upload_files[0].name
                source_path = f"{settings.uploads_path}/{job_id}/{source_file}"
                content = await volume_manager.read_file(source_path)
                source_content = content.decode("utf-8") if isinstance(content, bytes) else content
    except Exception as e:
        logger.warning(f"Failed to read source file for {job_id}: {e}", exc_info=True)
        debug_info["source_error"] = f"{type(e).__name__}: {e}"

    # Determine output base path
    output_base = (job.output_path if job and job.output_path
                   else f"{settings.outputs_path}/{job_id}")
    debug_info["output_base"] = output_base
    debug_info["job_found"] = job is not None
    debug_info["job_output_path"] = job.output_path if job else None
    debug_info["job_status"] = job.status.value if job and hasattr(job.status, 'value') else str(job.status) if job else None
    debug_info["settings_outputs_path"] = settings.outputs_path

    # Get output files
    output_files = []

    # Check for notebooks directory
    notebooks_path = f"{output_base}/notebooks"
    debug_info["notebooks_path"] = notebooks_path

    # First, list the output_base to see what's there
    try:
        base_contents = await volume_manager.list_files(output_base)
        debug_info["output_base_contents"] = [
            {"name": f.name, "path": f.path, "is_dir": f.is_directory}
            for f in base_contents
        ]
    except Exception as e:
        debug_info["output_base_list_error"] = f"{type(e).__name__}: {e}"

    # List and read notebooks
    try:
        notebook_files = await volume_manager.list_files(notebooks_path)
        debug_info["notebook_files_found"] = len(notebook_files)
        debug_info["notebook_files_detail"] = [
            {"name": f.name, "path": f.path, "is_dir": f.is_directory, "size": f.size}
            for f in notebook_files
        ]
        for nb in notebook_files:
            if nb.is_directory:
                continue
            # Use nb.path directly if available, otherwise construct it
            read_path = nb.path if nb.path else f"{notebooks_path}/{nb.name}"
            try:
                content = await volume_manager.read_file(read_path)
                output_files.append({
                    "name": nb.name,
                    "content": content.decode("utf-8") if isinstance(content, bytes) else content,
                    "type": "notebook",
                })
            except Exception as e:
                logger.warning(f"Failed to read notebook {nb.name} at {read_path}: {e}")
                debug_info[f"notebook_read_error_{nb.name}"] = f"{type(e).__name__}: {e} (path: {read_path})"
    except Exception as e:
        logger.warning(f"Failed to list notebooks for {job_id}: {e}", exc_info=True)
        debug_info["notebooks_error"] = f"{type(e).__name__}: {e}"

    # Read additional output files (workflow, quality report, gate report)
    for file_name, file_type in [
        ("workflow.json", "workflow"),
        ("quality_report.json", "quality"),
        ("gate_report.json", "gate"),
    ]:
        try:
            content = await volume_manager.read_file(f"{output_base}/{file_name}")
            output_files.append({
                "name": file_name,
                "content": content.decode("utf-8") if isinstance(content, bytes) else content,
                "type": file_type,
            })
        except Exception:
            pass

    return {
        "job_id": job_id,
        "source_file": source_file,
        "source_content": source_content,
        "output_files": output_files,
        "_debug": debug_info,
    }


@router.post("/{job_id}/browse")
async def browse_directory(
    job_id: str,
    path: str = "",
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Browse a directory within a job's output."""
    volume_manager = VolumeManager(user.workspace_client, settings)
    dir_path = f"{settings.outputs_path}/{job_id}/{path}".rstrip("/")

    try:
        files = await volume_manager.list_files(dir_path)
        return DirectoryListing(path=dir_path, files=files)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Directory not found: {e}")


# ===== Generic file access =====
# NOTE: Catch-all route MUST be last to avoid matching specific routes above

@router.get("/{job_id}", response_model=DirectoryListing)
async def list_files(
    job_id: str,
    path_type: str = "uploads",  # uploads, outputs, validation
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """List files for a job."""
    base_paths = {
        "uploads": settings.uploads_path,
        "outputs": settings.outputs_path,
        "validation": settings.validation_path,
    }

    if path_type not in base_paths:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid path_type. Allowed: {', '.join(base_paths.keys())}"
        )

    volume_manager = VolumeManager(user.workspace_client, settings)
    dir_path = f"{base_paths[path_type]}/{job_id}"

    try:
        files = await volume_manager.list_files(dir_path)
        return DirectoryListing(path=dir_path, files=files)
    except Exception as e:
        logger.warning(f"Failed to list files at {dir_path}: {e}")
        return DirectoryListing(path=dir_path, files=[])


@router.get("/{job_id}/{path:path}")
async def get_file_content(
    job_id: str,
    path: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get content of a specific file."""
    volume_manager = VolumeManager(user.workspace_client, settings)

    # Try to find file in uploads, outputs, or validation
    for base in [settings.uploads_path, settings.outputs_path, settings.validation_path]:
        full_path = f"{base}/{job_id}/{path}"
        try:
            content = await volume_manager.read_file(full_path)
            return {
                "path": full_path,
                "content": content.decode("utf-8") if isinstance(content, bytes) else content,
            }
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"File not found: {path}")
