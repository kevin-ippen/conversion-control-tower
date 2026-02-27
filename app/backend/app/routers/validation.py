"""
API Router for validation and quality scoring.
"""

import logging
from typing import List, Optional, Any
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..auth import get_user_context, UserContext
from ..config import get_settings, Settings
from ..models.validation import (
    ValidationResult,
    QualityScore,
    ValidationReport,
    ValidationRunRequest,
)
from ..services.validator import ValidationService

logger = logging.getLogger(__name__)
router = APIRouter()


class ColumnSchema(BaseModel):
    name: str
    type: str


class DataSample(BaseModel):
    columns: List[str]
    rows: List[List[Any]]


class DataSource(BaseModel):
    location: str
    catalog: Optional[str] = None
    schema_name: Optional[str] = None  # Note: 'schema' is reserved in pydantic
    table: Optional[str] = None
    row_count: int
    column_count: int
    columns: List[ColumnSchema]
    sample: DataSample


class DataComparison(BaseModel):
    row_count_match: bool
    schema_match: bool
    sample_match_rate: float
    mismatched_columns: List[str]
    summary: str


class DataCompareResponse(BaseModel):
    job_id: str
    original: Optional[DataSource] = None
    converted: Optional[DataSource] = None
    comparison: Optional[DataComparison] = None

# In-memory store for demo
_validation_results: dict[str, List[ValidationResult]] = {}
_quality_scores: dict[str, QualityScore] = {}
_pipeline_runs: dict[str, dict] = {}  # job_id -> pipeline run status


@router.post("/{job_id}/run-pipeline")
async def run_validation_pipeline(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Trigger the full 3-step validation pipeline.

    Steps:
    1. Original Simulator - generates expected output from SSIS logic
    2. Converted Runner - executes converted Databricks code
    3. Data Comparator - compares expected vs actual output

    This runs asynchronously and returns immediately with pipeline status.
    """
    from .conversions import _get_job
    from ..services.job_runner import JobRunner, JobRunnerError

    job = await _get_job(job_id, user, settings)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job must be completed before running validation pipeline. Current status: {job.status}"
        )

    # Check if pipeline is already running
    if job_id in _pipeline_runs and _pipeline_runs[job_id].get("status") == "running":
        return _pipeline_runs[job_id]

    runner = JobRunner(user.workspace_client, settings)

    # Mark as running
    _pipeline_runs[job_id] = {
        "conversion_job_id": job_id,
        "status": "running",
        "steps": {},
        "message": "Validation pipeline starting...",
    }

    # Run the pipeline in the background
    import asyncio

    async def _run_pipeline():
        try:
            result = await runner.submit_validation_pipeline(
                conversion_job_id=job_id,
                source_path=job.source_path,
                output_path=job.output_path or f"{settings.outputs_path}/{job_id}",
                validation_table=getattr(job, "validation_table", None),
            )
            _pipeline_runs[job_id] = result
        except JobRunnerError as e:
            _pipeline_runs[job_id] = {
                "conversion_job_id": job_id,
                "status": "failed",
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Pipeline failed for {job_id}: {e}")
            _pipeline_runs[job_id] = {
                "conversion_job_id": job_id,
                "status": "failed",
                "error": str(e),
            }

    asyncio.create_task(_run_pipeline())

    return {
        "status": "started",
        "job_id": job_id,
        "message": "Validation pipeline started. Poll /api/validation/{job_id}/pipeline-status for progress.",
    }


@router.get("/{job_id}/pipeline-status")
async def get_pipeline_status(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Get the current status of the validation pipeline."""
    if job_id not in _pipeline_runs:
        return {
            "conversion_job_id": job_id,
            "status": "not_started",
            "message": "Validation pipeline has not been run for this job.",
        }

    return _pipeline_runs[job_id]


@router.post("/{job_id}/run")
async def run_validation(
    job_id: str,
    request: ValidationRunRequest = None,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Run validation on a conversion job's output."""
    validator = ValidationService(user.workspace_client, settings)

    try:
        report = await validator.run_validation(
            job_id=job_id,
            compare_with_source=request.compare_with_source if request else True,
            source_table=request.source_table if request else None,
        )

        # Store results
        _validation_results[job_id] = report.results
        _quality_scores[job_id] = report.score

        logger.info(f"Validation complete for {job_id}: score={report.score.overall_score:.2f}")

        return {
            "status": "completed",
            "job_id": job_id,
            "overall_score": report.score.overall_score,
            "grade": report.score.grade,
            "passed_count": report.score.passed_count,
            "failed_count": report.score.failed_count,
            "is_promotable": report.score.is_promotable,
            "dimensions": [
                {
                    "name": d.name,
                    "score": d.score,
                    "weight": d.weight,
                    "description": d.description,
                    "passed_count": d.passed_count,
                    "failed_count": d.failed_count,
                    "grade": d.grade,
                }
                for d in report.score.dimensions
            ],
        }

    except Exception as e:
        logger.error(f"Validation failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


@router.get("/{job_id}/results", response_model=List[ValidationResult])
async def get_validation_results(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Get validation results for a job."""
    if job_id not in _validation_results:
        raise HTTPException(
            status_code=404,
            detail=f"No validation results for job {job_id}. Run validation first."
        )

    return _validation_results[job_id]


@router.get("/{job_id}/score", response_model=QualityScore)
async def get_quality_score(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Get quality score for a job."""
    if job_id not in _quality_scores:
        raise HTTPException(
            status_code=404,
            detail=f"No quality score for job {job_id}. Run validation first."
        )

    return _quality_scores[job_id]


@router.get("/{job_id}/report", response_model=ValidationReport)
async def get_validation_report(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Get full validation report for a job."""
    if job_id not in _validation_results or job_id not in _quality_scores:
        raise HTTPException(
            status_code=404,
            detail=f"No validation report for job {job_id}. Run validation first."
        )

    # Get job info (would come from database)
    from .conversions import _jobs_store
    job = _jobs_store.get(job_id)
    job_name = job.job_name if job else "Unknown"

    return ValidationReport(
        job_id=job_id,
        job_name=job_name,
        score=_quality_scores[job_id],
        results=_validation_results[job_id],
    )


@router.get("/{job_id}/report/download")
async def download_validation_report(
    job_id: str,
    format: str = "json",  # json, html, pdf
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Download validation report file."""
    report_path = f"{settings.validation_path}/{job_id}/validation_report.json"

    # In production, read from UC Volume
    # For demo, return a placeholder
    raise HTTPException(
        status_code=501,
        detail="Report download not yet implemented. Use /report endpoint for JSON."
    )


@router.get("/{job_id}/data-compare", response_model=DataCompareResponse)
async def get_data_compare(
    job_id: str,
    sample_size: int = Query(default=10, le=100, description="Number of sample rows"),
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get side-by-side data comparison between EXPECTED output and ACTUAL output.

    Reads real comparison results from the validation pipeline if available.
    Returns a "not yet run" status if the pipeline hasn't been executed.
    """
    from .conversions import _get_job
    from ..services.volume_manager import VolumeManager
    import json

    job = await _get_job(job_id, user, settings)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    validation_base = f"{settings.validation_path}/{job_id}"
    volume_manager = VolumeManager(user.workspace_client, settings)

    # Try to read real comparison results from the validation pipeline
    try:
        comparison_bytes = await volume_manager.read_file(
            f"{validation_base}/comparison_report.json"
        )
        comparison_data = json.loads(comparison_bytes.decode("utf-8"))

        # Build response from real pipeline data
        original = None
        converted = None
        comparison = None

        if "expected" in comparison_data:
            exp = comparison_data["expected"]
            original = DataSource(
                location=f"{validation_base}/expected_output",
                catalog=settings.catalog,
                schema_name=settings.schema_name,
                table=exp.get("table", "expected_output"),
                row_count=exp.get("row_count", 0),
                column_count=len(exp.get("columns", [])),
                columns=[ColumnSchema(name=c["name"], type=c["type"]) for c in exp.get("columns", [])],
                sample=DataSample(
                    columns=[c["name"] for c in exp.get("columns", [])],
                    rows=exp.get("sample_rows", [])[:sample_size],
                ),
            )

        if "actual" in comparison_data:
            act = comparison_data["actual"]
            converted = DataSource(
                location=f"{validation_base}/actual_output",
                catalog=settings.catalog,
                schema_name=settings.schema_name,
                table=act.get("table", "actual_output"),
                row_count=act.get("row_count", 0),
                column_count=len(act.get("columns", [])),
                columns=[ColumnSchema(name=c["name"], type=c["type"]) for c in act.get("columns", [])],
                sample=DataSample(
                    columns=[c["name"] for c in act.get("columns", [])],
                    rows=act.get("sample_rows", [])[:sample_size],
                ),
            )

        if "comparison" in comparison_data:
            comp = comparison_data["comparison"]
            comparison = DataComparison(
                row_count_match=comp.get("row_count_match", False),
                schema_match=comp.get("schema_match", False),
                sample_match_rate=comp.get("sample_match_rate", 0.0),
                mismatched_columns=comp.get("mismatched_columns", [])[:10],
                summary=comp.get("summary", ""),
            )

        return DataCompareResponse(
            job_id=job_id,
            original=original,
            converted=converted,
            comparison=comparison,
        )

    except Exception as e:
        logger.info(f"No comparison data for job {job_id}: {e}")
        # Return empty response indicating pipeline hasn't been run
        return DataCompareResponse(
            job_id=job_id,
            original=None,
            converted=None,
            comparison=DataComparison(
                row_count_match=False,
                schema_match=False,
                sample_match_rate=0.0,
                mismatched_columns=[],
                summary="Data comparison pipeline has not been run yet. Click 'Run Full Validation Pipeline' to generate real comparison data.",
            ),
        )
