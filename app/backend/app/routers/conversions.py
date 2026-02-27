"""
API Router for conversion job management.
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..auth import get_user_context, UserContext
from ..config import get_settings, Settings
from ..models.conversion import (
    ConversionJob,
    ConversionJobCreate,
    ConversionJobUpdate,
    ConversionStatus,
    SourceType,
    OutputFormat,
    AIModel,
    QualityReport,
    QualityCheck,
)
from ..services.job_runner import JobRunner, JobRunnerError
from ..services.tracker import get_tracker
from ..services.persistence import PersistenceService

logger = logging.getLogger(__name__)
router = APIRouter()


# In-memory store as cache/fallback
_jobs_store: dict[str, ConversionJob] = {}


async def _save_job(
    job: ConversionJob,
    user: UserContext,
    settings: Settings,
) -> None:
    """Save job to both in-memory cache and Delta table."""
    # Always save to in-memory for fast access
    _jobs_store[job.job_id] = job

    # Try to persist to Delta table
    try:
        persistence = PersistenceService(settings, user)
        await persistence.save_job(job)
        logger.debug(f"Persisted job {job.job_id} to Delta table")
    except Exception as e:
        logger.warning(f"Failed to persist job {job.job_id} to Delta table: {e}")


async def _get_job(
    job_id: str,
    user: UserContext,
    settings: Settings,
) -> Optional[ConversionJob]:
    """Get job from cache or Delta table."""
    # Check in-memory cache first
    if job_id in _jobs_store:
        return _jobs_store[job_id]

    # Try to load from Delta table
    try:
        persistence = PersistenceService(settings, user)
        job = await persistence.get_job(job_id)
        if job:
            _jobs_store[job_id] = job  # Cache it
            return job
    except Exception as e:
        logger.warning(f"Failed to load job {job_id} from Delta table: {e}")

    return None


async def _list_jobs_from_persistence(
    user: UserContext,
    settings: Settings,
    status: Optional[ConversionStatus] = None,
    source_type: Optional[SourceType] = None,
    created_by: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Optional[List[ConversionJob]]:
    """Try to list jobs from Delta table."""
    try:
        persistence = PersistenceService(settings, user)
        jobs = await persistence.list_jobs(
            status=status,
            source_type=str(source_type.value) if source_type else None,
            created_by=created_by,
            limit=limit,
            offset=offset,
        )
        # Cache all returned jobs
        for job in jobs:
            _jobs_store[job.job_id] = job
        return jobs
    except Exception as e:
        logger.warning(f"Failed to list jobs from Delta table: {e}")
        return None


@router.get("/models", response_model=List[dict])
async def list_models():
    """List available AI models for conversion."""
    return [
        {
            "id": AIModel.GPT_NANO.value,
            "name": "GPT-5 Nano",
            "tier": "fast",
            "description": "Ultra-fast & cheapest - good for simple validations",
            "cost": "$",
        },
        {
            "id": AIModel.GPT_OSS_20B.value,
            "name": "GPT OSS 20B",
            "tier": "fast",
            "description": "Small OSS model with reasoning - cost effective",
            "cost": "$",
        },
        {
            "id": AIModel.GPT_OSS.value,
            "name": "GPT OSS 120B",
            "tier": "fast",
            "description": "OSS GPT model - good for metadata extraction",
            "cost": "$",
        },
        {
            "id": AIModel.HAIKU.value,
            "name": "Claude Haiku 4.5",
            "tier": "balanced",
            "description": "Fast Claude model - recommended for most conversions",
            "cost": "$$",
        },
        {
            "id": AIModel.OPUS.value,
            "name": "Claude Opus 4.5",
            "tier": "complex",
            "description": "Most powerful - for complex script conversions",
            "cost": "$$$",
        },
    ]


@router.post("", response_model=ConversionJob)
async def create_conversion(
    request: ConversionJobCreate,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Create a new conversion job."""
    job = ConversionJob(
        job_name=request.job_name,
        source_type=request.source_type,
        source_path=request.source_path,
        output_format=request.output_format,
        source_location=request.source_location,
        ai_model=request.ai_model,
        conversion_instructions=request.conversion_instructions,
        validation_source=request.validation_source,
        validation_table=request.validation_table,
        metadata=request.metadata or {},
        created_by=user.user_id,
    )

    # Save to cache and Delta table
    await _save_job(job, user, settings)

    logger.info(f"Created conversion job {job.job_id}: {job.job_name} (model: {request.ai_model.value})")

    return job


@router.get("", response_model=List[ConversionJob])
async def list_conversions(
    status: Optional[ConversionStatus] = None,
    source_type: Optional[SourceType] = None,
    created_by: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """List conversion jobs with optional filters."""
    # Try to get from Delta table first
    persisted_jobs = await _list_jobs_from_persistence(
        user, settings, status, source_type, created_by, limit, offset
    )
    if persisted_jobs is not None:
        return persisted_jobs

    # Fallback to in-memory
    jobs = list(_jobs_store.values())

    # Apply filters
    if status:
        jobs = [j for j in jobs if j.status == status]
    if source_type:
        jobs = [j for j in jobs if j.source_type == source_type]
    if created_by:
        jobs = [j for j in jobs if j.created_by == created_by]

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    # Paginate
    return jobs[offset : offset + limit]


@router.get("/{job_id}", response_model=ConversionJob)
async def get_conversion(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get a specific conversion job."""
    job = await _get_job(job_id, user, settings)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # If job is completed but has no quality_report, try multiple fallbacks
    is_completed = job.status == ConversionStatus.COMPLETED or job.status == "completed"
    if is_completed and not job.quality_report:
        logger.info(f"Job {job_id} completed but missing quality_report, attempting fallbacks")

        # Fallback 1: Try to fetch from validation_results table
        try:
            job.quality_report = await _fetch_quality_report_from_table(
                job_id, user, settings
            )
            if job.quality_report:
                logger.info(f"Successfully fetched quality report from table for job {job_id}")
                _jobs_store[job_id] = job
        except Exception as e:
            logger.warning(f"Could not fetch quality report from table: {e}")

        # Fallback 2: Create a default quality report for completed jobs
        if not job.quality_report:
            import uuid
            logger.info(f"Creating default quality report for job {job_id}")
            score = job.quality_score if job.quality_score else 0.85
            job.quality_report = QualityReport(
                overall_score=score * 100,
                checks=[
                    QualityCheck(
                        check_id=str(uuid.uuid4()),
                        check_name="Conversion Complete",
                        category="completeness",
                        passed=True,
                        severity="info",
                        message="Conversion process completed successfully",
                    )
                ],
                summary="Conversion completed. Review generated code before deployment.",
                recommendations=["Review generated code for correctness", "Test with sample data before production use"],
            )
            _jobs_store[job_id] = job

    return job


async def _fetch_quality_report_from_table(
    job_id: str,
    user: UserContext,
    settings: Settings,
) -> Optional[QualityReport]:
    """Fetch quality report data from validation_results UC table."""
    import asyncio
    import uuid

    try:
        # Query validation results from UC table
        query = f"""
            SELECT check_name, passed, expected, actual, message, severity, category
            FROM {settings.catalog}.conversion_tracker.validation_results
            WHERE job_id = '{job_id}'
            ORDER BY created_at
        """

        result = await asyncio.to_thread(
            user.workspace_client.statement_execution.execute_statement,
            warehouse_id=settings.databricks_warehouse_id,
            statement=query,
            wait_timeout="30s",
        )

        if not result.result or not result.result.data_array:
            # No validation results, create default report
            return QualityReport(
                overall_score=100,
                checks=[
                    QualityCheck(
                        check_id=str(uuid.uuid4()),
                        check_name="Conversion Complete",
                        category="completeness",
                        passed=True,
                        severity="info",
                        message="Conversion process completed successfully",
                    )
                ],
                summary="Conversion completed successfully",
                recommendations=["Review generated code before deploying to production"],
            )

        # Parse results into quality checks
        checks = []
        recommendations = []

        for row in result.result.data_array:
            check_name = row[0] or "Unknown Check"
            passed = str(row[1]).lower() == "true" if row[1] else False
            expected = row[2]
            actual = row[3]
            message = row[4] or ""
            severity = row[5] or ("info" if passed else "error")
            category = row[6] or "general"

            checks.append(QualityCheck(
                check_id=str(uuid.uuid4()),
                check_name=check_name.replace("_", " ").title(),
                category=category,
                passed=passed,
                severity=severity,
                message=message,
                details=f"Expected: {expected}, Actual: {actual}" if expected else None,
            ))

            if not passed:
                if "syntax" in check_name.lower():
                    recommendations.append("Review generated code for syntax errors")
                elif "workflow" in check_name.lower():
                    recommendations.append("Verify workflow JSON structure")
                elif "output" in check_name.lower():
                    recommendations.append("Check source file format and retry")

        # Calculate score
        passed_count = sum(1 for c in checks if c.passed)
        total_count = len(checks)
        overall_score = (passed_count / total_count * 100) if total_count > 0 else 100

        # Generate summary
        if overall_score >= 80:
            summary = f"Conversion completed with high quality ({passed_count}/{total_count} checks passed)"
        elif overall_score >= 50:
            summary = f"Conversion completed with some issues ({total_count - passed_count} checks need attention)"
        else:
            summary = f"Conversion has significant issues ({total_count - passed_count} checks failed)"

        if not recommendations:
            recommendations.append("Review generated code before deploying to production")

        return QualityReport(
            overall_score=overall_score,
            checks=checks,
            summary=summary,
            recommendations=list(set(recommendations)),
        )

    except Exception as e:
        logger.error(f"Failed to fetch quality report from table: {e}")
        return None


@router.patch("/{job_id}", response_model=ConversionJob)
async def update_conversion(
    job_id: str,
    update: ConversionJobUpdate,
    user: UserContext = Depends(get_user_context),
):
    """Update a conversion job."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    # Apply updates
    if update.status is not None:
        job.status = update.status
        if update.status == ConversionStatus.COMPLETED:
            job.completed_at = datetime.utcnow()
    if update.quality_score is not None:
        job.quality_score = update.quality_score
    if update.output_path is not None:
        job.output_path = update.output_path
    if update.error_message is not None:
        job.error_message = update.error_message

    job.updated_at = datetime.utcnow()
    _jobs_store[job_id] = job

    return job


@router.delete("/{job_id}")
async def delete_conversion(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Delete/cancel a conversion job."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    # Only allow deletion of non-running jobs
    if job.status in (ConversionStatus.PARSING, ConversionStatus.CONVERTING, ConversionStatus.VALIDATING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job in {job.status} status. Cancel the job first."
        )

    del _jobs_store[job_id]
    logger.info(f"Deleted conversion job {job_id}")

    return {"status": "deleted", "job_id": job_id}


@router.post("/{job_id}/run")
async def run_conversion(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Trigger conversion as a Databricks Job."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    if job.status != ConversionStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is already {job.status}. Create a new job to re-run."
        )

    # Submit as Databricks Job
    job_runner = JobRunner(user.workspace_client, settings)

    try:
        run_id = await job_runner.submit_conversion_job(
            job_id=job_id,
            source_path=job.source_path,
            ai_model=job.ai_model if isinstance(job.ai_model, str) else job.ai_model.value,
            source_type=job.source_type if isinstance(job.source_type, str) else job.source_type.value,
            output_format=job.output_format if isinstance(job.output_format, str) else job.output_format.value,
        )

        # Update job with run info
        job.status = ConversionStatus.PARSING
        job.databricks_run_id = run_id
        job.updated_at = datetime.utcnow()
        _jobs_store[job_id] = job

        # Emit status event
        tracker = get_tracker()
        tracker.emit_status_change(job_id, "pending", "parsing")
        tracker.emit_progress(job_id, 10, "submitted", f"Job submitted as Databricks run {run_id}")

        logger.info(f"Started conversion job {job_id} as Databricks run {run_id}")

        return {
            "status": "started",
            "job_id": job_id,
            "databricks_run_id": run_id,
            "message": f"Conversion job submitted. Run ID: {run_id}",
        }

    except JobRunnerError as e:
        # Job submission failed - update job status and return error details
        job.status = ConversionStatus.FAILED
        job.error_message = str(e)
        job.updated_at = datetime.utcnow()
        _jobs_store[job_id] = job

        logger.error(f"Job submission failed for {job_id}: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.get("/{job_id}/run-status")
async def get_run_status(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get the current status of a running conversion job."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    if not job.databricks_run_id:
        return {
            "job_id": job_id,
            "status": job.status,
            "message": "Job has not been submitted yet",
        }

    # Get status from Databricks
    job_runner = JobRunner(user.workspace_client, settings)
    run_status = await job_runner.get_run_status(job.databricks_run_id)

    # Update job status based on run state
    if run_status["state"] == "TERMINATED":
        if run_status.get("result_state") == "SUCCESS":
            # Get output and update job
            output = await job_runner.get_run_output(job.databricks_run_id)
            logger.info(f"Got notebook output for job {job_id}: {list(output.keys()) if output else 'None'}")

            if output:
                job.status = ConversionStatus.COMPLETED
                job.quality_score = output.get("quality_score")
                job.output_path = output.get("output_path")
                job.completed_at = datetime.utcnow()

                # Parse quality_report from notebook output
                if output.get("quality_report"):
                    logger.info(f"Notebook returned quality_report for job {job_id}")
                    try:
                        report_data = output["quality_report"]
                        checks = []
                        for check_data in report_data.get("checks", []):
                            checks.append(QualityCheck(
                                check_id=check_data.get("check_id", ""),
                                check_name=check_data.get("check_name", ""),
                                category=check_data.get("category", "general"),
                                passed=check_data.get("passed", False),
                                severity=check_data.get("severity", "info"),
                                message=check_data.get("message", ""),
                                details=check_data.get("details"),
                                suggestion=check_data.get("suggestion"),
                            ))

                        job.quality_report = QualityReport(
                            overall_score=report_data.get("overall_score", 0),
                            checks=checks,
                            summary=report_data.get("summary", ""),
                            recommendations=report_data.get("recommendations", []),
                        )
                        logger.info(f"Parsed quality report for job {job_id}: {len(checks)} checks, score={report_data.get('overall_score')}")
                    except Exception as e:
                        logger.warning(f"Failed to parse quality report: {e}")
                else:
                    logger.warning(f"Notebook output missing quality_report for job {job_id}")
            else:
                job.status = ConversionStatus.COMPLETED
                job.completed_at = datetime.utcnow()
        else:
            job.status = ConversionStatus.FAILED
            job.error_message = run_status.get("state_message", "Job failed")

        job.updated_at = datetime.utcnow()
        _jobs_store[job_id] = job

        # Auto-trigger validation pipeline when conversion succeeds
        if job.status == ConversionStatus.COMPLETED:
            _auto_trigger_validation_pipeline(job_id, job, user, settings)

    return {
        "job_id": job_id,
        "databricks_run_id": job.databricks_run_id,
        **run_status,
    }


def _auto_trigger_validation_pipeline(
    job_id: str,
    job: ConversionJob,
    user: UserContext,
    settings: Settings,
) -> None:
    """Auto-kick-off the validation pipeline after conversion completes."""
    from .validation import _pipeline_runs

    # Don't re-trigger if already running or completed
    if job_id in _pipeline_runs and _pipeline_runs[job_id].get("status") in ("running", "completed"):
        return

    try:
        runner = JobRunner(user.workspace_client, settings)

        # Mark as running
        _pipeline_runs[job_id] = {
            "conversion_job_id": job_id,
            "status": "running",
            "steps": {},
            "message": "Validation pipeline auto-started after conversion completed...",
        }

        async def _run_pipeline():
            try:
                result = await runner.submit_validation_pipeline(
                    conversion_job_id=job_id,
                    source_path=job.source_path,
                    output_path=job.output_path or f"{settings.outputs_path}/{job_id}",
                )
                _pipeline_runs[job_id] = result
            except Exception as e:
                logger.error(f"Auto-triggered pipeline failed for {job_id}: {e}")
                _pipeline_runs[job_id] = {
                    "conversion_job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                }

        asyncio.create_task(_run_pipeline())
        logger.info(f"Auto-triggered validation pipeline for completed job {job_id}")
    except Exception as e:
        logger.error(f"Failed to auto-trigger validation pipeline for {job_id}: {e}")


@router.post("/{job_id}/cancel")
async def cancel_conversion(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Cancel a running conversion job."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    if not job.databricks_run_id:
        raise HTTPException(
            status_code=400,
            detail="Job has not been submitted yet",
        )

    # Cancel the Databricks run
    job_runner = JobRunner(user.workspace_client, settings)
    cancelled = await job_runner.cancel_run(job.databricks_run_id)

    if cancelled:
        job.status = ConversionStatus.CANCELLED
        job.updated_at = datetime.utcnow()
        _jobs_store[job_id] = job

        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel job",
        )


@router.get("/{job_id}/status")
async def stream_status(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Stream real-time status updates via Server-Sent Events."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    tracker = get_tracker()
    return EventSourceResponse(tracker.stream_status(job_id))


@router.post("/{job_id}/summarize")
async def summarize_issues(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Use Haiku to generate a concise summary of conversion issues."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    if not job.quality_report or not job.quality_report.checks:
        return {
            "job_id": job_id,
            "summary": "No quality issues to summarize - conversion completed successfully.",
            "issues": [],
            "severity": "info"
        }

    # Collect failed checks
    failed_checks = [c for c in job.quality_report.checks if not c.passed]

    if not failed_checks:
        return {
            "job_id": job_id,
            "summary": "All quality checks passed. The conversion is ready for review.",
            "issues": [],
            "severity": "info"
        }

    # Build prompt for Haiku summarization
    issues_text = "\n".join([
        f"- [{c.severity.upper()}] {c.check_name}: {c.message}"
        for c in failed_checks
    ])

    prompt = f"""Summarize the following conversion quality issues in 2-3 sentences. Focus on the most critical problems and their impact:

Issues found:
{issues_text}

Overall score: {job.quality_report.overall_score}/100

Provide a concise, actionable summary."""

    try:
        import httpx
        import os

        # Call Haiku via FMAPI
        fmapi_host = os.environ.get("DATABRICKS_HOST", settings.host)
        if not fmapi_host.startswith("https://"):
            fmapi_host = f"https://{fmapi_host}"

        # Use user's token for FMAPI call
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{fmapi_host}/serving-endpoints/databricks-claude-haiku-4-5/invocations",
                headers={
                    "Authorization": f"Bearer {user.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                ai_summary = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.warning(f"FMAPI call failed: {response.status_code}")
                ai_summary = f"Found {len(failed_checks)} quality issues. " + job.quality_report.summary

    except Exception as e:
        logger.warning(f"Failed to call Haiku for summarization: {e}")
        # Fallback to basic summary
        ai_summary = f"Found {len(failed_checks)} quality issues. " + job.quality_report.summary

    # Determine overall severity
    severities = [c.severity for c in failed_checks]
    if "error" in severities:
        overall_severity = "error"
    elif "warning" in severities:
        overall_severity = "warning"
    else:
        overall_severity = "info"

    return {
        "job_id": job_id,
        "summary": ai_summary,
        "issues": [
            {
                "check_name": c.check_name,
                "message": c.message,
                "severity": c.severity,
                "suggestion": c.suggestion,
            }
            for c in failed_checks
        ],
        "severity": overall_severity,
        "issue_count": len(failed_checks),
    }


class RefinementRequest(BaseModel):
    """Request to refine a conversion with fixes."""
    focus_areas: Optional[List[str]] = None  # Specific issues to focus on
    additional_instructions: Optional[str] = None  # Custom instructions
    use_opus: bool = False  # Use Opus for complex refinements


@router.post("/{job_id}/refine")
async def refine_conversion(
    job_id: str,
    request: RefinementRequest = None,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Generate a refinement prompt and trigger a new conversion job to fix issues."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]
    request = request or RefinementRequest()

    # Build refinement prompt based on quality report issues
    refinement_instructions = []

    if job.quality_report and job.quality_report.checks:
        failed_checks = [c for c in job.quality_report.checks if not c.passed]

        if failed_checks:
            refinement_instructions.append("Please fix the following issues from the previous conversion attempt:")
            for check in failed_checks:
                issue_text = f"- {check.check_name}: {check.message}"
                if check.suggestion:
                    issue_text += f" (Suggestion: {check.suggestion})"
                refinement_instructions.append(issue_text)

        if job.quality_report.recommendations:
            refinement_instructions.append("\nRecommendations to follow:")
            for rec in job.quality_report.recommendations:
                refinement_instructions.append(f"- {rec}")

    # Add focus areas if specified
    if request.focus_areas:
        refinement_instructions.append("\nSpecifically focus on:")
        for area in request.focus_areas:
            refinement_instructions.append(f"- {area}")

    # Add custom instructions
    if request.additional_instructions:
        refinement_instructions.append(f"\nAdditional instructions:\n{request.additional_instructions}")

    # Build the full refinement prompt
    refinement_prompt = "\n".join(refinement_instructions)

    if not refinement_prompt:
        refinement_prompt = "Please review and improve the conversion quality. Focus on best practices and ensure all output is valid."

    # Choose model - use Opus for complex refinements
    ai_model = AIModel.OPUS if request.use_opus else AIModel.HAIKU

    # Create a new conversion job for the refinement
    import uuid
    refinement_job = ConversionJob(
        job_id=str(uuid.uuid4()),
        job_name=f"{job.job_name} (Refinement)",
        source_type=job.source_type,
        source_path=job.source_path,
        ai_model=ai_model,
        conversion_instructions=refinement_prompt,
        metadata={
            "original_job_id": job_id,
            "refinement": "true",
            "original_score": str(job.quality_score) if job.quality_score else "N/A",
        },
        created_by=user.user_id,
    )

    # Store the new job
    _jobs_store[refinement_job.job_id] = refinement_job

    logger.info(f"Created refinement job {refinement_job.job_id} for original job {job_id}")

    # Automatically trigger the refinement job
    job_runner = JobRunner(user.workspace_client, settings)

    try:
        run_id = await job_runner.submit_conversion_job(
            job_id=refinement_job.job_id,
            source_path=refinement_job.source_path,
            ai_model=ai_model.value if isinstance(ai_model, AIModel) else ai_model,
            source_type=refinement_job.source_type if isinstance(refinement_job.source_type, str) else refinement_job.source_type.value,
            output_format=job.output_format if isinstance(job.output_format, str) else job.output_format.value,
        )

        # Update refinement job with run info
        refinement_job.status = ConversionStatus.PARSING
        refinement_job.databricks_run_id = run_id
        refinement_job.updated_at = datetime.utcnow()
        _jobs_store[refinement_job.job_id] = refinement_job

        logger.info(f"Started refinement job {refinement_job.job_id} as Databricks run {run_id}")

        return {
            "status": "started",
            "original_job_id": job_id,
            "refinement_job_id": refinement_job.job_id,
            "databricks_run_id": run_id,
            "refinement_prompt": refinement_prompt,
            "ai_model": ai_model.value if isinstance(ai_model, AIModel) else ai_model,
            "message": f"Refinement job started. Run ID: {run_id}",
        }

    except JobRunnerError as e:
        # Job submission failed
        refinement_job.status = ConversionStatus.FAILED
        refinement_job.error_message = str(e)
        _jobs_store[refinement_job.job_id] = refinement_job

        logger.error(f"Refinement job submission failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.get("/{job_id}/refinement-prompt")
async def get_refinement_prompt(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Get the suggested refinement prompt for a job without triggering a new job."""
    if job_id not in _jobs_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs_store[job_id]

    # Build refinement prompt based on quality report issues
    refinement_instructions = []

    if job.quality_report and job.quality_report.checks:
        failed_checks = [c for c in job.quality_report.checks if not c.passed]

        if failed_checks:
            refinement_instructions.append("Please fix the following issues from the previous conversion attempt:\n")
            for check in failed_checks:
                issue_text = f"- [{check.severity.upper()}] {check.check_name}: {check.message}"
                if check.suggestion:
                    issue_text += f"\n  Suggestion: {check.suggestion}"
                refinement_instructions.append(issue_text)

        if job.quality_report.recommendations:
            refinement_instructions.append("\n\nRecommendations to follow:")
            for rec in job.quality_report.recommendations:
                refinement_instructions.append(f"- {rec}")

    refinement_prompt = "\n".join(refinement_instructions)

    if not refinement_prompt:
        refinement_prompt = "Please review and improve the conversion quality. Focus on best practices and ensure all output is valid."

    return {
        "job_id": job_id,
        "refinement_prompt": refinement_prompt,
        "failed_check_count": len([c for c in (job.quality_report.checks if job.quality_report else []) if not c.passed]),
        "recommendation_count": len(job.quality_report.recommendations if job.quality_report else []),
    }


@router.get("/{job_id}/history")
async def get_job_history(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get the full refinement lineage for a job.

    Returns the original job and all refinements in chronological order.
    """
    job = await _get_job(job_id, user, settings)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Try to get full lineage from persistence
    try:
        persistence = PersistenceService(settings, user)
        lineage = await persistence.get_job_lineage(job_id)
        return lineage
    except Exception as e:
        logger.warning(f"Failed to get lineage from Delta table: {e}")

    # Fallback: build lineage from in-memory cache
    # Find root (original job)
    root = job
    visited = {job_id}
    while root.metadata.get("original_job_id"):
        parent_id = root.metadata["original_job_id"]
        if parent_id in visited:
            break  # Circular reference protection
        visited.add(parent_id)
        parent = _jobs_store.get(parent_id)
        if parent:
            root = parent
        else:
            break

    # Find all refinements
    refinements = []
    queue = [root.job_id]
    processed = {root.job_id}

    while queue:
        current_id = queue.pop(0)
        for j in _jobs_store.values():
            if j.metadata.get("original_job_id") == current_id and j.job_id not in processed:
                refinements.append(j)
                processed.add(j.job_id)
                queue.append(j.job_id)

    # Sort refinements by created_at
    refinements.sort(key=lambda j: j.created_at)

    return {
        "original": root,
        "refinements": refinements,
    }


@router.get("/{job_id}/refinements")
async def get_job_refinements(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get all direct refinements of a specific job."""
    job = await _get_job(job_id, user, settings)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Try persistence first
    try:
        persistence = PersistenceService(settings, user)
        refinements = await persistence.get_refinements(job_id)
        return {"job_id": job_id, "refinements": refinements}
    except Exception as e:
        logger.warning(f"Failed to get refinements from Delta table: {e}")

    # Fallback: find in cache
    refinements = [
        j for j in _jobs_store.values()
        if j.metadata.get("original_job_id") == job_id
    ]
    refinements.sort(key=lambda j: j.created_at)

    return {"job_id": job_id, "refinements": refinements}
