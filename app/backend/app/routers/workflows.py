"""
API Router for workflow deployment and promotion.
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_user_context, UserContext
from ..config import get_settings, Settings
from ..models.conversion import (
    PromotionRequest,
    PromotionHistory,
    ApprovalStatus,
    Environment,
)
from ..services.job_runner import JobRunner

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for demo
_promotion_history: dict[str, List[PromotionHistory]] = {}


@router.post("/{job_id}/deploy")
async def deploy_workflow(
    job_id: str,
    environment: Environment = Environment.DEV,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Deploy converted workflow to Databricks Jobs.

    Default deployment is to DEV environment.
    """
    # Get job info
    from .conversions import _jobs_store
    job = _jobs_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.output_path:
        raise HTTPException(
            status_code=400,
            detail="Job has no output. Run conversion first."
        )

    # Deploy via Jobs API
    job_runner = JobRunner(user.workspace_client, settings)
    deployed_job_id = await job_runner.deploy_workflow(
        job_id=job_id,
        workflow_path=f"{job.output_path}/workflow.json",
        environment=environment.value,
    )

    logger.info(f"Deployed job {job_id} to {environment}: Databricks Job {deployed_job_id}")

    return {
        "status": "deployed",
        "job_id": job_id,
        "environment": environment,
        "databricks_job_id": deployed_job_id,
    }


@router.get("/{job_id}/status")
async def get_deployment_status(
    job_id: str,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get deployment status for a job."""
    from .conversions import _jobs_store
    job = _jobs_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get latest promotion
    history = _promotion_history.get(job_id, [])
    latest = history[-1] if history else None

    return {
        "job_id": job_id,
        "job_status": job.status,
        "current_environment": latest.to_environment if latest else "dev",
        "latest_promotion": latest.dict() if latest else None,
    }


@router.post("/promote")
async def promote_workflow(
    request: PromotionRequest,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Initiate promotion to next environment.

    Workflow:
    - dev → qa: Requires QA approval
    - qa → prod: Requires designated approver(s)
    """
    from .conversions import _jobs_store
    from .validation import _quality_scores

    job = _jobs_store.get(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found")

    # Check quality score meets threshold
    score = _quality_scores.get(request.job_id)
    if score and not score.is_promotable:
        raise HTTPException(
            status_code=400,
            detail=f"Quality score {score.overall_score:.1%} below 70% threshold. Cannot promote."
        )

    # Determine current environment from history
    history = _promotion_history.get(request.job_id, [])
    current_env = history[-1].to_environment if history else Environment.DEV

    # Validate promotion path
    valid_promotions = {
        Environment.DEV: Environment.QA,
        Environment.QA: Environment.PROD,
    }
    if current_env not in valid_promotions:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot promote from {current_env}. Already at highest environment."
        )

    expected_next = valid_promotions[current_env]
    if request.to_environment != expected_next:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid promotion path. From {current_env}, can only promote to {expected_next}."
        )

    # Create promotion record
    promotion = PromotionHistory(
        job_id=request.job_id,
        from_environment=current_env,
        to_environment=request.to_environment,
        promoted_by=user.user_id,
        notes=request.notes,
        approval_status=ApprovalStatus.PENDING,
    )

    # Store promotion
    if request.job_id not in _promotion_history:
        _promotion_history[request.job_id] = []
    _promotion_history[request.job_id].append(promotion)

    logger.info(f"Promotion requested: {request.job_id} from {current_env} to {request.to_environment}")

    return {
        "status": "pending_approval",
        "promotion_id": promotion.promotion_id,
        "from_environment": promotion.from_environment,
        "to_environment": promotion.to_environment,
        "message": f"Promotion to {request.to_environment} requires approval.",
    }


@router.get("/{job_id}/history", response_model=List[PromotionHistory])
async def get_promotion_history(
    job_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Get promotion history for a job."""
    return _promotion_history.get(job_id, [])


@router.post("/promote/{promotion_id}/approve")
async def approve_promotion(
    promotion_id: str,
    approved: bool = True,
    rejection_reason: Optional[str] = None,
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Approve or reject a promotion request."""
    # Find promotion
    promotion = None
    job_id = None
    for jid, history in _promotion_history.items():
        for p in history:
            if p.promotion_id == promotion_id:
                promotion = p
                job_id = jid
                break
        if promotion:
            break

    if not promotion:
        raise HTTPException(status_code=404, detail=f"Promotion {promotion_id} not found")

    if promotion.approval_status != ApprovalStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Promotion already {promotion.approval_status}"
        )

    # Update approval status
    promotion.approved_at = datetime.utcnow()
    promotion.approver = user.user_id

    if approved:
        promotion.approval_status = ApprovalStatus.APPROVED

        # Deploy to target environment
        job_runner = JobRunner(user.workspace_client, settings)
        from .conversions import _jobs_store
        job = _jobs_store.get(job_id)

        if job and job.output_path:
            deployed_job_id = await job_runner.deploy_workflow(
                job_id=job_id,
                workflow_path=f"{job.output_path}/workflow.json",
                environment=promotion.to_environment.value,
            )
            promotion.deployed_job_id = deployed_job_id

        logger.info(f"Promotion {promotion_id} approved by {user.user_id}")

        return {
            "status": "approved",
            "promotion_id": promotion_id,
            "deployed_job_id": promotion.deployed_job_id,
            "environment": promotion.to_environment,
        }
    else:
        promotion.approval_status = ApprovalStatus.REJECTED
        promotion.rejection_reason = rejection_reason

        logger.info(f"Promotion {promotion_id} rejected by {user.user_id}: {rejection_reason}")

        return {
            "status": "rejected",
            "promotion_id": promotion_id,
            "reason": rejection_reason,
        }
