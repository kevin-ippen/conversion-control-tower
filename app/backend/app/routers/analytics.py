"""
API Router for analytics and aggregate metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ..auth import get_user_context, UserContext
from ..config import get_settings, Settings
from ..models.conversion import ConversionStatus, SourceType

logger = logging.getLogger(__name__)
router = APIRouter()


class OverviewMetrics(BaseModel):
    """Overview metrics for dashboard."""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    in_progress_jobs: int
    average_quality_score: float
    jobs_by_status: Dict[str, int]
    jobs_by_source_type: Dict[str, int]


class TrendDataPoint(BaseModel):
    """Single data point for trend charts."""
    date: str
    count: int
    avg_score: float


class QualityDistribution(BaseModel):
    """Distribution of quality scores."""
    grade_a: int  # 90-100%
    grade_b: int  # 80-89%
    grade_c: int  # 70-79%
    grade_d: int  # 60-69%
    grade_f: int  # 0-59%


class RecentActivity(BaseModel):
    """Recent activity item."""
    job_id: str
    job_name: str
    action: str
    user: str
    timestamp: datetime


@router.get("/overview", response_model=OverviewMetrics)
async def get_overview_metrics(
    user: UserContext = Depends(get_user_context),
    settings: Settings = Depends(get_settings),
):
    """Get aggregate metrics for the dashboard overview."""
    from .conversions import _jobs_store
    from .validation import _quality_scores

    jobs = list(_jobs_store.values())

    # Count by status
    status_counts = {}
    for status in ConversionStatus:
        status_counts[status.value] = sum(1 for j in jobs if j.status == status.value)

    # Count by source type
    source_counts = {}
    for st in SourceType:
        source_counts[st.value] = sum(1 for j in jobs if j.source_type == st.value)

    # Calculate average score
    scores = [s.overall_score for s in _quality_scores.values()]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return OverviewMetrics(
        total_jobs=len(jobs),
        completed_jobs=status_counts.get(ConversionStatus.COMPLETED.value, 0),
        failed_jobs=status_counts.get(ConversionStatus.FAILED.value, 0),
        in_progress_jobs=sum([
            status_counts.get(ConversionStatus.PARSING.value, 0),
            status_counts.get(ConversionStatus.CONVERTING.value, 0),
            status_counts.get(ConversionStatus.VALIDATING.value, 0),
        ]),
        average_quality_score=avg_score,
        jobs_by_status=status_counts,
        jobs_by_source_type=source_counts,
    )


@router.get("/trends", response_model=List[TrendDataPoint])
async def get_conversion_trends(
    days: int = Query(default=30, le=90),
    user: UserContext = Depends(get_user_context),
):
    """Get conversion trends over time."""
    from .conversions import _jobs_store
    from .validation import _quality_scores

    # Generate date range
    today = datetime.utcnow().date()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(days, -1, -1)]

    # Group jobs by date
    jobs = list(_jobs_store.values())
    trend_data = []

    for date_str in dates:
        day_jobs = [j for j in jobs if j.created_at.date().isoformat() == date_str]
        day_scores = [
            _quality_scores[j.job_id].overall_score
            for j in day_jobs
            if j.job_id in _quality_scores
        ]

        trend_data.append(TrendDataPoint(
            date=date_str,
            count=len(day_jobs),
            avg_score=sum(day_scores) / len(day_scores) if day_scores else 0.0,
        ))

    return trend_data


@router.get("/quality", response_model=QualityDistribution)
async def get_quality_distribution(
    user: UserContext = Depends(get_user_context),
):
    """Get distribution of quality scores by grade."""
    from .validation import _quality_scores

    distribution = QualityDistribution(
        grade_a=0,
        grade_b=0,
        grade_c=0,
        grade_d=0,
        grade_f=0,
    )

    for score in _quality_scores.values():
        pct = score.overall_score * 100
        if pct >= 90:
            distribution.grade_a += 1
        elif pct >= 80:
            distribution.grade_b += 1
        elif pct >= 70:
            distribution.grade_c += 1
        elif pct >= 60:
            distribution.grade_d += 1
        else:
            distribution.grade_f += 1

    return distribution


@router.get("/recent", response_model=List[RecentActivity])
async def get_recent_activity(
    limit: int = Query(default=10, le=50),
    user: UserContext = Depends(get_user_context),
):
    """Get recent activity feed."""
    from .conversions import _jobs_store
    from .workflows import _promotion_history

    activities = []

    # Add job creations
    for job in _jobs_store.values():
        activities.append(RecentActivity(
            job_id=job.job_id,
            job_name=job.job_name,
            action=f"created ({job.source_type})",
            user=job.created_by or "unknown",
            timestamp=job.created_at,
        ))

        if job.completed_at:
            activities.append(RecentActivity(
                job_id=job.job_id,
                job_name=job.job_name,
                action=f"completed (score: {job.quality_score:.0%})" if job.quality_score else "completed",
                user=job.created_by or "unknown",
                timestamp=job.completed_at,
            ))

    # Add promotions
    for job_id, history in _promotion_history.items():
        job = _jobs_store.get(job_id)
        job_name = job.job_name if job else "Unknown"
        for promotion in history:
            activities.append(RecentActivity(
                job_id=job_id,
                job_name=job_name,
                action=f"promoted to {promotion.to_environment} ({promotion.approval_status})",
                user=promotion.promoted_by,
                timestamp=promotion.promoted_at,
            ))

    # Sort by timestamp descending and limit
    activities.sort(key=lambda a: a.timestamp, reverse=True)
    return activities[:limit]
