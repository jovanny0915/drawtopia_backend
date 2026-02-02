"""
Monitoring and dashboard API for validation metrics and alerting.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["monitoring"])


@router.get("/api/monitoring/metrics")
async def get_validation_metrics(
    job_type: Optional[str] = Query(None, description="Filter by job_type: interactive_search, story_adventure"),
    since_hours: float = Query(24.0, ge=0.1, le=720, description="Time window in hours"),
    limit: int = Query(100, ge=1, le=500, description="Max jobs to include"),
):
    """
    Real-time validation metrics for dashboard.
    Returns average confidence, regeneration rate, generation times, and alerts.
    """
    import main
    from metrics import compute_validation_metrics, check_alerts

    if not main.queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    jobs = main.queue_manager.get_recent_completed_jobs(
        job_type=job_type,
        since_hours=since_hours,
        limit=limit,
    )
    metrics = compute_validation_metrics(jobs)
    alerts = check_alerts(metrics)

    # Log regeneration rate for monitoring (target <5%)
    regen = metrics.get("regeneration_rate_pct")
    if regen is not None:
        main.logger.info(
            "monitoring regeneration_rate_pct=%.2f job_count=%s",
            regen,
            metrics.get("period_job_count", 0),
            extra={"regeneration_rate_pct": regen, "period_job_count": metrics.get("period_job_count")},
        )
    if alerts:
        for a in alerts:
            main.logger.warning(
                "monitoring_alert %s: %s",
                a.get("code", ""),
                a.get("message", ""),
                extra=a,
            )

    return {
        "metrics": metrics,
        "alerts": alerts,
        "alerts_active": len(alerts) > 0,
    }
