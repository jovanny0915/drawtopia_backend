"""
Validation and generation metrics for monitoring dashboard and alerting.
Aggregates data from completed book generation jobs (scene validation, regeneration, timing).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Alert thresholds (Day 3 success criteria).
# MANUAL REVIEW: See docs/IMPLEMENTATION_NOTES.md — Jovanny may adjust based on production metrics.
AVG_CONFIDENCE_ALERT_THRESHOLD = 85  # Alert if average confidence drops below 85% (starting point)
REGENERATION_RATE_ALERT_THRESHOLD = 5.0  # Alert if regeneration rate exceeds 5% (starting point)
INTERACTIVE_SEARCH_TIME_BUDGET_SECONDS = 6 * 60  # 6 min
STORY_ADVENTURE_TIME_BUDGET_SECONDS = 4 * 60  # 4 min


def _parse_iso_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _extract_scene_validation_from_stages(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract validation data from scene_creation stages (result_data.validation)."""
    out = []
    for s in stages or []:
        if s.get("stage_name") != "scene_creation":
            continue
        rd = s.get("result_data") or {}
        validation = rd.get("validation") if isinstance(rd.get("validation"), dict) else {}
        confidence = rd.get("confidence_score")
        if confidence is None and isinstance(validation, dict):
            confidence = validation.get("confidence_score")
        out.append({
            "job_id": s.get("job_id"),
            "stage_id": s.get("id"),
            "scene_index": s.get("scene_index"),
            "confidence_score": confidence,
            "regeneration_count": (validation or {}).get("regeneration_count", 0),
            "approved_with_warning": (validation or {}).get("approved_with_warning", False),
        })
    return out


def _job_duration_seconds(job: Dict[str, Any]) -> Optional[float]:
    """Compute job duration from started_at to completed_at."""
    started = _parse_iso_datetime(job.get("started_at"))
    completed = _parse_iso_datetime(job.get("completed_at"))
    if not started or not completed:
        return None
    delta = completed - started
    if started.tzinfo:
        # Normalize to UTC for comparison
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        delta = completed - started
    return max(0, delta.total_seconds())


def compute_validation_metrics(
    jobs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate validation metrics from completed jobs.
    Returns dashboard-ready dict: average_confidence, regeneration_rate, generation_times, etc.
    """
    scene_validations: List[Dict[str, Any]] = []
    job_durations: List[float] = []
    by_type: Dict[str, Dict[str, Any]] = {
        "interactive_search": {"count": 0, "durations": [], "scenes": []},
        "story_adventure": {"count": 0, "durations": [], "scenes": []},
    }

    for job in jobs or []:
        jtype = job.get("job_type") or "unknown"
        if jtype not in by_type:
            by_type[jtype] = {"count": 0, "durations": [], "scenes": []}
        by_type[jtype]["count"] += 1

        duration = _job_duration_seconds(job)
        if duration is not None:
            job_durations.append(duration)
            by_type[jtype]["durations"].append(duration)

        stages = job.get("stages") or []
        for v in _extract_scene_validation_from_stages(stages):
            scene_validations.append(v)
            by_type[jtype]["scenes"].append(v)

    # Average confidence (0-100); only scenes with a numeric score
    confidences = [v["confidence_score"] for v in scene_validations if isinstance(v.get("confidence_score"), (int, float))]
    average_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None

    # Regeneration rate: % of scenes that had at least one regeneration
    scenes_with_regen = sum(1 for v in scene_validations if (v.get("regeneration_count") or 0) > 0)
    total_scenes = len(scene_validations)
    regeneration_rate = round(100.0 * scenes_with_regen / total_scenes, 2) if total_scenes else 0.0

    # Approved with warning rate
    approved_with_warning = sum(1 for v in scene_validations if v.get("approved_with_warning"))
    approved_with_warning_rate = round(100.0 * approved_with_warning / total_scenes, 2) if total_scenes else 0.0

    # Generation time stats (seconds)
    avg_duration = round(sum(job_durations) / len(job_durations), 2) if job_durations else None
    min_duration = min(job_durations) if job_durations else None
    max_duration = max(job_durations) if job_durations else None

    by_type_agg = {}
    for k, v in by_type.items():
        durs = v["durations"]
        scs = v["scenes"]
        confs = [x["confidence_score"] for x in scs if isinstance(x.get("confidence_score"), (int, float))]
        regens = sum(1 for x in scs if (x.get("regeneration_count") or 0) > 0)
        by_type_agg[k] = {
            "job_count": v["count"],
            "scene_count": len(scs),
            "average_confidence": round(sum(confs) / len(confs), 2) if confs else None,
            "regeneration_rate_pct": round(100.0 * regens / len(scs), 2) if scs else 0.0,
            "avg_duration_seconds": round(sum(durs) / len(durs), 2) if durs else None,
            "time_budget_seconds": INTERACTIVE_SEARCH_TIME_BUDGET_SECONDS if k == "interactive_search" else STORY_ADVENTURE_TIME_BUDGET_SECONDS,
        }

    return {
        "period_job_count": len(jobs),
        "total_scenes_validated": total_scenes,
        "average_confidence": average_confidence,
        "regeneration_rate_pct": regeneration_rate,
        "approved_with_warning_rate_pct": approved_with_warning_rate,
        "avg_generation_time_seconds": avg_duration,
        "min_generation_time_seconds": min_duration,
        "max_generation_time_seconds": max_duration,
        "by_job_type": by_type_agg,
        "computed_at": datetime.utcnow().isoformat() + "Z",
        # Placeholder for API cost per book (can be wired to usage APIs later).
        # Vision API: monitor monthly; see docs/IMPLEMENTATION_NOTES.md (Cost Management).
        "api_cost_per_book": None,
    }


def check_alerts(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate metrics against thresholds; return list of alert objects.
    Alerts if average confidence < 85% or regeneration rate > 5%.
    """
    alerts: List[Dict[str, Any]] = []
    avg = metrics.get("average_confidence")
    if avg is not None and avg < AVG_CONFIDENCE_ALERT_THRESHOLD:
        alerts.append({
            "level": "warning",
            "code": "LOW_AVG_CONFIDENCE",
            "message": f"Average validation confidence is {avg}% (below {AVG_CONFIDENCE_ALERT_THRESHOLD}% threshold)",
            "value": avg,
            "threshold": AVG_CONFIDENCE_ALERT_THRESHOLD,
        })
    regen = metrics.get("regeneration_rate_pct")
    if regen is not None and regen > REGENERATION_RATE_ALERT_THRESHOLD:
        alerts.append({
            "level": "warning",
            "code": "HIGH_REGENERATION_RATE",
            "message": f"Regeneration rate is {regen}% (above {REGENERATION_RATE_ALERT_THRESHOLD}% threshold)",
            "value": regen,
            "threshold": REGENERATION_RATE_ALERT_THRESHOLD,
        })
    for jtype, agg in (metrics.get("by_job_type") or {}).items():
        budget = agg.get("time_budget_seconds")
        avg_dur = agg.get("avg_duration_seconds")
        if budget is not None and avg_dur is not None and avg_dur > budget:
            alerts.append({
                "level": "warning",
                "code": "GENERATION_TIME_OVER_BUDGET",
                "message": f"{jtype}: average generation time {avg_dur}s exceeds budget {budget}s",
                "job_type": jtype,
                "avg_seconds": avg_dur,
                "budget_seconds": budget,
            })
    return alerts
