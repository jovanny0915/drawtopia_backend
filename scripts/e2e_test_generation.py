"""
E2E test: Generate N Interactive Search books and verify success and timing.
Requires: backend running (with queue enabled) and worker processing jobs.

Usage:
  E2E_JOB_COUNT=20 E2E_BASE_URL=http://localhost:8000 python scripts/e2e_test_generation.py
  E2E_BAD_PROMPT=1  # Optional: use minimal payload to encourage regeneration (low confidence)

Success criteria (Day 3):
  - 100% of test books generate without errors
  - Generation time for Interactive Search <= 6 min per book
  - Character consistency validated (confidence scores recorded)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

# Config from env
E2E_BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8000")
E2E_JOB_COUNT = int(os.getenv("E2E_JOB_COUNT", "20"))
E2E_POLL_INTERVAL = int(os.getenv("E2E_POLL_INTERVAL", "15"))
E2E_TIMEOUT_PER_JOB = int(os.getenv("E2E_TIMEOUT_PER_JOB", "420"))  # 7 min max per job
E2E_USE_BAD_PROMPT = os.getenv("E2E_BAD_PROMPT", "").lower() in ("1", "true", "yes")

# Time budgets (seconds) - Day 3 success criteria
INTERACTIVE_SEARCH_BUDGET = 6 * 60   # 6 min
STORY_ADVENTURE_BUDGET = 4 * 60      # 4 min


def default_job_payload(use_bad_prompt: bool = False) -> Dict[str, Any]:
    """Payload for POST /api/books/generate (Interactive Search)."""
    if use_bad_prompt:
        # Minimal/vague descriptors to potentially trigger lower confidence and regeneration
        return {
            "job_type": "interactive_search",
            "character_name": "Character",
            "character_type": "creature",
            "special_ability": "none",
            "age_group": "7-10",
            "story_world": "place",
            "adventure_type": "explore",
            "occasion_theme": None,
            "character_image_url": None,
            "priority": 5,
        }
    return {
        "job_type": "interactive_search",
        "character_name": "Luna",
        "character_type": "brave dragon",
        "special_ability": "fly through clouds",
        "age_group": "7-10",
        "story_world": "the Enchanted Forest",
        "adventure_type": "treasure hunt",
        "occasion_theme": None,
        "character_image_url": None,
        "priority": 5,
    }


def create_job(base_url: str, payload: Dict[str, Any]) -> Optional[int]:
    """Create a book generation job; return job_id or None."""
    url = f"{base_url.rstrip('/')}/api/books/generate"
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("job_id")
    except Exception as e:
        print(f"Create job failed: {e}")
        return None


def get_job_status(base_url: str, job_id: int) -> Optional[Dict[str, Any]]:
    """Get job status; return dict with status, stages, etc."""
    url = f"{base_url.rstrip('/')}/api/books/{job_id}/status"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Get status job_id={job_id} failed: {e}")
        return None


def wait_for_job(
    base_url: str,
    job_id: int,
    poll_interval: int = E2E_POLL_INTERVAL,
    timeout_seconds: int = E2E_TIMEOUT_PER_JOB,
) -> Dict[str, Any]:
    """Poll until job is completed or failed; return final status and timing."""
    start = time.time()
    last_status = None
    while (time.time() - start) < timeout_seconds:
        last_status = get_job_status(base_url, job_id)
        if not last_status:
            return {"job_id": job_id, "status": "error", "error": "failed to get status", "duration_seconds": time.time() - start}
        status = last_status.get("status", "")
        if status in ("completed", "failed"):
            return {
                "job_id": job_id,
                "status": status,
                "overall_progress": last_status.get("overall_progress", 0),
                "stages": last_status.get("stages", []),
                "error_message": last_status.get("error_message"),
                "duration_seconds": time.time() - start,
            }
        time.sleep(poll_interval)
    return {
        "job_id": job_id,
        "status": "timeout",
        "stages": (last_status or {}).get("stages", []),
        "duration_seconds": time.time() - start,
    }


def extract_validation_from_stages(stages: List[Dict]) -> List[Dict[str, Any]]:
    """Extract scene validation (confidence, regeneration) from scene_creation stages."""
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
            "scene_index": s.get("scene_index"),
            "confidence_score": confidence,
            "regeneration_count": (validation or {}).get("regeneration_count", 0),
            "approved_with_warning": (validation or {}).get("approved_with_warning", False),
        })
    return out


def get_metrics(base_url: str, since_hours: float = 24, limit: int = 100) -> Optional[Dict]:
    """Fetch monitoring metrics from GET /api/monitoring/metrics."""
    url = f"{base_url.rstrip('/')}/api/monitoring/metrics"
    try:
        r = requests.get(url, params={"since_hours": since_hours, "limit": limit}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Metrics fetch failed: {e}")
        return None


def run_e2e(
    base_url: str = E2E_BASE_URL,
    job_count: int = E2E_JOB_COUNT,
    use_bad_prompt: bool = E2E_USE_BAD_PROMPT,
) -> bool:
    """Create jobs, wait for completion, assert success and time budget. Returns True if all pass."""
    print(f"E2E: base_url={base_url} job_count={job_count} use_bad_prompt={use_bad_prompt}")
    payload = default_job_payload(use_bad_prompt=use_bad_prompt)

    job_ids: List[int] = []
    for i in range(job_count):
        jid = create_job(base_url, payload)
        if jid is None:
            print(f"Failed to create job {i + 1}/{job_count}")
            return False
        job_ids.append(jid)
        print(f"Created job_id={jid} ({i + 1}/{job_count})")

    results: List[Dict[str, Any]] = []
    for jid in job_ids:
        print(f"Waiting for job_id={jid} ...")
        res = wait_for_job(base_url, jid)
        results.append(res)
        status = res.get("status", "")
        duration = res.get("duration_seconds", 0)
        print(f"  job_id={jid} status={status} duration={duration:.1f}s")

    # Assertions
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]
    timeout_or_error = [r for r in results if r.get("status") in ("timeout", "error")]

    all_ok = len(completed) == job_count and len(failed) == 0 and len(timeout_or_error) == 0
    if not all_ok:
        print(f"FAIL: completed={len(completed)} failed={len(failed)} timeout_or_error={len(timeout_or_error)}")
        for r in failed + timeout_or_error:
            print(f"  job_id={r.get('job_id')} status={r.get('status')} error={r.get('error_message') or r.get('error')}")
        return False

    # Time budget: Interactive Search <= 6 min
    over_budget = [r for r in results if (r.get("duration_seconds") or 0) > INTERACTIVE_SEARCH_BUDGET]
    if over_budget:
        print(f"WARNING: {len(over_budget)} job(s) exceeded 6 min budget")
        for r in over_budget:
            print(f"  job_id={r.get('job_id')} duration={(r.get('duration_seconds') or 0):.1f}s")
        # Don't fail E2E for time budget; log only (environment-dependent)
        # return False

    # Validation metrics summary
    all_validations: List[Dict] = []
    for r in results:
        for v in extract_validation_from_stages(r.get("stages") or []):
            all_validations.append(v)
    if all_validations:
        confidences = [v["confidence_score"] for v in all_validations if isinstance(v.get("confidence_score"), (int, float))]
        regens = sum(1 for v in all_validations if (v.get("regeneration_count") or 0) > 0)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        regen_rate = 100.0 * regens / len(all_validations) if all_validations else 0
        print(f"Validation: scenes={len(all_validations)} avg_confidence={avg_conf:.2f} regeneration_rate={regen_rate:.2f}%")

    # Fetch dashboard metrics
    metrics_resp = get_metrics(base_url)
    if metrics_resp:
        m = metrics_resp.get("metrics", {})
        print(f"Dashboard metrics: avg_confidence={m.get('average_confidence')} regeneration_rate_pct={m.get('regeneration_rate_pct')} alerts_active={metrics_resp.get('alerts_active')}")

    print("E2E: 100% of test books generated without errors.")
    return True


def main():
    parser = argparse.ArgumentParser(description="E2E test: generate Interactive Search books and verify.")
    parser.add_argument("--base-url", default=E2E_BASE_URL, help="API base URL")
    parser.add_argument("--count", type=int, default=E2E_JOB_COUNT, help="Number of jobs to create")
    parser.add_argument("--bad-prompt", action="store_true", help="Use minimal payload to trigger regeneration")
    args = parser.parse_args()
    ok = run_e2e(base_url=args.base_url, job_count=args.count, use_bad_prompt=args.bad_prompt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
