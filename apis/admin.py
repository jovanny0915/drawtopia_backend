"""
Admin API routes for book template management
Handles all admin operations including:
- Template CRUD operations
- Image uploads to Supabase storage
- Storage bucket file management
- Image optimization before upload
"""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
from rate_limiter import limiter
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json
import random
import os
import logging
from uuid import uuid4
from image_optimizer import TemplateImageOptimizer
from storage_utils import (
    delete_story_images,
    delete_character_images,
    delete_files_from_storage,
    collect_book_template_image_urls,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize image optimizer for template images
image_optimizer = TemplateImageOptimizer()


# ==================== Pydantic Models ====================

class BookTemplateCreate(BaseModel):
    """Request model for creating a new book template"""
    name: str
    story_world: Optional[str] = None  # 'forest', 'underwater', or 'outerspace'
    story_style: Optional[str] = None  # '3d', 'anime', or 'cartoon'
    story_type: Optional[str] = None   # alias for story_style
    story_format: Optional[str] = None  # e.g. adventure_story, interactive_story; free-form text
    character_for_finding: Optional[List[str]] = None
    # positions: JSON array of coordinate objects for template characters
    positions: Optional[List[Dict[str, float]]] = None


class BookTemplateUpdate(BaseModel):
    """Request model for updating book template metadata"""
    name: Optional[str] = None
    story_world: Optional[str] = None  # 'forest', 'underwater', or 'outerspace'
    story_style: Optional[str] = None  # '3d', 'anime', or 'cartoon'
    story_type: Optional[str] = None   # alias for story_style
    cover_image: Optional[str] = None
    copyright_page_image: Optional[str] = None
    dedication_page_image: Optional[str] = None
    story_page_images: Optional[List[str]] = None
    character_for_finding: Optional[List[str]] = None
    # positions: list of coordinate objects: [{"x": float, "y": float}, ...]
    positions: Optional[List[Dict[str, float]]] = None
    last_words_page_image: Optional[str] = None
    last_story_page_image: Optional[str] = None
    back_cover_image: Optional[str] = None
    story_format: Optional[str] = None  # e.g. adventure_story, interactive_story; free-form text


class BookTemplateResponse(BaseModel):
    """Response model for book template"""
    id: str
    name: str
    story_world: Optional[str] = None  # 'forest', 'underwater', or 'outerspace'
    story_style: Optional[str] = None  # '3d', 'anime', or 'cartoon'
    story_type: Optional[str] = None   # alias for story_style
    cover_image: Optional[str] = None
    copyright_page_image: Optional[str] = None
    dedication_page_image: Optional[str] = None
    story_page_images: Optional[List[str]] = None
    character_for_finding: Optional[List[str]] = None
    positions: Optional[List[Dict[str, float]]] = None
    last_words_page_image: Optional[str] = None
    last_story_page_image: Optional[str] = None
    back_cover_image: Optional[str] = None
    story_format: Optional[str] = None  # e.g. adventure_story, interactive_story; free-form text
    created_at: Optional[str] = None


class AdminUserCreate(BaseModel):
    """Request model for creating a user profile from admin panel"""
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[str] = "user"
    subscription_status: Optional[str] = None
    credit: Optional[int] = 0


class AdminUserUpdate(BaseModel):
    """Request model for updating user profile fields from admin panel"""
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[str] = None
    subscription_status: Optional[str] = None
    credit: Optional[int] = None


# ==================== Helper Functions ====================

def get_supabase_client():
    """Get Supabase client from main module"""
    import main
    if not main.supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
    return main.supabase


VALID_TEMPLATE_STORY_STYLES = {
    "3d",
    "anime",
    "cartoon",
    "story",
    "search",
    "adventure",
    "search-and-find",
}

# Book product line: adventure (linear) vs interactive; stored on book_templates.story_format
VALID_BOOK_TEMPLATE_STORY_FORMATS = frozenset({"adventure_story", "interactive_story"})


def normalize_book_template_story_format(value: Optional[str]) -> Optional[str]:
    """Return canonical story_format or None if empty."""
    if value is None:
        return None
    s = value.strip().lower().replace("-", "_")
    return s or None


def effective_template_story_format(row_story_format: Optional[str]) -> str:
    """DB null/empty is treated as adventure_story for legacy rows."""
    n = normalize_book_template_story_format(row_story_format)
    return n if n in VALID_BOOK_TEMPLATE_STORY_FORMATS else "adventure_story"


def normalize_story_style(value: Optional[str]) -> Optional[str]:
    """Normalize story style values for consistent validation/storage."""
    if value is None:
        return None

    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    return normalized or None


SUBSCRIPTION_ACTIVE_STATUSES = {
    "premium",
    "active",
    "trialing",
    "cancelled",
    "canceled",
    "past_due",
    "unpaid",
    "incomplete",
    "incomplete_expired",
}

FOUNDING_MEMBER_MAX_AMOUNT = 170.0


def _normalize_text_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _safe_parse_datetime(value: Optional[Any], end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            if len(candidate) == 10:
                parsed_date = datetime.fromisoformat(candidate)
                if end_of_day:
                    return parsed_date + timedelta(days=1) - timedelta(microseconds=1)
                return parsed_date
            return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _normalize_datetime_for_compare(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_parse_amount(value: Optional[Any]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if amount > 1000:
        return amount / 100.0
    return amount


def _full_name(row: Dict[str, Any]) -> str:
    return f"{row.get('first_name') or ''} {row.get('last_name') or ''}".strip()


def _derive_account_type(
    user_row: Dict[str, Any],
    latest_subscription: Optional[Dict[str, Any]],
    story_count: int,
    purchase_count: int,
    child_count: int,
) -> str:
    subscription_status = _normalize_text_filter(user_row.get("subscription_status"))
    has_subscription = latest_subscription is not None or subscription_status in SUBSCRIPTION_ACTIVE_STATUSES

    if has_subscription:
        amount = _safe_parse_amount((latest_subscription or {}).get("amount"))
        if amount is not None and amount <= FOUNDING_MEMBER_MAX_AMOUNT:
            return "founding_member"
        return "family"

    if purchase_count > 0 or story_count > 0 or child_count > 0:
        return "individual"

    return "free"


def _build_user_summary(
    user_row: Dict[str, Any],
    story_count_by_user: Dict[str, int],
    latest_login_by_user: Dict[str, Optional[str]],
    latest_subscription_by_user: Dict[str, Dict[str, Any]],
    purchase_summary_by_user: Dict[str, Dict[str, Any]],
    child_count_by_user: Dict[str, int],
) -> Dict[str, Any]:
    user_id = str(user_row.get("id"))
    story_count = story_count_by_user.get(user_id, 0)
    purchase_summary = purchase_summary_by_user.get(user_id, {})
    child_count = child_count_by_user.get(user_id, 0)
    latest_subscription = latest_subscription_by_user.get(user_id)

    last_login = user_row.get("last_login") or latest_login_by_user.get(user_id)
    account_type = _derive_account_type(
        user_row=user_row,
        latest_subscription=latest_subscription,
        story_count=story_count,
        purchase_count=int(purchase_summary.get("purchase_count", 0) or 0),
        child_count=child_count,
    )

    return {
        "id": user_id,
        "email": user_row.get("email"),
        "first_name": user_row.get("first_name"),
        "last_name": user_row.get("last_name"),
        "full_name": _full_name(user_row),
        "avatar_url": user_row.get("avatar_url"),
        "role": user_row.get("role"),
        "subscription_status": user_row.get("subscription_status"),
        "subscription_expires": user_row.get("subscription_expires"),
        "credit": user_row.get("credit"),
        "created_at": user_row.get("created_at"),
        "registration_date": user_row.get("created_at"),
        "last_login": last_login,
        "account_type": account_type,
        "total_stories_created": story_count,
        "story_count": story_count,
        "child_count": child_count,
        "purchase_count": int(purchase_summary.get("purchase_count", 0) or 0),
        "total_amount_paid": purchase_summary.get("total_amount_paid", 0.0) or 0.0,
        "latest_subscription": latest_subscription,
    }


def _matches_date_range(value: Optional[Any], start: Optional[datetime], end: Optional[datetime]) -> bool:
    if start is None and end is None:
        return True
    candidate = _normalize_datetime_for_compare(_safe_parse_datetime(value))
    if candidate is None:
        return False
    start = _normalize_datetime_for_compare(start)
    end = _normalize_datetime_for_compare(end)
    if start is not None and candidate < start:
        return False
    if end is not None and candidate > end:
        return False
    return True


def _filter_user_summaries(
    summaries: List[Dict[str, Any]],
    search: Optional[str],
    account_type: Optional[str],
    subscription_status: Optional[str],
    registered_from: Optional[datetime],
    registered_to: Optional[datetime],
    story_count_min: Optional[int],
    story_count_max: Optional[int],
) -> List[Dict[str, Any]]:
    normalized_search = _normalize_text_filter(search)
    normalized_account_type = _normalize_text_filter(account_type)
    normalized_subscription_status = _normalize_text_filter(subscription_status)

    filtered: List[Dict[str, Any]] = []
    for summary in summaries:
        if normalized_search:
            haystack = " ".join([
                str(summary.get("id") or ""),
                str(summary.get("email") or ""),
                str(summary.get("full_name") or ""),
            ]).lower()
            if normalized_search not in haystack:
                continue

        if normalized_account_type and _normalize_text_filter(summary.get("account_type")) != normalized_account_type:
            continue

        if normalized_subscription_status and _normalize_text_filter(summary.get("subscription_status")) != normalized_subscription_status:
            continue

        if not _matches_date_range(summary.get("created_at"), registered_from, registered_to):
            continue

        story_count = int(summary.get("total_stories_created", 0) or 0)
        if story_count_min is not None and story_count < story_count_min:
            continue
        if story_count_max is not None and story_count > story_count_max:
            continue

        filtered.append(summary)

    return filtered


def _resolve_story_owner_user_id(
    story_row: Dict[str, Any],
    child_parent_by_id: Dict[str, str],
) -> Optional[str]:
    user_id = story_row.get("user_id")
    if user_id:
        return str(user_id)

    child_profile_id = story_row.get("child_profile_id")
    if child_profile_id is None:
        return None

    return child_parent_by_id.get(str(child_profile_id))


def _build_story_count_by_user(
    stories: List[Dict[str, Any]],
    child_parent_by_id: Dict[str, str],
) -> Dict[str, int]:
    story_count_by_user: Dict[str, int] = defaultdict(int)
    for row in stories:
        owner_user_id = _resolve_story_owner_user_id(row, child_parent_by_id)
        if owner_user_id:
            story_count_by_user[owner_user_id] += 1
    return story_count_by_user


INTERACTIVE_STORY_FORMATS = frozenset({
    "interactive",
    "interactive_search",
    "interactive_story",
    "search",
    "search_and_find",
    "search-and-find",
    "intersearch",
})


def _normalize_story_format(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _effective_story_format(story_row: Dict[str, Any], job_row: Optional[Dict[str, Any]] = None) -> str:
    for raw_value in (
        story_row.get("story_format"),
        story_row.get("story_type"),
        (job_row or {}).get("job_type"),
    ):
        normalized = _normalize_story_format(raw_value)
        if normalized in INTERACTIVE_STORY_FORMATS:
            return "interactive_search"
        if normalized in {"story", "story_adventure", "adventure_story", "adventure"}:
            return "story_adventure"
    return "story_adventure"


def _effective_story_status(story_row: Dict[str, Any], job_row: Optional[Dict[str, Any]] = None) -> str:
    job_status = _normalize_text_filter((job_row or {}).get("status"))
    if job_status in {"processing", "pending"}:
        return "generating"
    if job_status in {"completed", "failed", "cancelled"}:
        return "failed" if job_status == "cancelled" else job_status

    story_status = _normalize_text_filter(story_row.get("status"))
    if story_status in {"draft", "generating", "completed", "failed"}:
        return story_status
    return "draft"


def _calculate_job_duration_seconds(job_row: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(job_row, dict):
        return None

    started_at = _safe_parse_datetime(job_row.get("started_at"))
    completed_at = _safe_parse_datetime(job_row.get("completed_at"))
    if started_at is None or completed_at is None:
        return None

    started_at = _normalize_datetime_for_compare(started_at)
    completed_at = _normalize_datetime_for_compare(completed_at)
    if started_at is None or completed_at is None or completed_at < started_at:
        return None

    return round((completed_at - started_at).total_seconds(), 2)


def _group_jobs_by_book_id(job_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    jobs_by_book_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job_row in job_rows:
        book_id = job_row.get("book_id")
        if book_id is None:
            continue
        jobs_by_book_id[str(book_id)].append(job_row)

    for book_id, rows in jobs_by_book_id.items():
        rows.sort(
            key=lambda row: _safe_parse_datetime(row.get("created_at")) or datetime.min,
            reverse=True,
        )
        jobs_by_book_id[book_id] = rows

    return jobs_by_book_id


def _pick_latest_story_job(
    story_row: Dict[str, Any],
    jobs_by_book_id: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    story_id = story_row.get("id")
    if story_id is None:
        return None
    jobs = jobs_by_book_id.get(str(story_id)) or []
    return jobs[0] if jobs else None


def _extract_story_page_texts(story_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    story_content = story_row.get("story_content")
    if story_content is None:
        return []

    parsed_content: Any = story_content
    if isinstance(parsed_content, str):
        trimmed = parsed_content.strip()
        if not trimmed:
            return []
        try:
            parsed_content = json.loads(trimmed)
        except Exception:
            return [{"page_number": 1, "text": trimmed, "audio_url": None}]

    pages = parsed_content.get("pages") if isinstance(parsed_content, dict) else parsed_content
    if not isinstance(pages, list):
        return []

    results: List[Dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        if isinstance(page, str):
            text = page.strip()
            audio_url = None
        elif isinstance(page, dict):
            text = str(
                page.get("text")
                or page.get("story")
                or page.get("pageText")
                or ""
            ).strip()
            audio_url = page.get("audioUrl") or page.get("audio_url")
        else:
            continue

        if not text:
            continue

        page_number = page.get("pageNumber") if isinstance(page, dict) else index
        try:
            normalized_page_number = int(page_number)
        except Exception:
            normalized_page_number = index

        results.append({
            "page_number": normalized_page_number,
            "text": text,
            "audio_url": audio_url,
        })

    return results


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _listify_urls(value: Any) -> List[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _build_story_pages(
    story_row: Dict[str, Any],
    story_format: str,
    story_page_texts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []

    for label, image_url in (
        ("Copyright", _first_non_empty(story_row.get("copyright_image"), story_row.get("copyright_page_image"))),
        ("Dedication", _first_non_empty(story_row.get("dedication_image"), story_row.get("dedication_page_image"))),
    ):
        if image_url:
            pages.append({
                "key": label.lower(),
                "label": label,
                "image_url": image_url,
                "page_number": None,
                "text": None,
            })

    scene_images = _listify_urls(story_row.get("scene_images"))
    item_label = "Scene" if story_format == "interactive_search" else "Page"
    for index, image_url in enumerate(scene_images, start=1):
        matching_text = next((item for item in story_page_texts if item.get("page_number") == index), None)
        pages.append({
            "key": f"scene-{index}",
            "label": f"{item_label} {index}",
            "image_url": image_url,
            "page_number": index,
            "text": matching_text.get("text") if matching_text else None,
        })

    for label, image_url in (
        ("Last Words", _first_non_empty(story_row.get("last_word_page_image"), story_row.get("last_words_page_image"))),
        ("Final Page", _first_non_empty(story_row.get("last_admin_page_image"), story_row.get("last_story_page_image"))),
        ("Back Cover", _first_non_empty(story_row.get("back_cover_image"), story_row.get("back_page_image"))),
    ):
        if image_url:
            pages.append({
                "key": label.lower().replace(" ", "-"),
                "label": label,
                "image_url": image_url,
                "page_number": None,
                "text": None,
            })

    return pages


def _build_story_owner_summary(
    story_row: Dict[str, Any],
    users_by_id: Dict[str, Dict[str, Any]],
    child_parent_by_id: Dict[str, str],
    child_profiles_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    owner_user_id = _resolve_story_owner_user_id(story_row, child_parent_by_id)
    owner = users_by_id.get(str(owner_user_id)) if owner_user_id else None
    child_profile_id = story_row.get("child_profile_id")
    child_profile = child_profiles_by_id.get(str(child_profile_id)) if child_profile_id is not None else None

    owner_name = _full_name(owner or {})
    if not owner_name:
        owner_name = (owner or {}).get("email") or "Unknown user"

    return {
        "user_id": str(owner_user_id) if owner_user_id else None,
        "user_email": (owner or {}).get("email"),
        "user_name": owner_name,
        "child_name": (child_profile or {}).get("first_name"),
    }


def _build_admin_story_summary(
    story_row: Dict[str, Any],
    users_by_id: Dict[str, Dict[str, Any]],
    child_parent_by_id: Dict[str, str],
    child_profiles_by_id: Dict[str, Dict[str, Any]],
    characters_by_id: Dict[str, Dict[str, Any]],
    jobs_by_book_id: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    latest_job = _pick_latest_story_job(story_row, jobs_by_book_id)
    owner_summary = _build_story_owner_summary(
        story_row=story_row,
        users_by_id=users_by_id,
        child_parent_by_id=child_parent_by_id,
        child_profiles_by_id=child_profiles_by_id,
    )
    character = characters_by_id.get(str(story_row.get("character_id"))) if story_row.get("character_id") is not None else None
    story_format = _effective_story_format(story_row, latest_job)

    character_name = (
        story_row.get("character_name")
        or (character or {}).get("character_name")
        or "Untitled character"
    )

    return {
        "id": str(story_row.get("id")) if story_row.get("id") is not None else "",
        "uid": story_row.get("uid"),
        "story_title": story_row.get("story_title") or "Untitled story",
        "character_name": character_name,
        "format": story_format,
        "status": _effective_story_status(story_row, latest_job),
        "created_at": story_row.get("created_at"),
        "generation_duration_seconds": _calculate_job_duration_seconds(latest_job),
        "user_id": owner_summary["user_id"],
        "user_email": owner_summary["user_email"],
        "user_name": owner_summary["user_name"],
        "child_name": owner_summary["child_name"],
        "cover_image": story_row.get("story_cover"),
        "error_message": (latest_job or {}).get("error_message"),
    }


def _filter_admin_story_summaries(
    summaries: List[Dict[str, Any]],
    search: Optional[str],
    status: Optional[str],
    format_type: Optional[str],
    created_from: Optional[datetime],
    created_to: Optional[datetime],
) -> List[Dict[str, Any]]:
    normalized_search = _normalize_text_filter(search)
    normalized_status = _normalize_text_filter(status)
    normalized_format = _normalize_story_format(format_type)

    filtered: List[Dict[str, Any]] = []
    for summary in summaries:
        if normalized_search:
            haystack = " ".join([
                str(summary.get("user_email") or ""),
                str(summary.get("character_name") or ""),
                str(summary.get("story_title") or ""),
            ]).lower()
            if normalized_search not in haystack:
                continue

        if normalized_status and _normalize_text_filter(summary.get("status")) != normalized_status:
            continue

        if normalized_format and _normalize_story_format(summary.get("format")) != normalized_format:
            continue

        if not _matches_date_range(summary.get("created_at"), created_from, created_to):
            continue

        filtered.append(summary)

    return filtered


def _fetch_user_admin_context(supabase) -> Dict[str, Any]:

    # Fetch users from Supabase 'users' table (view of auth.users)
    # Fetch users from 'users' table (Supabase auth exposes first_name, last_name, avatar_url, etc. as columns)
    users_response = supabase.table("users").select(
        "id,email,created_at,first_name,last_name,avatar_url,role,subscription_status,subscription_expires,credit,stripe_customer_id"
    ).order("created_at", desc=True).execute()
    users = users_response.data or []

    # Set last_login to None for compatibility. All other fields come from users table only.
    for user in users:
        user["last_login"] = None
        # Ensure these fields are only from users table
        user["role"] = user.get("role")
        user["subscription_status"] = user.get("subscription_status")
        user["subscription_expires"] = user.get("subscription_expires")
        user["credit"] = user.get("credit")
        user["stripe_customer_id"] = user.get("stripe_customer_id")

    child_profiles_response = supabase.table("child_profiles").select("id,parent_id,first_name,created_at,avatar_url").execute()
    child_profiles = child_profiles_response.data or []
    child_parent_by_id: Dict[str, str] = {}
    child_count_by_user: Dict[str, int] = defaultdict(int)
    for row in child_profiles:
        child_id = row.get("id")
        parent_id = row.get("parent_id")
        if child_id and parent_id:
            child_parent_by_id[str(child_id)] = str(parent_id)
            child_count_by_user[str(parent_id)] += 1

    stories_response = supabase.table("stories").select(
        "uid,user_id,child_profile_id,story_title,created_at,status,story_type,character_id,purchased"
    ).execute()
    stories = stories_response.data or []
    story_count_by_user = _build_story_count_by_user(stories, child_parent_by_id)

    auth_history_response = (
        supabase
        .table("user_auth_history")
        .select("user_id,event_type,created_at")
        .eq("event_type", "login")
        .order("created_at", desc=True)
        .execute()
    )
    latest_login_by_user: Dict[str, Optional[str]] = {}
    for row in (auth_history_response.data or []):
        user_id = row.get("user_id")
        if user_id and str(user_id) not in latest_login_by_user:
            latest_login_by_user[str(user_id)] = row.get("created_at")

    subscriptions_response = (
        supabase
        .table("subscriptions")
        .select(
            "user_id,status,plan_type,amount,current_period_start,current_period_end,"
            "created_at,updated_at,stripe_subscription_id"
        )
        .order("created_at", desc=True)
        .execute()
    )
    latest_subscription_by_user: Dict[str, Dict[str, Any]] = {}
    for row in (subscriptions_response.data or []):
        user_id = row.get("user_id")
        if user_id and str(user_id) not in latest_subscription_by_user:
            latest_subscription_by_user[str(user_id)] = row


    # Calculate purchase summary by user using stories.purchased
    purchase_summary_by_user: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "purchase_count": 0,
        "total_amount_paid": 0.0,  # No amount info in stories, so always 0.0
    })
    for row in stories:
        owner_user_id = _resolve_story_owner_user_id(row, child_parent_by_id)
        if not owner_user_id:
            continue
        key = str(owner_user_id)
        if row.get("purchased"):
            purchase_summary_by_user[key]["purchase_count"] += 1

    return {
        "users": users,
        "stories": stories,
        "story_count_by_user": story_count_by_user,
        "latest_login_by_user": latest_login_by_user,
        "latest_subscription_by_user": latest_subscription_by_user,
        "purchase_summary_by_user": purchase_summary_by_user,
        "child_profiles": child_profiles,
        "child_parent_by_id": child_parent_by_id,
        "child_count_by_user": child_count_by_user,
    }


async def upload_to_storage(file: UploadFile, bucket_name: str, file_path: str) -> str:
    """
    Upload file to Supabase storage with optimization and return public URL.
    Images are automatically optimized to WebP format before upload.
    """
    supabase = get_supabase_client()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Optimize image before upload
        logger.info(f"🔧 Optimizing image before upload: {file.filename}")
        try:
            optimized_content, content_type, extension = image_optimizer.optimize_image(
                file_content,
                filename=file.filename
            )
            
            # Update file path to use optimized extension
            if not file_path.endswith(f".{extension}"):
                # Replace original extension with optimized extension
                base_path = file_path.rsplit(".", 1)[0] if "." in file_path else file_path
                file_path = f"{base_path}.{extension}"
            
            logger.info(
                f"✅ Image optimized: {len(file_content) / 1024:.1f}KB → "
                f"{len(optimized_content) / 1024:.1f}KB "
                f"({content_type})"
            )
            
            # Use optimized content
            upload_content = optimized_content
            upload_content_type = content_type
            
        except Exception as opt_error:
            logger.warning(f"⚠️ Image optimization failed, uploading original: {opt_error}")
            # Fallback to original if optimization fails
            upload_content = file_content
            upload_content_type = file.content_type or "image/jpeg"
        
        # Upload to storage with upsert (overwrites existing file)
        response = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=upload_content,
            file_options={
                "content-type": upload_content_type,
                "upsert": "true"
            }
        )
        
        # Get public URL
        public_url_response = supabase.storage.from_(bucket_name).get_public_url(file_path)
        public_url = public_url_response
        
        logger.info(f"✅ Uploaded optimized file to storage: {file_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Error uploading file to storage: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file to storage: {str(e)}"
        )


def _delete_urls_or_raise(supabase_client, urls: List[str], context: str) -> None:
    """Delete storage URLs and raise when any deletion fails."""
    urls_to_delete = [url for url in urls if isinstance(url, str) and url.strip()]
    if not urls_to_delete:
        return

    deletion_stats = delete_files_from_storage(supabase_client, urls_to_delete)
    if deletion_stats.get("errors", 0) > 0:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to delete one or more files from storage for {context}. "
                "Database was not updated."
            )
        )


def _safe_delete_eq(supabase_client, table_name: str, column: str, value: Any) -> int:
    """Delete rows by equality and return deleted count (best-effort)."""
    try:
        response = supabase_client.table(table_name).delete().eq(column, value).execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        logger.warning(f"⚠️ Could not delete from {table_name} where {column}={value}: {e}")
        return 0


def _safe_delete_in(supabase_client, table_name: str, column: str, values: List[Any]) -> int:
    """Delete rows by IN list and return deleted count (best-effort)."""
    if not values:
        return 0
    try:
        response = supabase_client.table(table_name).delete().in_(column, values).execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        logger.warning(f"⚠️ Could not delete from {table_name} where {column} IN (...): {e}")
        return 0


# ==================== API Endpoints ====================

@router.get("/admin/analysis/story-counts-by-day")
@limiter.limit("60/minute")
async def get_story_counts_by_day(request: Request, days: int = Query(90, ge=7, le=365)):
    """
    Get counts of story generation per day from the stories table.
    Returns list of { date: "YYYY-MM-DD", count: number } for the last `days` days.
    """
    supabase = get_supabase_client()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        response = supabase.table("stories").select("created_at").gte("created_at", since).execute()
        rows = response.data if response.data else []
        # Group by date (day only)
        by_day = defaultdict(int)
        for row in rows:
            created = row.get("created_at")
            if not created:
                continue
            if isinstance(created, str):
                day = created[:10]  # "YYYY-MM-DD"
            else:
                day = datetime.fromisoformat(str(created).replace("Z", "+00:00")).strftime("%Y-%m-%d")
            by_day[day] += 1
        # Sort by date and return list
        result = [{"date": d, "count": c} for d, c in sorted(by_day.items())]
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error fetching story counts by day: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch story counts: {str(e)}")


@router.get("/admin/analysis/user-auth-counts-by-day")
@limiter.limit("60/minute")
async def get_user_auth_counts_by_day(request: Request, days: int = Query(90, ge=7, le=365)):
    """
    Get daily counts for login/register events from user_auth_history table.
    Returns:
    [
      { date: "YYYY-MM-DD", login_count: number, register_count: number, total_count: number }
    ]
    """
    supabase = get_supabase_client()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        response = (
            supabase
            .table("user_auth_history")
            .select("created_at,event_type")
            .gte("created_at", since)
            .execute()
        )
        rows = response.data if response.data else []

        # Group by day and auth event type.
        by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: {"login_count": 0, "register_count": 0})
        for row in rows:
            created = row.get("created_at")
            event_type = row.get("event_type")
            if not created or event_type not in ("login", "register"):
                continue

            if isinstance(created, str):
                day = created[:10]  # "YYYY-MM-DD"
            else:
                day = datetime.fromisoformat(str(created).replace("Z", "+00:00")).strftime("%Y-%m-%d")

            if event_type == "login":
                by_day[day]["login_count"] += 1
            elif event_type == "register":
                by_day[day]["register_count"] += 1

        result = []
        for day, counts in sorted(by_day.items()):
            login_count = counts["login_count"]
            register_count = counts["register_count"]
            result.append({
                "date": day,
                "login_count": login_count,
                "register_count": register_count,
                "total_count": login_count + register_count
            })

        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error fetching user auth counts by day: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user auth counts: {str(e)}")


@router.get("/admin/stories")
@limiter.limit("30/minute")
async def get_admin_stories(
    request: Request,
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    format_type: Optional[str] = Query(None),
    created_from: Optional[str] = Query(None),
    created_to: Optional[str] = Query(None),
):
    """Get platform-wide admin story summaries."""
    supabase = get_supabase_client()

    try:
        created_from_dt = _safe_parse_datetime(created_from)
        created_to_dt = _safe_parse_datetime(created_to, end_of_day=True)
        context = _fetch_user_admin_context(supabase)

        stories_response = supabase.table("stories").select("*").order("created_at", desc=True).execute()
        stories = stories_response.data or []

        characters_response = supabase.table("characters").select(
            "id,character_name,original_image_url,enhanced_images"
        ).execute()
        characters = characters_response.data or []
        characters_by_id = {str(row.get("id")): row for row in characters if row.get("id") is not None}

        jobs_response = (
            supabase
            .table("book_generation_jobs")
            .select("id,book_id,job_type,status,created_at,started_at,completed_at,error_message")
            .order("created_at", desc=True)
            .execute()
        )
        jobs_by_book_id = _group_jobs_by_book_id(jobs_response.data or [])

        users_by_id = {
            str(row.get("id")): row
            for row in context["users"]
            if row.get("id") is not None
        }
        child_profiles_by_id = {
            str(row.get("id")): row
            for row in context["child_profiles"]
            if row.get("id") is not None
        }

        summaries = [
            _build_admin_story_summary(
                story_row=story_row,
                users_by_id=users_by_id,
                child_parent_by_id=context["child_parent_by_id"],
                child_profiles_by_id=child_profiles_by_id,
                characters_by_id=characters_by_id,
                jobs_by_book_id=jobs_by_book_id,
            )
            for story_row in stories
        ]

        filtered = _filter_admin_story_summaries(
            summaries=summaries,
            search=search,
            status=status,
            format_type=format_type,
            created_from=created_from_dt,
            created_to=created_to_dt,
        )

        return {
            "success": True,
            "data": filtered,
            "meta": {
                "total": len(filtered),
                "filters": {
                    "search": search,
                    "status": status,
                    "format_type": format_type,
                    "created_from": created_from,
                    "created_to": created_to,
                },
            },
        }
    except Exception as e:
        logger.error(f"❌ Error fetching admin stories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch admin stories: {str(e)}")


@router.get("/admin/stories/{story_id}")
@limiter.limit("30/minute")
async def get_admin_story_detail(request: Request, story_id: str):
    """Get a single story detail payload for the admin story modal."""
    supabase = get_supabase_client()

    try:
        story_response = supabase.table("stories").select("*").eq("uid", story_id).execute()
        story_row = story_response.data[0] if story_response.data else None

        if story_row is None:
            try:
                numeric_story_id = int(story_id)
            except ValueError:
                numeric_story_id = None

            if numeric_story_id is not None:
                numeric_story_response = supabase.table("stories").select("*").eq("id", numeric_story_id).execute()
                story_row = numeric_story_response.data[0] if numeric_story_response.data else None

        if story_row is None:
            raise HTTPException(status_code=404, detail="Story not found")

        context = _fetch_user_admin_context(supabase)
        users_by_id = {
            str(row.get("id")): row
            for row in context["users"]
            if row.get("id") is not None
        }
        child_profiles_by_id = {
            str(row.get("id")): row
            for row in context["child_profiles"]
            if row.get("id") is not None
        }

        character = None
        character_id = story_row.get("character_id")
        if character_id is not None:
            character_response = (
                supabase
                .table("characters")
                .select("id,character_name,original_image_url,enhanced_images,created_at")
                .eq("id", character_id)
                .execute()
            )
            character = character_response.data[0] if character_response.data else None

        job_rows_response = (
            supabase
            .table("book_generation_jobs")
            .select("*")
            .eq("book_id", story_row.get("id"))
            .order("created_at", desc=True)
            .execute()
        )
        job_rows = job_rows_response.data or []
        jobs_by_book_id = _group_jobs_by_book_id(job_rows)
        latest_job = _pick_latest_story_job(story_row, jobs_by_book_id)

        owner_summary = _build_story_owner_summary(
            story_row=story_row,
            users_by_id=users_by_id,
            child_parent_by_id=context["child_parent_by_id"],
            child_profiles_by_id=child_profiles_by_id,
        )
        story_format = _effective_story_format(story_row, latest_job)
        story_page_texts = _extract_story_page_texts(story_row)

        return {
            "success": True,
            "data": {
                "id": str(story_row.get("id")) if story_row.get("id") is not None else "",
                "uid": story_row.get("uid"),
                "story_title": story_row.get("story_title") or "Untitled story",
                "character_name": (
                    story_row.get("character_name")
                    or (character or {}).get("character_name")
                    or "Untitled character"
                ),
                "status": _effective_story_status(story_row, latest_job),
                "format": story_format,
                "created_at": story_row.get("created_at"),
                "generation_duration_seconds": _calculate_job_duration_seconds(latest_job),
                "owner": {
                    "user_id": owner_summary["user_id"],
                    "email": owner_summary["user_email"],
                    "name": owner_summary["user_name"],
                },
                "child_profile": (
                    child_profiles_by_id.get(str(story_row.get("child_profile_id")))
                    if story_row.get("child_profile_id") is not None
                    else None
                ),
                "character": {
                    "id": str((character or {}).get("id")) if (character or {}).get("id") is not None else None,
                    "character_name": (
                        (character or {}).get("character_name")
                        or story_row.get("character_name")
                    ),
                    "original_image_url": (
                        (character or {}).get("original_image_url")
                        or story_row.get("original_image_url")
                    ),
                    "enhanced_images": (
                        (character or {}).get("enhanced_images")
                        or story_row.get("enhanced_images")
                        or []
                    ),
                },
                "cover_image": story_row.get("story_cover"),
                "pages": _build_story_pages(
                    story_row=story_row,
                    story_format=story_format,
                    story_page_texts=story_page_texts,
                ),
                "story_pages_text": story_page_texts,
                "jobs": job_rows,
                "raw_story": story_row,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching admin story detail {story_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch admin story detail: {str(e)}")


@router.get("/admin/templates")
@limiter.limit("30/minute")
async def get_templates(
    request: Request,
    story_format: Optional[str] = Query(
        None,
        description=(
            "Filter templates: 'adventure_story' (includes rows with null story_format), "
            "'interactive_story', or omit for all templates."
        ),
    ),
):
    """Get book templates, optionally filtered by story_format."""
    supabase = get_supabase_client()

    try:
        query = supabase.table("book_templates").select("*")
        fmt = normalize_book_template_story_format(story_format)
        if fmt is not None:
            if fmt not in VALID_BOOK_TEMPLATE_STORY_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid story_format query. Use 'adventure_story', 'interactive_story', "
                        "or omit the parameter."
                    ),
                )
            if fmt == "adventure_story":
                # Legacy rows have NULL story_format; treat them as adventure.
                query = query.or_("story_format.eq.adventure_story,story_format.is.null")
            else:
                query = query.eq("story_format", "interactive_story")

        response = query.order("created_at", desc=True).execute()

        return {
            "success": True,
            "data": response.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch templates: {str(e)}")


@router.get("/templates/random")
@limiter.limit("60/minute")
async def get_random_template_by_story_world(
    request: Request,
    story_world: str = Query(..., description="Story world: forest, underwater, or outerspace"),
    story_style: Optional[str] = Query(
        None,
        description="Optional story style to match exactly (e.g. cartoon, anime, 3d)"
    ),
    story_format: Optional[str] = Query(
        None,
        description=(
            "Optional story format filter: 'adventure_story' (includes null rows) "
            "or 'interactive_story'."
        ),
    ),
    for_dedication: bool = Query(
        False,
        description=(
            "When true, only templates with dedication_page_image set are returned "
            "(e.g. dedication page preview)."
        ),
    ),
):
    """Get one random template by story world (public endpoint for cover generation)."""
    supabase = get_supabase_client()

    valid_story_worlds = ["forest", "underwater", "outerspace"]
    normalized_world = (story_world or "").strip().lower()
    if normalized_world not in valid_story_worlds:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid story_world. Must be one of: {', '.join(valid_story_worlds)}"
        )

    try:
        normalized_format = normalize_book_template_story_format(story_format)
        if normalized_format and normalized_format not in VALID_BOOK_TEMPLATE_STORY_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid story_format query. Use 'adventure_story', 'interactive_story', "
                    "or omit it."
                ),
            )

        query = (
            supabase
            .table("book_templates")
            .select("*")
            .eq("story_world", normalized_world)
        )
        if for_dedication:
            query = query.not_.is_("dedication_page_image", "null")
        else:
            query = query.not_.is_("cover_image", "null")
        normalized_style = normalize_story_style(story_style)
        if normalized_style:
            query = query.eq("story_style", normalized_style)
        if normalized_format == "adventure_story":
            # Legacy rows with null story_format should still be considered adventure templates.
            query = query.or_("story_format.eq.adventure_story,story_format.is.null")
        elif normalized_format == "interactive_story":
            query = query.eq("story_format", "interactive_story")

        response = query.execute()

        templates = response.data or []
        if len(templates) == 0:
            scope = "dedication page" if for_dedication else "cover"
            if normalized_style:
                format_scope = (
                    f" and story format '{normalized_format}'" if normalized_format else ""
                )
                return {
                    "success": False,
                    "error": (
                        f"No templates found for {scope} with "
                        f"story world '{normalized_world}' and story style '{normalized_style}'{format_scope}"
                    )
                }
            if normalized_format:
                return {
                    "success": False,
                    "error": (
                        f"No templates found for {scope} with "
                        f"story world '{normalized_world}' and story format '{normalized_format}'"
                    ),
                }
            return {
                "success": False,
                "error": (
                    f"No templates found for {scope} with story world: {normalized_world}"
                ),
            }

        return {
            "success": True,
            "data": random.choice(templates)
        }
    except Exception as e:
        logger.error(f"❌ Error fetching random template for {normalized_world}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch random template: {str(e)}")


@router.get("/admin/users")
@limiter.limit("30/minute")
async def get_users(
    request: Request,
    search: Optional[str] = Query(None),
    account_type: Optional[str] = Query(None),
    subscription_status: Optional[str] = Query(None),
    registered_from: Optional[str] = Query(None),
    registered_to: Optional[str] = Query(None),
    story_count_min: Optional[int] = Query(None, ge=0),
    story_count_max: Optional[int] = Query(None, ge=0),
):
    """Get users list for admin user management table"""
    supabase = get_supabase_client()

    try:
        registered_from_dt = _safe_parse_datetime(registered_from)
        registered_to_dt = _safe_parse_datetime(registered_to, end_of_day=True)
        context = _fetch_user_admin_context(supabase)
        summaries = [
            _build_user_summary(
                user_row=row,
                story_count_by_user=context["story_count_by_user"],
                latest_login_by_user=context["latest_login_by_user"],
                latest_subscription_by_user=context["latest_subscription_by_user"],
                purchase_summary_by_user=context["purchase_summary_by_user"],
                child_count_by_user=context["child_count_by_user"],
            )
            for row in context["users"]
        ]

        filtered = _filter_user_summaries(
            summaries=summaries,
            search=search,
            account_type=account_type,
            subscription_status=subscription_status,
            registered_from=registered_from_dt,
            registered_to=registered_to_dt,
            story_count_min=story_count_min,
            story_count_max=story_count_max,
        )

        return {
            "success": True,
            "data": filtered,
            "meta": {
                "total": len(filtered),
                "filters": {
                    "search": search,
                    "account_type": account_type,
                    "subscription_status": subscription_status,
                    "registered_from": registered_from,
                    "registered_to": registered_to,
                    "story_count_min": story_count_min,
                    "story_count_max": story_count_max,
                },
            },
        }
    except Exception as e:
        logger.error(f"❌ Error fetching users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")


@router.get("/admin/users/{user_id}")
@limiter.limit("30/minute")
async def get_user_detail(request: Request, user_id: str):
    """Get detailed admin profile for a single user."""
    supabase = get_supabase_client()

    try:
        context = _fetch_user_admin_context(supabase)
        user_row = next((row for row in context["users"] if str(row.get("id")) == user_id), None)
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")

        summary = _build_user_summary(
            user_row=user_row,
            story_count_by_user=context["story_count_by_user"],
            latest_login_by_user=context["latest_login_by_user"],
            latest_subscription_by_user=context["latest_subscription_by_user"],
            purchase_summary_by_user=context["purchase_summary_by_user"],
            child_count_by_user=context["child_count_by_user"],
        )

        user_stories = [
            row
            for row in context["stories"]
            if _resolve_story_owner_user_id(row, context["child_parent_by_id"]) == user_id
        ]
        user_stories.sort(key=lambda row: _safe_parse_datetime(row.get("created_at")) or datetime.min, reverse=True)

        child_profiles = [row for row in context["child_profiles"] if str(row.get("parent_id")) == user_id]
        child_profiles.sort(key=lambda row: _safe_parse_datetime(row.get("created_at")) or datetime.min, reverse=True)

        characters_response = (
            supabase
            .table("characters")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        payment_history_response = (
            supabase
            .table("book_purchases")
            .select("*")
            .eq("user_id", user_id)
            .order("purchase_date", desc=True)
            .execute()
        )
        generation_history_response = (
            supabase
            .table("book_generation_jobs")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        subscriptions_response = (
            supabase
            .table("subscriptions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        auth_history_response = (
            supabase
            .table("user_auth_history")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "success": True,
            "data": {
                "account_information": {
                    **summary,
                    "stripe_customer_id": user_row.get("stripe_customer_id"),
                    "children": child_profiles,
                    "subscriptions": subscriptions_response.data or [],
                    "auth_history": auth_history_response.data or [],
                },
                "characters": characters_response.data or [],
                "story_library": user_stories,
                "payment_history": payment_history_response.data or [],
                "generation_history": generation_history_response.data or [],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching user detail {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user detail: {str(e)}")


@router.post("/admin/users")
@limiter.limit("10/minute")
async def create_user(request: Request, body: AdminUserCreate):
    """Create user record from admin panel"""
    supabase = get_supabase_client()

    email = (body.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    try:
        existing = (
            supabase
            .table("users")
            .select("id")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="User with this email already exists")

        insert_data = {
            "id": str(uuid4()),
            "email": email,
            "first_name": (body.first_name or "").strip() or None,
            "last_name": (body.last_name or "").strip() or None,
            "role": (body.role or "user").strip() or "user",
            "subscription_status": (body.subscription_status or "").strip() or None,
            "credit": max(0, body.credit or 0),
        }

        response = supabase.table("users").insert(insert_data).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create user")

        return {"success": True, "data": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")


@router.patch("/admin/users/{user_id}")
@limiter.limit("20/minute")
async def update_user(request: Request, user_id: str, body: AdminUserUpdate):
    """Update user record from admin panel"""
    supabase = get_supabase_client()

    try:
        update_data: Dict[str, Any] = {}
        if body.email is not None:
            update_data["email"] = body.email.strip().lower()
        if body.first_name is not None:
            update_data["first_name"] = body.first_name.strip() or None
        if body.last_name is not None:
            update_data["last_name"] = body.last_name.strip() or None
        if body.role is not None:
            update_data["role"] = body.role.strip() or "user"
        if body.subscription_status is not None:
            update_data["subscription_status"] = body.subscription_status.strip() or None
        if body.credit is not None:
            update_data["credit"] = max(0, body.credit)

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        response = supabase.table("users").update(update_data).eq("id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="User not found")

        return {"success": True, "data": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")


@router.delete("/admin/users/{user_id}")
@limiter.limit("10/minute")
async def delete_user(request: Request, user_id: str):
    """Delete user and all related data/storage assets from admin panel"""
    supabase = get_supabase_client()

    try:
        # Verify existence for cleaner error messages
        existing = supabase.table("users").select("id,email").eq("id", user_id).single().execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="User not found")

        user_email = existing.data.get("email")

        # 1) Collect child profiles for this parent
        child_profiles = (
            supabase.table("child_profiles")
            .select("id,avatar_url")
            .eq("parent_id", user_id)
            .execute()
        )
        child_rows = child_profiles.data or []
        child_ids = [row.get("id") for row in child_rows if row.get("id") is not None]
        child_avatar_urls = [row.get("avatar_url") for row in child_rows if row.get("avatar_url")]

        # 2) Collect characters for this user
        characters_response = (
            supabase.table("characters")
            .select("id,original_image_url,enhanced_images")
            .eq("user_id", user_id)
            .execute()
        )
        character_rows = characters_response.data or []
        character_ids = [row.get("id") for row in character_rows if row.get("id") is not None]

        # 3) Collect stories related to this user via user/child/character relations
        story_map: Dict[Any, Dict[str, Any]] = {}

        user_stories = (
            supabase.table("stories")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        for row in (user_stories.data or []):
            story_map[row.get("id")] = row

        if child_ids:
            child_stories = (
                supabase.table("stories")
                .select("*")
                .in_("child_profile_id", child_ids)
                .execute()
            )
            for row in (child_stories.data or []):
                story_map[row.get("id")] = row

        if character_ids:
            character_stories = (
                supabase.table("stories")
                .select("*")
                .in_("character_id", character_ids)
                .execute()
            )
            for row in (character_stories.data or []):
                story_map[row.get("id")] = row

        story_rows = list(story_map.values())
        story_ids = [row.get("id") for row in story_rows if row.get("id") is not None]

        # 4) Delete related storage files (best-effort)
        storage_files_deleted = 0
        storage_files_failed = 0

        protected_template_urls: Set[str] = set()
        try:
            protected_template_urls = collect_book_template_image_urls(supabase)
        except Exception as template_lookup_error:
            logger.warning(
                f"⚠️ Could not load shared template URLs for user-delete cleanup protection: {template_lookup_error}"
            )

        for story in story_rows:
            try:
                # Full cleanup for admin delete: include character/enhanced images too.
                deletion_result = delete_story_images(
                    supabase,
                    story,
                    exclude_character_images=False,
                    protected_urls=protected_template_urls,
                )
                storage_files_deleted += deletion_result.get("success", 0)
                storage_files_failed += deletion_result.get("errors", 0)
            except Exception as e:
                logger.warning(f"⚠️ Story storage cleanup failed for story_id={story.get('id')}: {e}")

        for character in character_rows:
            try:
                deletion_result = delete_character_images(supabase, character)
                storage_files_deleted += deletion_result.get("success", 0)
                storage_files_failed += deletion_result.get("errors", 0)
            except Exception as e:
                logger.warning(f"⚠️ Character storage cleanup failed for character_id={character.get('id')}: {e}")

        if child_avatar_urls:
            try:
                child_avatar_cleanup = delete_files_from_storage(supabase, child_avatar_urls)
                storage_files_deleted += child_avatar_cleanup.get("success", 0)
                storage_files_failed += child_avatar_cleanup.get("errors", 0)
            except Exception as e:
                logger.warning(f"⚠️ Child avatar cleanup failed for user_id={user_id}: {e}")

        # 5) Delete related rows from all known tables (best-effort per table)
        deleted_counts: Dict[str, int] = {}

        # Gifts and user activity
        deleted_counts["gifts_by_user_id"] = _safe_delete_eq(supabase, "gifts", "user_id", user_id)
        deleted_counts["gifts_by_from_user_id"] = _safe_delete_eq(supabase, "gifts", "from_user_id", user_id)
        deleted_counts["gifts_by_to_user_id"] = _safe_delete_eq(supabase, "gifts", "to_user_id", user_id)
        deleted_counts["user_auth_history"] = _safe_delete_eq(supabase, "user_auth_history", "user_id", user_id)
        deleted_counts["push_subscriptions"] = _safe_delete_eq(supabase, "push_subscriptions", "user_id", user_id)
        deleted_counts["subscriptions"] = _safe_delete_eq(supabase, "subscriptions", "user_id", user_id)
        deleted_counts["book_generation_jobs_by_user"] = _safe_delete_eq(supabase, "book_generation_jobs", "user_id", user_id)
        deleted_counts["book_purchases_by_user"] = _safe_delete_eq(supabase, "book_purchases", "user_id", user_id)
        deleted_counts["search_game_results_by_user"] = _safe_delete_eq(supabase, "search_game_results", "user_id", user_id)

        # Story-linked records
        deleted_counts["stories"] = _safe_delete_in(supabase, "stories", "id", story_ids)
        deleted_counts["book_purchases_by_story"] = _safe_delete_in(supabase, "book_purchases", "story_id", story_ids)
        deleted_counts["search_game_results_by_story"] = _safe_delete_in(supabase, "search_game_results", "story_id", story_ids)
        deleted_counts["gifts_by_story"] = _safe_delete_in(supabase, "gifts", "story_id", story_ids)

        # Character-linked and child-linked records
        deleted_counts["characters"] = _safe_delete_in(supabase, "characters", "id", character_ids)
        deleted_counts["search_game_results_by_character"] = _safe_delete_in(supabase, "search_game_results", "character_id", character_ids)
        deleted_counts["child_profiles"] = _safe_delete_in(supabase, "child_profiles", "id", child_ids)
        deleted_counts["search_game_results_by_child"] = _safe_delete_in(supabase, "search_game_results", "child_profile_id", child_ids)
        deleted_counts["book_generation_jobs_by_child"] = _safe_delete_in(supabase, "book_generation_jobs", "child_profile_id", child_ids)
        deleted_counts["gifts_by_child"] = _safe_delete_in(supabase, "gifts", "child_profile_id", child_ids)

        # 6) Delete user row in custom users table
        deleted_counts["users"] = _safe_delete_eq(supabase, "users", "id", user_id)
        if deleted_counts["users"] == 0:
            # Safety: if row wasn't deleted, return error because this is primary target.
            raise HTTPException(status_code=500, detail="Failed to delete user row from users table")

        # 7) Try deleting auth user as final step (best-effort)
        auth_user_deleted = False
        try:
            # Supabase admin SDKs differ by method name across versions.
            if hasattr(supabase.auth.admin, "delete_user"):
                supabase.auth.admin.delete_user(user_id)
                auth_user_deleted = True
            elif hasattr(supabase.auth.admin, "deleteUser"):
                supabase.auth.admin.deleteUser(user_id)
                auth_user_deleted = True
        except Exception as e:
            logger.warning(f"⚠️ Could not delete auth user {user_id}: {e}")

        return {
            "success": True,
            "message": "User and related data deleted successfully",
            "data": {
                "user_id": user_id,
                "email": user_email,
                "auth_user_deleted": auth_user_deleted,
                "related_story_count": len(story_ids),
                "related_character_count": len(character_ids),
                "related_child_profile_count": len(child_ids),
                "storage_cleanup": {
                    "files_deleted": storage_files_deleted,
                    "files_failed": storage_files_failed
                },
                "deleted_counts": deleted_counts
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")


@router.post("/admin/templates")
@limiter.limit("10/minute")
async def create_template(request: Request, body: BookTemplateCreate):
    """Create a new book template"""
    supabase = get_supabase_client()
    
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="Template name is required")
    
    # Validate story_world if provided
    valid_story_worlds = ['forest', 'underwater', 'outerspace']
    if body.story_world and body.story_world not in valid_story_worlds:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid story_world. Must be one of: {', '.join(valid_story_worlds)}"
        )

    # Validate story style/type if provided
    requested_story_style = body.story_type if body.story_type is not None else body.story_style
    requested_story_style = normalize_story_style(requested_story_style)
    if requested_story_style and requested_story_style not in VALID_TEMPLATE_STORY_STYLES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid story_style. Must be one of: "
                f"{', '.join(sorted(VALID_TEMPLATE_STORY_STYLES))}"
            )
        )
    
    try:
        insert_data = {"name": body.name.strip()}
        if body.story_world:
            insert_data["story_world"] = body.story_world
        if requested_story_style:
            insert_data["story_style"] = requested_story_style

        raw_format = normalize_book_template_story_format(body.story_format)
        if not raw_format:
            raw_format = "adventure_story"
        if raw_format not in VALID_BOOK_TEMPLATE_STORY_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid story_format. Must be one of: "
                    f"{', '.join(sorted(VALID_BOOK_TEMPLATE_STORY_FORMATS))}"
                ),
            )
        insert_data["story_format"] = raw_format

        response = supabase.table("book_templates").insert(insert_data).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create template")
        
        logger.info(
            f"✅ Created template: {body.name} "
            f"(story_world: {body.story_world or 'none'}, story_style: {requested_story_style or 'none'}, "
            f"story_format: {raw_format})"
        )
        
        return {
            "success": True,
            "data": response.data[0]
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@router.delete("/admin/templates/{template_id}")
@limiter.limit("10/minute")
async def delete_template(
    request: Request,
    template_id: str,
    story_format: Optional[str] = Query(
        None,
        description=(
            "When set, delete only if the template's story_format matches "
            "(null counts as adventure_story). Prevents deleting the wrong list in the admin UI."
        ),
    ),
):
    """Delete a book template and all associated images from storage"""
    supabase = get_supabase_client()

    try:
        # Load template and delete only assets referenced by this row.
        # This prevents cross-template deletion when names are duplicated.
        response = (
            supabase
            .table("book_templates")
            .select(
                "name,story_format,cover_image,copyright_page_image,dedication_page_image,"
                "story_page_images,last_words_page_image,last_story_page_image,back_cover_image"
            )
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        expected_fmt = normalize_book_template_story_format(story_format)
        if expected_fmt is not None:
            if expected_fmt not in VALID_BOOK_TEMPLATE_STORY_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid story_format query for delete.",
                )
            actual = effective_template_story_format(response.data.get("story_format"))
            if actual != expected_fmt:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Template story_format does not match the requested filter. "
                        "Refresh the list and try again."
                    ),
                )
        
        template_data = response.data
        template_name = template_data["name"]
        urls_to_delete = [
            template_data.get("cover_image"),
            template_data.get("copyright_page_image"),
            template_data.get("dedication_page_image"),
            template_data.get("last_words_page_image"),
            template_data.get("last_story_page_image"),
            template_data.get("back_cover_image"),
        ]
        story_page_images = template_data.get("story_page_images") or []
        if isinstance(story_page_images, list):
            urls_to_delete.extend(story_page_images)

        _delete_urls_or_raise(supabase, urls_to_delete, f"template {template_id}")
        
        # Delete template from database
        delete_response = supabase.table("book_templates").delete().eq("id", template_id).execute()
        
        logger.info(f"✅ Deleted template: {template_name} (ID: {template_id})")
        
        return {
            "success": True,
            "message": f"Template '{template_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-image")
@limiter.limit("20/minute")
async def upload_template_image(
    request: Request,
    template_id: str,
    file: UploadFile = File(...),
    field_key: str = Form(...),
):
    """
    Upload a single image for a book template field.
    Images are automatically optimized to WebP format before upload to save storage space.
    
    Args:
        template_id: ID of the template
        file: Image file to upload (will be optimized to WebP)
        field_key: Database field name (cover_image, back_cover_image)
    
    Returns:
        JSON with success status, updated template data, and optimized image URL
    """
    supabase = get_supabase_client()
    
    # Validate field_key
    valid_fields = [
        "cover_image",
        "copyright_page_image",
        "dedication_page_image",
        "back_cover_image",
        "last_words_page_image",
        "last_story_page_image",
    ]
    if field_key not in valid_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field_key. Must be one of: {', '.join(valid_fields)}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Use template UUID in storage path to avoid collisions.
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_path = f"book-templates/{template_id}/{field_key}.{file_ext}"
        
        # Upload to storage (will be optimized to WebP automatically)
        public_url = await upload_to_storage(file, "book-images", file_path)
        
        # Update database
        update_data = {field_key: public_url}
        response = supabase.table("book_templates").update(update_data).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")
        
        logger.info(f"✅ Uploaded {field_key} for template ID: {template_id}")
        
        return {
            "success": True,
            "data": response.data[0],
            "image_url": public_url,
            "optimized": True,
            "format": "WebP"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-story-page")
@limiter.limit("30/minute")
async def upload_single_story_page(
    request: Request,
    template_id: str,
    file: UploadFile = File(...),
    page_index: int = Form(...)  # Index position for this page (0-based)
):
    """
    Upload a single story page image for a book template.
    Images are automatically optimized to WebP format before upload.
    This endpoint should be called multiple times (once per image) to avoid 413 errors.
    
    Args:
        template_id: ID of the template
        file: Single image file to upload (will be optimized to WebP)
        page_index: Index position for this page in the story_page_images array (0-based)
    
    Returns:
        JSON with success status, updated template data, and image URL
    """
    supabase = get_supabase_client()
    
    # Validate file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")
    
    try:
        # Get current template to retrieve existing story pages
        template_response = supabase.table("book_templates").select("story_page_images").eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        existing_urls = template_response.data.get("story_page_images") or []
        
        # Use template UUID in storage path to avoid collisions.
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_path = f"book-templates/{template_id}/story-page-{page_index + 1}.{file_ext}"
        
        # Upload to storage (with automatic optimization)
        public_url = await upload_to_storage(file, "book-images", file_path)
        
        # Insert or update the URL at the specified index
        if page_index < len(existing_urls):
            # Update existing position
            existing_urls[page_index] = public_url
        else:
            # Append to the end (fill gaps if needed)
            while len(existing_urls) < page_index:
                existing_urls.append(None)  # Placeholder for gaps
            existing_urls.append(public_url)
        
        # Remove any None placeholders
        existing_urls = [url for url in existing_urls if url is not None]
        
        # Update database with new array
        response = supabase.table("book_templates").update({
            "story_page_images": existing_urls
        }).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")
        
        logger.info(f"✅ Uploaded story page {page_index + 1} for template ID: {template_id}")
        
        return {
            "success": True,
            "data": response.data[0],
            "image_url": public_url,
            "page_index": page_index,
            "total_pages": len(existing_urls),
            "optimized": True,
            "format": "WebP"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading story page: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload story page: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-main-character-image")
@limiter.limit("30/minute")
async def upload_main_character_image(
    request: Request,
    template_id: str,
    file: UploadFile = File(...),
    image_index: int = Form(...)
):
    """
    Upload a single main character image for a book template.
    Stored in book_templates.main_character_images (TEXT[]), indexed 0-based.
    """
    supabase = get_supabase_client()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")

    try:
        template_response = (
            supabase
            .table("book_templates")
            .select("main_character_images")
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        existing_urls = template_response.data.get("main_character_images") or []

        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_path = f"book-templates/{template_id}/main-character-{image_index + 1}.{file_ext}"

        public_url = await upload_to_storage(file, "book-images", file_path)

        if image_index < len(existing_urls):
            existing_urls[image_index] = public_url
        else:
            while len(existing_urls) < image_index:
                existing_urls.append(None)
            existing_urls.append(public_url)

        existing_urls = [url for url in existing_urls if url is not None]

        response = (
            supabase
            .table("book_templates")
            .update({"main_character_images": existing_urls})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")

        logger.info(f"✅ Uploaded main character image {image_index + 1} for template ID: {template_id}")

        return {
            "success": True,
            "data": response.data[0],
            "image_url": public_url,
            "image_index": image_index,
            "total_images": len(existing_urls),
            "optimized": True,
            "format": "WebP"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading main character image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload main character image: {str(e)}")


@router.delete("/admin/templates/{template_id}/main-character-image/{image_index}")
@limiter.limit("30/minute")
async def delete_main_character_image(
    request: Request,
    template_id: str,
    image_index: int
):
    """Delete one main character image by index and remove from storage + DB array."""
    supabase = get_supabase_client()

    try:
        if image_index < 0:
            raise HTTPException(status_code=400, detail="image_index must be >= 0")

        template_response = (
            supabase
            .table("book_templates")
            .select("main_character_images")
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        urls = template_response.data.get("main_character_images") or []
        if not isinstance(urls, list):
            urls = []
        if image_index >= len(urls):
            raise HTTPException(status_code=400, detail="image_index out of range")

        url_to_delete = urls[image_index]

        _delete_urls_or_raise(
            supabase,
            [url_to_delete] if url_to_delete else [],
            f"main character image index {image_index}"
        )

        new_urls = [u for idx, u in enumerate(urls) if idx != image_index]

        response = (
            supabase
            .table("book_templates")
            .update({"main_character_images": new_urls})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")

        return {"success": True, "data": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting main character image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete main character image: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-character-for-finding-image")
@limiter.limit("30/minute")
async def upload_character_for_finding_image(
    request: Request,
    template_id: str,
    file: UploadFile = File(...),
    image_index: int = Form(...)
):
    """
    Upload a single character-for-finding image for a book template.
    Stored in book_templates.character_for_finding (TEXT[]), indexed 0-based.
    """
    supabase = get_supabase_client()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")

    try:
        template_response = (
            supabase
            .table("book_templates")
            .select("character_for_finding")
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        existing_urls = template_response.data.get("character_for_finding") or []

        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_path = f"book-templates/{template_id}/character-for-finding-{image_index + 1}.{file_ext}"

        public_url = await upload_to_storage(file, "book-images", file_path)

        if image_index < len(existing_urls):
            existing_urls[image_index] = public_url
        else:
            while len(existing_urls) < image_index:
                existing_urls.append(None)
            existing_urls.append(public_url)

        existing_urls = [url for url in existing_urls if url is not None]

        response = (
            supabase
            .table("book_templates")
            .update({"character_for_finding": existing_urls})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")

        logger.info(f"✅ Uploaded character-for-finding image {image_index + 1} for template ID: {template_id}")

        return {
            "success": True,
            "data": response.data[0],
            "image_url": public_url,
            "image_index": image_index,
            "total_images": len(existing_urls),
            "optimized": True,
            "format": "WebP"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading character-for-finding image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload character-for-finding image: {str(e)}")


@router.delete("/admin/templates/{template_id}/character-for-finding-image/{image_index}")
@limiter.limit("30/minute")
async def delete_character_for_finding_image(
    request: Request,
    template_id: str,
    image_index: int
):
    """Delete one character-for-finding image by index and remove from storage + DB array."""
    supabase = get_supabase_client()

    try:
        if image_index < 0:
            raise HTTPException(status_code=400, detail="image_index must be >= 0")

        template_response = (
            supabase
            .table("book_templates")
            .select("character_for_finding")
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        urls = template_response.data.get("character_for_finding") or []
        if not isinstance(urls, list):
            urls = []
        if image_index >= len(urls):
            raise HTTPException(status_code=400, detail="image_index out of range")

        url_to_delete = urls[image_index]

        _delete_urls_or_raise(
            supabase,
            [url_to_delete] if url_to_delete else [],
            f"character-for-finding image index {image_index}"
        )

        new_urls = [u for idx, u in enumerate(urls) if idx != image_index]

        response = (
            supabase
            .table("book_templates")
            .update({"character_for_finding": new_urls})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")

        return {"success": True, "data": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting character-for-finding image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete character-for-finding image: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-story-pages")
@limiter.limit("20/minute")
async def upload_story_pages(
    request: Request,
    template_id: str,
    files: List[UploadFile] = File(...),
    existing_images: str = Form(default="[]")  # JSON string of existing image URLs
):
    """
    Upload multiple story page images for a book template (DEPRECATED - use upload-story-page instead).
    All images are automatically optimized to WebP format before upload to save storage space.
    
    NOTE: This endpoint may cause 413 errors with many/large files. 
    Use POST /upload-story-page endpoint instead to upload one image at a time.
    
    Args:
        template_id: ID of the template
        files: List of image files to upload (will be optimized to WebP)
        existing_images: JSON string array of existing image URLs to preserve
    
    Returns:
        JSON with success status, updated template data, upload count, and optimization info
    """
    supabase = get_supabase_client()
    
    try:
        import json
        existing_urls = json.loads(existing_images) if existing_images else []
        
        if not isinstance(existing_urls, list):
            raise HTTPException(status_code=400, detail="existing_images must be a JSON array")
        
        # Validate all files are images
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")
        
        # Use template UUID in storage path to avoid collisions.
        new_urls = []
        
        for idx, file in enumerate(files):
            current_index = len(existing_urls) + idx
            file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            file_path = f"book-templates/{template_id}/story-page-{current_index + 1}.{file_ext}"
            
            # Upload to storage (with automatic optimization)
            public_url = await upload_to_storage(file, "book-images", file_path)
            new_urls.append(public_url)
        
        # Combine existing and new URLs
        all_urls = existing_urls + new_urls
        
        # Update database
        response = supabase.table("book_templates").update({
            "story_page_images": all_urls
        }).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")
        
        logger.info(f"✅ Uploaded {len(new_urls)} story pages for template ID: {template_id}")
        
        return {
            "success": True,
            "data": response.data[0],
            "uploaded_count": len(new_urls),
            "total_count": len(all_urls),
            "optimized": True,
            "format": "WebP"
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid existing_images JSON format")
    except Exception as e:
        logger.error(f"❌ Error uploading story pages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload story pages: {str(e)}")


@router.delete("/admin/templates/{template_id}/image")
@limiter.limit("30/minute")
async def delete_template_image(
    request: Request,
    template_id: str,
    field_key: str = Query(...)
):
    """Delete a single template image from storage and clear its DB field."""
    supabase = get_supabase_client()

    valid_fields = [
        "cover_image",
        "copyright_page_image",
        "dedication_page_image",
        "back_cover_image",
        "last_words_page_image",
        "last_story_page_image",
    ]
    if field_key not in valid_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field_key. Must be one of: {', '.join(valid_fields)}"
        )

    try:
        template_response = (
            supabase
            .table("book_templates")
            .select(field_key)
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        existing_url = template_response.data.get(field_key)
        _delete_urls_or_raise(
            supabase,
            [existing_url] if existing_url else [],
            f"template field '{field_key}'"
        )

        response = (
            supabase
            .table("book_templates")
            .update({field_key: None})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Template not found")

        logger.info(f"✅ Deleted template image '{field_key}' for template {template_id}")
        return {
            "success": True,
            "data": response.data[0],
            "deleted_url": existing_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting template image '{field_key}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template image: {str(e)}")


@router.delete("/admin/templates/{template_id}/story-page/{page_index}")
@limiter.limit("30/minute")
async def delete_story_page_image(
    request: Request,
    template_id: str,
    page_index: int
):
    """Delete one story page image from storage and remove it from DB array."""
    supabase = get_supabase_client()

    if page_index < 0:
        raise HTTPException(status_code=400, detail="page_index must be >= 0")

    try:
        template_response = (
            supabase
            .table("book_templates")
            .select("story_page_images")
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        story_page_images = template_response.data.get("story_page_images") or []
        if not isinstance(story_page_images, list):
            story_page_images = []

        if page_index >= len(story_page_images):
            raise HTTPException(status_code=400, detail="Invalid story page index")

        removed_url = story_page_images[page_index]
        _delete_urls_or_raise(
            supabase,
            [removed_url] if removed_url else [],
            f"story page index {page_index}"
        )

        next_story_page_images = [url for idx, url in enumerate(story_page_images) if idx != page_index]
        response = (
            supabase
            .table("book_templates")
            .update({"story_page_images": next_story_page_images})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Template not found")

        logger.info(f"✅ Deleted story page {page_index + 1} for template {template_id}")
        return {
            "success": True,
            "data": response.data[0],
            "deleted_url": removed_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting story page {page_index} for template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete story page image: {str(e)}")


@router.patch("/admin/templates/{template_id}")
@limiter.limit("20/minute")
async def update_template(
    request: Request,
    template_id: str,
    body: BookTemplateUpdate
):
    """Update book template metadata (name, story_world, or image URLs)"""
    supabase = get_supabase_client()
    
    try:
        provided_fields = getattr(body, "__fields_set__", set())

        # Validate story_world if provided
        if "story_world" in provided_fields and body.story_world is not None:
            valid_story_worlds = ['forest', 'underwater', 'outerspace']
            # Empty string means clear the story_world
            if body.story_world and body.story_world not in valid_story_worlds:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid story_world. Must be one of: {', '.join(valid_story_worlds)}"
                )

        # Validate story style/type if provided
        story_type_field_provided = "story_type" in provided_fields
        story_style_field_provided = "story_style" in provided_fields
        requested_story_style = body.story_type if story_type_field_provided else body.story_style
        requested_story_style = normalize_story_style(requested_story_style)
        if (story_type_field_provided or story_style_field_provided) and requested_story_style is not None:
            # Empty string means clear the story_style
            if requested_story_style and requested_story_style not in VALID_TEMPLATE_STORY_STYLES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid story_style. Must be one of: "
                        f"{', '.join(sorted(VALID_TEMPLATE_STORY_STYLES))}"
                    )
                )
        
        # Build update data from non-None fields
        update_data = {}
        if "name" in provided_fields and body.name is not None:
            update_data["name"] = body.name.strip()
        if "story_world" in provided_fields:
            # Empty string or null means clear the field
            update_data["story_world"] = body.story_world if body.story_world else None
        if story_type_field_provided or story_style_field_provided:
            # Empty string or null means clear the field
            update_data["story_style"] = requested_story_style if requested_story_style else None
        if "story_format" in provided_fields:
            raw_sf = normalize_book_template_story_format(body.story_format)
            if raw_sf is None:
                update_data["story_format"] = None
            elif raw_sf not in VALID_BOOK_TEMPLATE_STORY_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid story_format. Must be one of: "
                        f"{', '.join(sorted(VALID_BOOK_TEMPLATE_STORY_FORMATS))}"
                    ),
                )
            else:
                update_data["story_format"] = raw_sf
        if "cover_image" in provided_fields:
            update_data["cover_image"] = body.cover_image
        if "story_page_images" in provided_fields:
            update_data["story_page_images"] = body.story_page_images
        if "character_for_finding" in provided_fields:
            update_data["character_for_finding"] = body.character_for_finding
        if "copyright_page_image" in provided_fields:
            update_data["copyright_page_image"] = body.copyright_page_image
        if "dedication_page_image" in provided_fields:
            update_data["dedication_page_image"] = body.dedication_page_image
        if "last_words_page_image" in provided_fields:
            update_data["last_words_page_image"] = body.last_words_page_image
        if "last_story_page_image" in provided_fields:
            update_data["last_story_page_image"] = body.last_story_page_image
        if "back_cover_image" in provided_fields:
            update_data["back_cover_image"] = body.back_cover_image
        # Handle positions update (validate structure)
        if "positions" in provided_fields:
            positions_val = body.positions
            if positions_val is None:
                # Clear positions
                update_data["positions"] = None
            else:
                if not isinstance(positions_val, list):
                    raise HTTPException(status_code=400, detail="positions must be a list of {x: float, y: float} objects")
                validated_positions = []
                if len(positions_val) > 16:
                    raise HTTPException(status_code=400, detail="positions can contain at most 16 coordinate objects")
                for idx, coord in enumerate(positions_val):
                    if not isinstance(coord, dict):
                        raise HTTPException(status_code=400, detail=f"positions[{idx}] must be an object with x and y floats")
                    if "x" not in coord or "y" not in coord:
                        raise HTTPException(status_code=400, detail=f"positions[{idx}] must contain keys 'x' and 'y'")
                    try:
                        x = float(coord.get("x"))
                        y = float(coord.get("y"))
                    except Exception:
                        raise HTTPException(status_code=400, detail=f"positions[{idx}].x and .y must be numbers")
                    # Optionally validate ranges 0.0-1.0
                    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                        raise HTTPException(status_code=400, detail=f"positions[{idx}] coordinates must be between 0.0 and 1.0")
                    validated_positions.append({"x": x, "y": y})
                update_data["positions"] = validated_positions

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Update database
        response = supabase.table("book_templates").update(update_data).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(f"✅ Updated template (ID: {template_id})")
        
        return {
            "success": True,
            "data": response.data[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")
