import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apis.admin import (
    _build_single_scene_image_update,
    _build_story_pages,
    _build_story_owner_summary,
    _build_story_count_by_user,
    _build_user_generation_history_from_stories,
    _filter_admin_story_summaries,
    _filter_user_summaries,
    _fetch_user_payment_history_safe,
    _listify_urls,
    _matches_date_range,
    _pick_latest_story_job,
    _safe_parse_datetime,
)


def test_registered_from_filter_handles_naive_ui_date_against_aware_created_at():
    registered_from = _safe_parse_datetime("2026-04-22")

    assert _matches_date_range(
        "2026-04-22T08:30:00+00:00",
        registered_from,
        None,
    )


def test_registered_to_filter_treats_date_input_as_end_of_day():
    registered_to = _safe_parse_datetime("2026-04-22", end_of_day=True)

    assert _matches_date_range(
        "2026-04-22T23:59:59+00:00",
        None,
        registered_to,
    )
    assert not _matches_date_range(
        "2026-04-23T00:00:00+00:00",
        None,
        registered_to,
    )


def test_story_count_filter_respects_min_and_max_bounds():
    summaries = [
        {"id": "user-1", "total_stories_created": 1},
        {"id": "user-2", "total_stories_created": 3},
        {"id": "user-3", "total_stories_created": 5},
    ]

    filtered = _filter_user_summaries(
        summaries=summaries,
        search=None,
        account_type=None,
        subscription_status=None,
        registered_from=None,
        registered_to=None,
        story_count_min=2,
        story_count_max=4,
    )

    assert [summary["id"] for summary in filtered] == ["user-2"]


def test_story_counts_include_child_profile_owned_stories():
    stories = [
        {"uid": "story-1", "user_id": "parent-1", "child_profile_id": None},
        {"uid": "story-2", "user_id": None, "child_profile_id": 101},
        {"uid": "story-3", "user_id": None, "child_profile_id": 101},
        {"uid": "story-4", "user_id": None, "child_profile_id": 202},
    ]
    child_parent_by_id = {
        "101": "parent-1",
        "202": "parent-2",
    }

    counts = _build_story_count_by_user(stories, child_parent_by_id)

    assert counts == {
        "parent-1": 3,
        "parent-2": 1,
    }


def test_story_counts_prefer_explicit_user_id_without_double_counting():
    stories = [
        {"uid": "story-1", "user_id": "parent-1", "child_profile_id": 101},
    ]
    child_parent_by_id = {
        "101": "parent-1",
    }

    counts = _build_story_count_by_user(stories, child_parent_by_id)

    assert counts == {
        "parent-1": 1,
    }


def test_admin_story_filters_map_schema_format_and_status_values():
    summaries = [
        {
            "id": "story-1",
            "user_email": "parent@example.com",
            "user_name": "Parent User",
            "character_name": "Nova",
            "story_title": "Nova and the Hidden Map",
            "format": "interactive_search",
            "status": "generating",
            "created_at": "2026-04-22T10:00:00+00:00",
        },
        {
            "id": "story-2",
            "user_email": "other@example.com",
            "user_name": "Other User",
            "character_name": "Milo",
            "story_title": "Milo's Story",
            "format": "story_adventure",
            "status": "completed",
            "created_at": "2026-04-21T10:00:00+00:00",
        },
    ]

    filtered = _filter_admin_story_summaries(
        summaries=summaries,
        search="parent user",
        status="processing",
        format_type="interactive_story",
        created_from=_safe_parse_datetime("2026-04-22"),
        created_to=_safe_parse_datetime("2026-04-22", end_of_day=True),
    )

    assert [summary["id"] for summary in filtered] == ["story-1"]


def test_pick_latest_story_job_falls_back_to_story_job_id():
    story_row = {"id": 123, "job_id": 999}
    jobs_by_book_id = {}
    jobs_by_id = {
        "999": {"id": 999, "status": "failed", "error_message": "generation crashed"},
    }

    latest_job = _pick_latest_story_job(
        story_row=story_row,
        jobs_by_book_id=jobs_by_book_id,
        jobs_by_id=jobs_by_id,
    )

    assert latest_job == jobs_by_id["999"]


def test_build_story_pages_returns_all_adventure_page_text_entries():
    pages = _build_story_pages(
        story_row={"scene_images": ["page-1.png", "page-2.png", "page-3.png"]},
        story_format="story_adventure",
        story_page_texts=[
            {"page_number": 1, "text": "Page one", "audio_url": None},
            {"page_number": 2, "text": "Page two", "audio_url": None},
            {"page_number": 3, "text": "Page three", "audio_url": None},
            {"page_number": 4, "text": "Page four", "audio_url": None},
            {"page_number": 5, "text": "Page five", "audio_url": None},
        ],
    )

    story_pages = [page for page in pages if page["page_number"]]

    assert [page["label"] for page in story_pages] == ["Page 1", "Page 2", "Page 3", "Page 4", "Page 5"]
    assert [page["image_url"] for page in story_pages] == [
        "page-1.png",
        "page-2.png",
        "page-3.png",
        None,
        None,
    ]


def test_build_story_pages_uses_story_content_scene_image_fallbacks():
    pages = _build_story_pages(
        story_row={
            "story_content": {
                "pages": [
                    {"pageNumber": 1, "text": "Page one", "sceneImage": "content-page-1.png"},
                    {"pageNumber": 2, "text": "Page two", "sceneImage": "content-page-2.png"},
                    {"pageNumber": 3, "text": "Page three", "sceneImage": "content-page-3.png"},
                    {"pageNumber": 4, "text": "Page four", "sceneImage": "content-page-4.png"},
                    {"pageNumber": 5, "text": "Page five", "sceneImage": "content-page-5.png"},
                ],
            },
        },
        story_format="story_adventure",
        story_page_texts=[
            {"page_number": 1, "text": "Page one", "audio_url": None},
            {"page_number": 2, "text": "Page two", "audio_url": None},
            {"page_number": 3, "text": "Page three", "audio_url": None},
            {"page_number": 4, "text": "Page four", "audio_url": None},
            {"page_number": 5, "text": "Page five", "audio_url": None},
        ],
    )

    story_pages = [page for page in pages if page["page_number"]]

    assert [page["image_url"] for page in story_pages] == [
        "content-page-1.png",
        "content-page-2.png",
        "content-page-3.png",
        "content-page-4.png",
        "content-page-5.png",
    ]


def test_build_story_pages_splits_blobbed_scene_image_urls():
    pages = _build_story_pages(
        story_row={
            "scene_images": (
                "https://example.supabase.co/storage/v1/object/public/images/page-1.png\n"
                "https://example.supabase.co/storage/v1/object/public/images/page-2.png, "
                "https://example.supabase.co/storage/v1/object/public/images/page-3.png"
            )
        },
        story_format="story_adventure",
        story_page_texts=[
            {"page_number": 1, "text": "Page one", "audio_url": None},
            {"page_number": 2, "text": "Page two", "audio_url": None},
            {"page_number": 3, "text": "Page three", "audio_url": None},
        ],
    )

    story_pages = [page for page in pages if page["page_number"]]

    assert [page["image_url"] for page in story_pages] == [
        "https://example.supabase.co/storage/v1/object/public/images/page-1.png",
        "https://example.supabase.co/storage/v1/object/public/images/page-2.png",
        "https://example.supabase.co/storage/v1/object/public/images/page-3.png",
    ]


def test_listify_urls_normalizes_json_and_blobbed_enhanced_images():
    urls = _listify_urls(
        '["https://example.supabase.co/a.png", "https://example.supabase.co/b.png"]'
    )
    blobbed_urls = _listify_urls(
        "https://example.supabase.co/c.png https://example.supabase.co/d.png"
    )

    assert urls == ["https://example.supabase.co/a.png", "https://example.supabase.co/b.png"]
    assert blobbed_urls == ["https://example.supabase.co/c.png", "https://example.supabase.co/d.png"]


def test_single_scene_image_update_only_replaces_target_page():
    update_data = _build_single_scene_image_update(
        story_row={
            "scene_images": ["page-1.png", "page-2.png", "page-3.png"],
            "story_content": {
                "pages": [
                    {"pageNumber": 1, "text": "Page one", "sceneImage": "page-1.png"},
                    {"pageNumber": 2, "text": "Page two", "sceneImage": "page-2.png"},
                    {"pageNumber": 3, "text": "Page three", "sceneImage": "page-3.png"},
                ],
            },
        },
        page_number=2,
        image_url="new-page-2.png",
    )

    story_content = json.loads(update_data["story_content"])

    assert update_data["scene_images"] == ["page-1.png", "new-page-2.png", "page-3.png"]
    assert [page["sceneImage"] for page in story_content["pages"]] == [
        "page-1.png",
        "new-page-2.png",
        "page-3.png",
    ]


def test_single_scene_image_update_preserves_missing_page_slots():
    update_data = _build_single_scene_image_update(
        story_row={"scene_images": ["page-1.png", "page-2.png"]},
        page_number=5,
        image_url="new-page-5.png",
    )

    assert update_data["scene_images"] == ["page-1.png", "page-2.png", "", "", "new-page-5.png"]


def test_single_scene_image_update_replaces_target_image_index():
    update_data = _build_single_scene_image_update(
        story_row={
            "scene_images": ["page-1a.png\npage-1b.png", "page-2.png"],
            "story_content": {
                "pages": [
                    {"pageNumber": 1, "text": "Page one", "sceneImage": ["page-1a.png", "page-1b.png"]},
                    {"pageNumber": 2, "text": "Page two", "sceneImage": "page-2.png"},
                ],
            },
        },
        page_number=1,
        image_url="new-page-1b.png",
        image_index=1,
    )

    story_content = json.loads(update_data["story_content"])

    assert update_data["scene_images"] == ["page-1a.png\nnew-page-1b.png", "page-2.png"]
    assert story_content["pages"][0]["sceneImage"] == ["page-1a.png", "new-page-1b.png"]
    assert story_content["pages"][1]["sceneImage"] == "page-2.png"


def test_story_owner_summary_falls_back_to_character_owner():
    summary = _build_story_owner_summary(
        story_row={"id": 1, "user_id": None, "child_profile_id": None},
        users_by_id={
            "parent-1": {
                "id": "parent-1",
                "email": "parent@example.com",
                "first_name": "Parent",
                "last_name": "User",
            }
        },
        child_parent_by_id={"101": "parent-1"},
        child_profiles_by_id={"101": {"id": 101, "first_name": "Ava"}},
        character={"id": 55, "user_id": "parent-1", "child_profile_id": 101},
    )

    assert summary == {
        "user_id": "parent-1",
        "user_email": "parent@example.com",
        "user_name": "Parent User",
        "child_name": "Ava",
    }


def test_user_payment_history_uses_purchased_stories_for_user():
    stories = [
        {
            "uid": "story-1",
            "user_id": "user-1",
            "child_profile_id": None,
            "story_title": "Purchased Directly",
            "created_at": "2026-04-21T10:00:00+00:00",
            "purchased": True,
        },
        {
            "uid": "story-2",
            "user_id": None,
            "child_profile_id": 101,
            "story_title": "Purchased Through Child",
            "created_at": "2026-04-22T10:00:00+00:00",
            "purchased": True,
        },
        {
            "uid": "story-3",
            "user_id": "user-1",
            "child_profile_id": None,
            "story_title": "Not Purchased",
            "created_at": "2026-04-23T10:00:00+00:00",
            "purchased": False,
        },
        {
            "uid": "story-4",
            "user_id": "user-2",
            "child_profile_id": None,
            "story_title": "Other User",
            "created_at": "2026-04-24T10:00:00+00:00",
            "purchased": True,
        },
    ]

    history = _fetch_user_payment_history_safe(
        stories=stories,
        user_id="user-1",
        child_parent_by_id={"101": "user-1"},
    )

    assert [row["story_id"] for row in history] == ["story-2", "story-1"]
    assert all(row["purchase_status"] == "completed" for row in history)


def test_user_generation_history_uses_stories_for_user():
    stories = [
        {
            "uid": "story-1",
            "user_id": "user-1",
            "child_profile_id": None,
            "story_type": "story_adventure",
            "status": "completed",
            "created_at": "2026-04-21T10:00:00+00:00",
        },
        {
            "uid": "story-2",
            "user_id": None,
            "child_profile_id": 101,
            "story_type": "interactive_search",
            "status": "processing",
            "created_at": "2026-04-22T10:00:00+00:00",
        },
        {
            "uid": "story-3",
            "user_id": "user-2",
            "child_profile_id": None,
            "story_type": "story_adventure",
            "status": "completed",
            "created_at": "2026-04-23T10:00:00+00:00",
        },
    ]

    history = _build_user_generation_history_from_stories(
        stories=stories,
        user_id="user-1",
        child_parent_by_id={"101": "user-1"},
    )

    assert [row["story_id"] for row in history] == ["story-2", "story-1"]
    assert history[0]["job_type"] == "interactive_search"
    assert history[0]["status"] == "generating"
