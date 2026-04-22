import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apis.admin import (
    _build_story_count_by_user,
    _filter_user_summaries,
    _matches_date_range,
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
