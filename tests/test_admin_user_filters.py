import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apis.admin import _matches_date_range, _safe_parse_datetime


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
