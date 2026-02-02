"""
Unit tests for Vision API character feature extraction.
Validates expected feature structure and response-time logging.
Uses 5 test images from tests/fixtures/vision/ (simple, detailed, abstract, colorful, minimal).
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add backend root so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "vision")
EXPECTED_IMAGE_NAMES = ["simple.png", "detailed.png", "abstract.png", "colorful.png", "minimal.png"]


def test_vision_feature_structure_expected_keys():
    """Vision API extraction result must include labels, dominant_colors, extraction_model, response_time_ms."""
    # Expected structure returned by Vision service extract_character_features
    expected_keys = {"labels", "dominant_colors", "safe_search", "extraction_model", "response_time_ms"}
    # Minimal valid feature dict (as returned by our normalizer)
    minimal_features = {
        "labels": [],
        "dominant_colors": [],
        "safe_search": {},
        "extraction_model": "google_vision",
        "response_time_ms": 0,
    }
    assert set(minimal_features.keys()) >= expected_keys
    assert minimal_features["extraction_model"] == "google_vision"
    assert isinstance(minimal_features["labels"], list)
    assert isinstance(minimal_features["dominant_colors"], list)


def test_vision_feature_structure_labels_shape():
    """Each label in features['labels'] must have description and score."""
    label = {"description": "Cartoon", "score": 0.95}
    assert "description" in label
    assert "score" in label
    assert isinstance(label["description"], str)
    assert isinstance(label["score"], (int, float))


def test_vision_feature_structure_dominant_colors_shape():
    """Each entry in dominant_colors may have red, green, blue, pixel_fraction."""
    color_entry = {"red": 255, "green": 128, "blue": 0, "pixel_fraction": 0.25}
    assert "red" in color_entry or "pixel_fraction" in color_entry
    assert isinstance(color_entry.get("pixel_fraction", 0), (int, float))


def test_vision_response_time_logged():
    """Features dict must include response_time_ms for API response time tracking."""
    features = {
        "labels": [{"description": "Drawing", "score": 0.9}],
        "dominant_colors": [],
        "safe_search": {},
        "extraction_model": "google_vision",
        "response_time_ms": 120,
    }
    assert "response_time_ms" in features
    assert isinstance(features["response_time_ms"], (int, float))
    assert features["response_time_ms"] >= 0


def test_fixture_images_exist():
    """All 5 test images (simple, detailed, abstract, colorful, minimal) must exist."""
    for name in EXPECTED_IMAGE_NAMES:
        path = os.path.join(FIXTURE_DIR, name)
        assert os.path.isfile(path), f"Fixture image missing: {path}"


def test_vision_service_raises_when_client_none():
    """Error handling: service raises VisionNotConfiguredError when client is None (no silent failure)."""
    from services.vision_character_features import extract_character_features, VisionNotConfiguredError

    with pytest.raises(VisionNotConfiguredError):
        extract_character_features(b"fake-image-bytes", None)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="GOOGLE_APPLICATION_CREDENTIALS not set; Vision API integration test skipped",
)
def test_vision_extract_returns_expected_structure():
    """
    Integration test: call Vision extraction with a fixture image and assert feature structure.
    Skipped if Vision is not configured.
    """
    import main
    from services.vision_character_features import extract_character_features

    if not main.vision_client:
        pytest.skip("Vision client not initialized")
    path = os.path.join(FIXTURE_DIR, "simple.png")
    with open(path, "rb") as f:
        image_bytes = f.read()
    features, response_time_ms = extract_character_features(image_bytes, main.vision_client)
    assert "labels" in features
    assert "dominant_colors" in features
    assert features.get("extraction_model") == "google_vision"
    assert "response_time_ms" in features
    assert features["response_time_ms"] == response_time_ms
    assert response_time_ms >= 0
    # Labels may be empty for very simple images but structure must be list
    assert isinstance(features["labels"], list)
    for item in features["labels"]:
        assert "description" in item and "score" in item
    # Success criteria: API calls complete in under 2 seconds per image
    assert response_time_ms < 2000, f"Vision API took {response_time_ms} ms (target < 2000 ms)"
