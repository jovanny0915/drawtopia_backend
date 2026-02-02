"""
Google Vision API service layer for character feature extraction.
Uses Google Cloud credentials from:
  - GOOGLE_SERVICE_ACCOUNT_JSON_B64 (recommended): base64-encoded service account JSON
  - GOOGLE_SERVICE_ACCOUNT_JSON: raw JSON string (minified, one line)
  - GOOGLE_APPLICATION_CREDENTIALS: base64-encoded service account JSON content (not a file path)
Extracts labels and dominant colors from uploaded drawing images;
implements retry on timeout (max 2 retries) and structured response-time logging.
"""

import base64
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Max 2 retries on timeout => 3 attempts total
VISION_EXTRACT_MAX_RETRIES = 2


class VisionNotConfiguredError(Exception):
    """Raised when no Google credentials are set or client init fails."""

    pass


class VisionAPIError(Exception):
    """Raised when Vision API call fails after retries (e.g. timeout, API error)."""

    pass


def _credentials_from_env():
    """
    Build Google OAuth2 credentials from env if set.
    Precedence:
      1. GOOGLE_SERVICE_ACCOUNT_JSON_B64 – base64-encoded service account JSON
      2. GOOGLE_SERVICE_ACCOUNT_JSON – raw JSON string
      3. GOOGLE_APPLICATION_CREDENTIALS – base64-encoded service account JSON (not a file path)
    Returns None if none set or parsing fails.
    """
    import google.oauth2.service_account as sa

    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        try:
            raw = base64.b64decode(b64.strip()).decode("utf-8")
            info = json.loads(raw)
            return sa.Credentials.from_service_account_info(info)
        except Exception as e:
            logger.warning("Failed to load GOOGLE_SERVICE_ACCOUNT_JSON_B64: %s", e)
            return None

    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw_json:
        try:
            info = json.loads(raw_json.strip())
            return sa.Credentials.from_service_account_info(info)
        except Exception as e:
            logger.warning("Failed to load GOOGLE_SERVICE_ACCOUNT_JSON: %s", e)
            return None

    # GOOGLE_APPLICATION_CREDENTIALS = base64 of credential JSON content (not a file path)
    creds_b64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_b64:
        try:
            raw = base64.b64decode(creds_b64.strip()).decode("utf-8")
            info = json.loads(raw)
            return sa.Credentials.from_service_account_info(info)
        except Exception as e:
            logger.warning("Failed to load GOOGLE_APPLICATION_CREDENTIALS (base64): %s", e)
            return None

    return None


def get_vision_client():
    """
    Create and return a Google Vision ImageAnnotatorClient.
    Credentials (in order of precedence):
      1. GOOGLE_SERVICE_ACCOUNT_JSON_B64 – base64-encoded service account JSON
      2. GOOGLE_SERVICE_ACCOUNT_JSON – raw JSON string (minified, one line)
      3. GOOGLE_APPLICATION_CREDENTIALS – base64-encoded service account JSON content (not a file path)
    Returns None if no credentials are set or client init fails.
    """
    try:
        from google.cloud import vision

        credentials = _credentials_from_env()
        if credentials is None:
            logger.info(
                "No Google credentials set (GOOGLE_SERVICE_ACCOUNT_JSON_B64, "
                "GOOGLE_SERVICE_ACCOUNT_JSON, or GOOGLE_APPLICATION_CREDENTIALS base64); Vision API disabled."
            )
            return None

        client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("Google Vision API client initialized from env credentials")
        return client
    except Exception as e:
        logger.warning("Google Vision API client not available: %s", e)
        return None


def _is_timeout_error(e: Exception) -> bool:
    """Return True if the exception is a timeout/deadline error."""
    msg = str(e).lower()
    if "deadline" in msg or "timeout" in msg or "timed out" in msg:
        return True
    try:
        from google.api_core.exceptions import DeadlineExceeded

        if isinstance(e, DeadlineExceeded):
            return True
    except ImportError:
        pass
    return False


def extract_character_features(
    image_bytes: bytes,
    vision_client: Any,
) -> Tuple[Dict[str, Any], int]:
    """
    Extract character-relevant features from a drawing image using Google Vision API.

    Uses label_detection and image_properties_detection (dominant colors).
    Retries up to 2 times on timeout. Logs API response time for monitoring.

    Args:
        image_bytes: Raw image bytes (e.g. PNG/JPEG).
        vision_client: Initialized google.cloud.vision.ImageAnnotatorClient.
            Must not be None; caller should check get_vision_client() first.

    Returns:
        Tuple of (features_dict, response_time_ms). features_dict includes:
        labels, dominant_colors, safe_search, extraction_model, response_time_ms.

    Raises:
        VisionNotConfiguredError: If vision_client is None.
        VisionAPIError: If API fails after retries (prevents silent failures).
    """
    from google.cloud import vision

    if vision_client is None:
        raise VisionNotConfiguredError("Vision API not configured. Set GOOGLE_APPLICATION_CREDENTIALS.")

    features: Dict[str, Any] = {
        "labels": [],
        "dominant_colors": [],
        "safe_search": {},
        "extraction_model": "google_vision",
    }
    response_time_ms = 0
    last_error: Optional[Exception] = None

    for attempt in range(VISION_EXTRACT_MAX_RETRIES + 1):
        try:
            start = time.perf_counter()
            image = vision.Image(content=image_bytes)

            # Label detection
            resp_labels = vision_client.label_detection(image=image)
            if resp_labels.error.message:
                raise RuntimeError(resp_labels.error.message)
            labels = [
                {"description": label.description, "score": round(label.score, 4)}
                for label in (resp_labels.label_annotations or [])
            ]
            features["labels"] = labels[:50]

            # Image properties (dominant colors)
            resp_props = vision_client.image_properties_detection(image=image)
            if resp_props.error.message:
                pass  # non-fatal
            else:
                props_ann = getattr(
                    resp_props, "image_properties_annotation", None
                ) or getattr(resp_props, "image_annotation", None)
                dom = getattr(props_ann, "dominant_colors", None) if props_ann else None
                colors_list = getattr(dom, "colors", None) if dom else None
                if colors_list:
                    features["dominant_colors"] = [
                        {
                            "red": getattr(c.color, "red", 0) if getattr(c, "color", None) else 0,
                            "green": getattr(c.color, "green", 0) if getattr(c, "color", None) else 0,
                            "blue": getattr(c.color, "blue", 0) if getattr(c, "color", None) else 0,
                            "pixel_fraction": round(getattr(c, "pixel_fraction", 0) or 0, 4),
                        }
                        for c in colors_list[:10]
                    ]

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            response_time_ms = elapsed_ms
            features["response_time_ms"] = elapsed_ms

            # Structured logging for API response times (success criteria: under 2 seconds)
            logger.info(
                "vision_api_response_time_ms=%d attempt=%d",
                elapsed_ms,
                attempt + 1,
                extra={"vision_response_time_ms": elapsed_ms, "attempt": attempt + 1},
            )
            if elapsed_ms > 2000:
                logger.warning(
                    "Vision API response exceeded 2s target: %d ms",
                    elapsed_ms,
                    extra={"vision_response_time_ms": elapsed_ms},
                )
            return features, response_time_ms

        except Exception as e:
            last_error = e
            if _is_timeout_error(e) and attempt < VISION_EXTRACT_MAX_RETRIES:
                logger.warning(
                    "Vision API timeout (attempt %d), retrying...",
                    attempt + 1,
                    exc_info=False,
                )
                continue
            logger.error(
                "Vision character feature extraction failed: %s",
                e,
                exc_info=True,
            )
            raise VisionAPIError(f"Vision API error: {e}") from e

    raise VisionAPIError(
        f"Vision extraction failed after {VISION_EXTRACT_MAX_RETRIES + 1} attempts: {last_error}"
    )


def _label_similarity(ref_labels: list, scene_labels: list) -> float:
    """
    Compute similarity between two label lists (0.0–1.0).
    Uses weighted overlap: same description contributes by average score.
    """
    if not ref_labels and not scene_labels:
        return 1.0
    ref_set = {(e.get("description", "").lower().strip(), e.get("score", 0)) for e in (ref_labels or [])}
    scene_map = {e.get("description", "").lower().strip(): e.get("score", 0) for e in (scene_labels or [])}
    if not scene_map:
        return 0.0
    total = 0.0
    count = 0
    for desc, ref_score in ref_set:
        if not desc:
            continue
        scene_score = scene_map.get(desc, 0)
        total += (ref_score + scene_score) / 2.0
        count += 1
    if count == 0:
        return 0.0
    # Normalize by number of reference labels so 0–1
    return min(1.0, total / max(len(ref_set), 1))


def _color_similarity(ref_colors: list, scene_colors: list) -> float:
    """
    Compute dominant color similarity (0.0–1.0) using normalized RGB distance.
    """
    if not ref_colors and not scene_colors:
        return 1.0
    ref = (ref_colors or [])[:5]
    scene = (scene_colors or [])[:5]
    if not ref or not scene:
        return 0.5
    total = 0.0
    for rc in ref:
        r0, g0, b0 = rc.get("red", 0), rc.get("green", 0), rc.get("blue", 0)
        best = 0.0
        for sc in scene:
            r1, g1, b1 = sc.get("red", 0), sc.get("green", 0), sc.get("blue", 0)
            dist = ((r0 - r1) ** 2 + (g0 - g1) ** 2 + (b0 - b1) ** 2) ** 0.5
            max_dist = (255 ** 2 * 3) ** 0.5
            sim = 1.0 - min(1.0, dist / max_dist)
            best = max(best, sim)
        total += best
    return total / len(ref) if ref else 0.0


def validate_scene_against_reference(
    reference_bytes: bytes,
    scene_bytes: bytes,
    vision_client: Any,
) -> Tuple[Dict[str, Any], int]:
    """
    Compare generated scene image to reference character image using Vision API.
    Extracts features from both, computes similarity, returns confidence 0–100.

    Edge cases (extremely abstract drawings, minimal detail): we still run comparison;
    low label/color overlap may yield low confidence. On API error we approve (confidence 100)
    so we don't block. Handling may need refinement—see docs/IMPLEMENTATION_NOTES.md.

    Args:
        reference_bytes: Raw bytes of the reference character image.
        scene_bytes: Raw bytes of the generated scene image.
        vision_client: Initialized Vision ImageAnnotatorClient, or None.

    Returns:
        Tuple of (result_dict, confidence_0_100).
        result_dict includes: confidence_score (0–100), timed_out (bool),
        skipped (bool), label_similarity, color_similarity, response_time_ms.
    """
    result: Dict[str, Any] = {
        "confidence_score": 0,
        "timed_out": False,
        "skipped": False,
        "label_similarity": 0.0,
        "color_similarity": 0.0,
        "response_time_ms": 0,
    }
    if vision_client is None:
        result["confidence_score"] = 100
        result["skipped"] = True
        result["reason"] = "vision_not_configured"
        logger.info("Vision API not configured; scene validation skipped, confidence=100")
        return (result, 100)

    try:
        start = time.perf_counter()
        ref_features, _ = extract_character_features(reference_bytes, vision_client)
        scene_features, _ = extract_character_features(scene_bytes, vision_client)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result["response_time_ms"] = elapsed_ms

        label_sim = _label_similarity(
            ref_features.get("labels", []),
            scene_features.get("labels", []),
        )
        color_sim = _color_similarity(
            ref_features.get("dominant_colors", []),
            scene_features.get("dominant_colors", []),
        )
        result["label_similarity"] = round(label_sim, 4)
        result["color_similarity"] = round(color_sim, 4)
        # Combined score: 60% labels, 40% colors, scale to 0–100
        confidence = int(round((0.6 * label_sim + 0.4 * color_sim) * 100))
        confidence = max(0, min(100, confidence))
        result["confidence_score"] = confidence

        logger.info(
            "vision_validation confidence_score=%d label_sim=%.3f color_sim=%.3f response_time_ms=%d",
            confidence,
            label_sim,
            color_sim,
            elapsed_ms,
            extra={
                "confidence_score": confidence,
                "label_similarity": label_sim,
                "color_similarity": color_sim,
                "response_time_ms": elapsed_ms,
            },
        )
        return (result, confidence)
    except (VisionNotConfiguredError, VisionAPIError) as e:
        logger.warning("Vision validation failed, approving scene: %s", e)
        result["confidence_score"] = 100
        result["skipped"] = True
        result["error"] = str(e)
        return (result, 100)
    except Exception as e:
        logger.exception("Vision validation error: %s", e)
        result["confidence_score"] = 100
        result["skipped"] = True
        result["error"] = str(e)
        return (result, 100)
