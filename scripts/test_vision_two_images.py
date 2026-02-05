"""
Test script to get Vision API analysis for two images.
Run from backend root: python scripts/test_vision_two_images.py [image1.png] [image2.png]
If no paths given, uses tests/fixtures/vision/simple.png and colorful.png.
"""

import json
import os
import sys

# Load .env and add backend root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from services.vision_character_features import (
    extract_character_features,
    get_vision_client,
    VisionNotConfiguredError,
    VisionAPIError,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures", "vision"
)
DEFAULT_IMAGE_1 = os.path.join(FIXTURE_DIR, "simple.png")
DEFAULT_IMAGE_2 = os.path.join(FIXTURE_DIR, "colorful.png")


def main():
    img1 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_1
    img2 = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE_2

    for path in [img1, img2]:
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)

    client = get_vision_client()
    if not client:
        print("Error: Vision API not configured. Set GOOGLE_VISION_API_KEY or GOOGLE_SERVICE_ACCOUNT_JSON_B64 in .env")
        sys.exit(1)

    print("=" * 60)
    print("Vision API Analysis - Two Images")
    print("=" * 60)

    for i, path in enumerate([img1, img2], 1):
        print(f"\n--- Image {i}: {path} ---")
        with open(path, "rb") as f:
            image_bytes = f.read()
        try:
            features, response_time_ms = extract_character_features(image_bytes, client)
            print(f"Response time: {response_time_ms} ms")
            print("\nLabels:")
            for lbl in features.get("labels", [])[:15]:
                print(f"  - {lbl['description']}: {lbl['score']:.3f}")
            if len(features.get("labels", [])) > 15:
                print(f"  ... and {len(features['labels']) - 15} more")

            print("\nDominant colors:")
            for c in features.get("dominant_colors", [])[:10]:
                r = c.get("red", 0)
                g = c.get("green", 0)
                b = c.get("blue", 0)
                frac = c.get("pixel_fraction", 0)
                print(f"  RGB({r},{g},{b}) pixel_fraction={frac:.3f}")

            print("\nFull JSON (minified):")
            print(json.dumps(features, indent=2))
        except VisionNotConfiguredError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except VisionAPIError as e:
            print(f"Vision API error: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
