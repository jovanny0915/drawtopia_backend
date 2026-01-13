"""
Validation utilities for character consistency
"""

import logging
import base64
import time
import json
import re
from typing import Dict, Any, Optional
from pydantic import BaseModel
from image_utils import detect_image_mime_type
from google.genai import types

logger = logging.getLogger(__name__)


class ConsistencyValidationResult(BaseModel):
    is_consistent: bool
    similarity_score: float  # 0.0 to 1.0
    validation_time_seconds: float
    flagged: bool  # True if score < 0.5
    details: Optional[Dict[str, Any]] = None


def validate_character_consistency(
    scene_image_data: bytes,
    reference_image_data: bytes,
    page_number: int,
    gemini_client,
    gemini_text_model: str = "gemini-2.5-flash",
    timeout_seconds: int = 15
) -> ConsistencyValidationResult:
    """
    Validate character consistency between a scene image and reference image using Gemini model.
    Compares the character in the scene against the reference (normal.png) image.
    
    Args:
        scene_image_data: Bytes of the generated scene image
        reference_image_data: Bytes of the reference character image (normal.png)
        page_number: Page number for logging
        gemini_client: Gemini client instance
        gemini_text_model: Model name for text generation
        timeout_seconds: Maximum time allowed for validation (default 15 seconds)
    
    Returns:
        ConsistencyValidationResult with similarity score and validation details
    """
    if not gemini_client:
        logger.warning("Gemini client not available for consistency validation")
        return ConsistencyValidationResult(
            is_consistent=True,  # Default to consistent if validation unavailable
            similarity_score=0.5,
            validation_time_seconds=0.0,
            flagged=False,
            details={"validation_available": False, "error": "Gemini client not initialized"}
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Starting character consistency validation for page {page_number}...")
        
        # Detect MIME types
        scene_mime_type = detect_image_mime_type(scene_image_data)
        reference_mime_type = detect_image_mime_type(reference_image_data)
        
        # Encode images to base64
        scene_base64 = base64.b64encode(scene_image_data).decode('utf-8')
        reference_base64 = base64.b64encode(reference_image_data).decode('utf-8')
        
        # Create validation prompt for Gemini
        validation_prompt = """Analyze these two images and determine how consistent the character appearance is between them.

IMAGE 1 (REFERENCE): This is the reference character image (normal.png) showing the character's standard appearance.

IMAGE 2 (SCENE): This is a scene from a storybook that should contain the same character.

Your task is to compare the character in the scene image against the reference image and provide a similarity score.

Focus on these character features:
1. Facial features (eyes, nose, mouth, face shape)
2. Body proportions and structure
3. Hair style, color, and texture
4. Skin tone and color
5. Clothing design, colors, and patterns
6. Overall character design and visual style
7. Character's artistic style consistency

Return your analysis in the following JSON format (ONLY valid JSON, no additional text):
{
  "similarity_score": <float between 0.0 and 1.0>,
  "is_consistent": <boolean>,
  "character_match_details": {
    "facial_features_match": <float 0.0-1.0>,
    "body_proportions_match": <float 0.0-1.0>,
    "hair_match": <float 0.0-1.0>,
    "skin_tone_match": <float 0.0-1.0>,
    "clothing_match": <float 0.0-1.0>,
    "overall_style_match": <float 0.0-1.0>
  },
  "issues": [<array of specific inconsistency issues found>],
  "confidence": <float 0.0-1.0>
}

Scoring guidelines:
- 0.9-1.0: Character is nearly identical or identical to reference
- 0.7-0.89: Character is very similar with minor differences
- 0.5-0.69: Character is somewhat similar but has noticeable differences
- 0.3-0.49: Character has significant differences from reference
- 0.0-0.29: Character is very different or unrecognizable compared to reference

Threshold: A score of 0.5 or higher indicates consistency. Below 0.5 should be flagged as inconsistent."""
        
        # Call Gemini API for validation
        validation_start = time.time()
        
        response = gemini_client.models.generate_content(
            model=gemini_text_model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": validation_prompt},
                        {
                            "inline_data": {
                                "mime_type": reference_mime_type,
                                "data": reference_base64
                            }
                        },
                        {
                            "text": "\n\nIMAGE 2 (SCENE):"
                        },
                        {
                            "inline_data": {
                                "mime_type": scene_mime_type,
                                "data": scene_base64
                            }
                        }
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT'],
                temperature=0.1  # Lower temperature for more consistent validation
            )
        )
        
        # Extract text response
        validation_text = ""
        for part in response.parts:
            if part.text:
                validation_text += part.text
        
        # Check if we exceeded timeout
        elapsed_time = time.time() - validation_start
        if elapsed_time > timeout_seconds:
            logger.warning(f"Consistency validation for page {page_number} exceeded timeout ({elapsed_time:.2f}s > {timeout_seconds}s)")
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
        if json_match:
            validation_json = json.loads(json_match.group())
        else:
            validation_json = json.loads(validation_text)
        
        # Extract validation results
        similarity_score = float(validation_json.get("similarity_score", 0.5))
        is_consistent = validation_json.get("is_consistent", similarity_score >= 0.5)
        character_match_details = validation_json.get("character_match_details", {})
        issues = validation_json.get("issues", [])
        confidence = validation_json.get("confidence", 0.5)
        
        # Determine if flagged (score < 0.5)
        flagged = similarity_score < 0.5
        
        total_time = time.time() - start_time
        
        result = ConsistencyValidationResult(
            is_consistent=is_consistent,
            similarity_score=similarity_score,
            validation_time_seconds=total_time,
            flagged=flagged,
            details={
                "character_match_details": character_match_details,
                "issues": issues,
                "confidence": confidence,
                "model_used": gemini_text_model,
                "timeout_seconds": timeout_seconds
            }
        )
        
        # Log results
        logger.info(f"✅ Consistency validation for page {page_number} completed in {total_time:.2f}s")
        logger.info(f"   Similarity score: {similarity_score:.3f} | Consistent: {is_consistent} | Flagged: {flagged}")
        if issues:
            logger.warning(f"   Issues found: {', '.join(issues[:3])}")  # Log first 3 issues
        
        if flagged:
            logger.warning(f"⚠️ Page {page_number} flagged as INCONSISTENT (score: {similarity_score:.3f} < 0.5)")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse consistency validation JSON response for page {page_number}: {e}")
        logger.error(f"Response text: {validation_text[:500] if 'validation_text' in locals() else 'N/A'}")
        total_time = time.time() - start_time
        return ConsistencyValidationResult(
            is_consistent=True,  # Default to consistent on parse error
            similarity_score=0.5,
            validation_time_seconds=total_time,
            flagged=False,
            details={"validation_available": False, "error": "JSON parse error", "raw_response": validation_text[:200] if 'validation_text' in locals() else None}
        )
    except Exception as e:
        logger.error(f"Error during consistency validation for page {page_number}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        total_time = time.time() - start_time
        return ConsistencyValidationResult(
            is_consistent=True,  # Default to consistent on error
            similarity_score=0.5,
            validation_time_seconds=total_time,
            flagged=False,
            details={"validation_available": False, "error": str(e)}
        )

