"""
Image API routes
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from io import BytesIO
from datetime import datetime
import uuid
import base64
import re
from rate_limiter import limiter

router = APIRouter()


class ImageRequest(BaseModel):
    image_url: HttpUrl
    prompt: str


class CompareSimilarityRequest(BaseModel):
    image1_url: HttpUrl
    image2_url: HttpUrl


class CompareSimilarityResponse(BaseModel):
    success: bool
    score: float
    message: str


class SearchGameHintRequest(BaseModel):
    env_img: HttpUrl
    character_img: HttpUrl


class SearchGameHintResponse(BaseModel):
    success: bool
    hint_text: str
    message: str


@router.post("/validate-image-quality/")
@limiter.limit("30/minute")
async def validate_image_quality_endpoint(request: Request, body: ImageRequest):
    """
    Standalone endpoint to validate image quality without editing.
    Useful for pre-validation before processing.
    """
    import main  # Import here to avoid circular import
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(body.image_url)
        
        # Download the image from the URL provided
        main.logger.info(f"Downloading image for quality validation from: {image_url_str}")
        image_data = main.download_image_from_url(image_url_str)
        
        # Validate image quality
        validation_result = main.validate_image_quality(image_data, image_url_str)
        
        return main.QualityValidationResponse(
            success=True,
            validation=validation_result
        )
        
    except HTTPException as e:
        main.logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in validate_image_quality_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/edit-image/")
@limiter.limit("20/minute")
async def edit_image_endpoint(request: Request, body: ImageRequest):
    import main  # Import here to avoid circular import
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(body.image_url)
        
        # Download the image from the URL provided
        main.logger.info(f"Downloading image from: {image_url_str}")
        image_data = main.download_image_from_url(image_url_str)

        # Validate image quality before processing
        main.logger.info("Validating image quality...")
        quality_validation = main.validate_image_quality(image_data, image_url_str)
        
        # Log validation results
        if not quality_validation.get("is_valid", True):
            main.logger.warning(f"Image quality validation failed: {quality_validation.get('issues', [])}")
        
        # Send the image to Gemini API for editing
        main.logger.info(f"Received prompt: {body.prompt}")
        edited_image = main.edit_image(image_data, body.prompt, image_url_str)
        
        # Optimize image to JPG format for smaller file size
        main.logger.info("Optimizing image to JPG format...")
        optimized_image = main.optimize_image_to_jpg(edited_image)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"edited_image_{timestamp}_{unique_id}.jpg"
        
        # Upload optimized image to Supabase storage
        storage_result = main.upload_to_supabase(optimized_image, filename)
        
        if storage_result["uploaded"]:
            return main.ImageResponse(
                success=True,
                message="Image edited and uploaded successfully to Supabase storage",
                storage_info=storage_result,
                quality_validation=quality_validation
            )
        else:
            # Even if upload fails, we can still return the image data
            main.logger.warning("Supabase upload failed, but image was processed successfully")
            return main.ImageResponse(
                success=True,
                message="Image edited successfully, but storage upload failed",
                storage_info=storage_result,
                quality_validation=quality_validation
            )
            
    except HTTPException as e:
        main.logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in edit_image_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/edit-image-stream/")
@limiter.limit("20/minute")
async def edit_image_stream_endpoint(request: Request, body: ImageRequest):
    """Alternative endpoint that returns the image as a stream (for direct download)"""
    import main  # Import here to avoid circular import
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(body.image_url)
        
        # Download the image from the URL provided
        main.logger.info(f"Downloading image from: {image_url_str}")
        image_data = main.download_image_from_url(image_url_str)

        # Send the image to Gemini API for editing
        main.logger.info(f"Received prompt: {body.prompt}")
        edited_image = main.edit_image(image_data, body.prompt, image_url_str)
        
        # Optimize image to JPG format for smaller file size
        main.logger.info("Optimizing image to JPG format...")
        optimized_image = main.optimize_image_to_jpg(edited_image)
        
        return StreamingResponse(
            BytesIO(optimized_image), 
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=edited_image.jpg"}
        )
    except HTTPException as e:
        main.logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in edit_image_stream_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/compare-similarity/")
@limiter.limit("30/minute")
async def compare_similarity_endpoint(request: Request, body: CompareSimilarityRequest):
    """
    Compare two images and return a similarity score from 0-10.
    Uses Gemini API to analyze visual similarity between the two images.
    """
    import main  # Import here to avoid circular import
    
    if not main.gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized. Please check GEMINI_API_KEY.")
    
    try:
        # Convert HttpUrl to string for processing
        image_url1_str = str(body.image1_url)
        image_url2_str = str(body.image2_url)
        
        # Download both images
        main.logger.info(f"Downloading first image from: {image_url1_str}")
        image_data1 = main.download_image_from_url(image_url1_str)
        
        main.logger.info(f"Downloading second image from: {image_url2_str}")
        image_data2 = main.download_image_from_url(image_url2_str)
        
        # Detect MIME types
        mime_type1 = main.detect_image_mime_type(image_data1)
        mime_type2 = main.detect_image_mime_type(image_data2)
        
        # Encode images to base64
        image_base64_1 = base64.b64encode(image_data1).decode('utf-8')
        image_base64_2 = base64.b64encode(image_data2).decode('utf-8')
        
        # Prepare prompt for similarity comparison
        comparison_prompt = """Compare these two images and provide a similarity score from 0 to 10, where:
- 0 means completely different (no similarity)
- 5 means moderately similar (some shared elements)
- 10 means identical or nearly identical

Consider visual elements such as:
- Overall composition and layout
- Color scheme and palette
- Subject matter and objects
- Style and artistic approach
- Background and setting

Respond with ONLY a number between 0 and 10 (including decimals like 7.5), and optionally a brief explanation after the number. Example: "7.5 - Similar color palette and composition" or just "8.2"."""
        
        # Send request to Gemini API with both images
        main.logger.info("Sending images to Gemini API for similarity comparison...")
        response = main.gemini_client.models.generate_content(
            model=main.MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": comparison_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type1,
                                "data": image_base64_1
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type2,
                                "data": image_base64_2
                            }
                        }
                    ]
                }
            ],
            config=main.types.GenerateContentConfig(
                response_modalities=['TEXT']
            )
        )
        
        # Extract text response
        response_text = ""
        for part in response.parts:
            if part.text:
                response_text += part.text
        
        main.logger.info(f"Gemini API response: {response_text}")
        
        # Extract score from response (look for number between 0-10)
        # Try to find decimal numbers like 7.5, 8.2, 10, etc.
        score = None
        decimal_pattern = r'\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\b'
        matches = re.findall(decimal_pattern, response_text)
        
        # Try each match and find the first valid score between 0-10
        for match in matches:
            try:
                potential_score = float(match)
                if 0 <= potential_score <= 10:
                    score = potential_score
                    break
            except ValueError:
                continue
        
        # If no valid score found, try to extract any number and clamp it
        if score is None:
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                try:
                    potential_score = float(numbers[0])
                    score = max(0, min(10, potential_score))  # Clamp between 0-10
                except ValueError:
                    score = 5.0  # Default fallback
            else:
                score = 5.0  # Default fallback if no numbers found
        
        # Ensure score is between 0-10
        score = max(0.0, min(10.0, float(score)))
        
        return CompareSimilarityResponse(
            success=True,
            score=score,
            message=f"Similarity comparison completed. Score: {score}/10"
        )
        
    except HTTPException as e:
        main.logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in compare_similarity_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/search-game-hint/")
@limiter.limit("30/minute")
async def search_game_hint_endpoint(request: Request, body: SearchGameHintRequest):
    """
    Generate a hint describing where the character image is located in the environment image.
    Uses Gemini API (text model) to analyze both images and provide a descriptive hint.
    """
    import main  # Import here to avoid circular import
    
    if not main.gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized. Please check GEMINI_API_KEY.")
    
    try:
        # Convert HttpUrl to string for processing
        env_img_url = str(body.env_img)
        character_img_url = str(body.character_img)
        
        # Download both images
        main.logger.info(f"Downloading environment image from: {env_img_url}")
        env_image_data = main.download_image_from_url(env_img_url)
        
        main.logger.info(f"Downloading character image from: {character_img_url}")
        character_image_data = main.download_image_from_url(character_img_url)
        
        # Detect MIME types
        env_mime_type = main.detect_image_mime_type(env_image_data)
        character_mime_type = main.detect_image_mime_type(character_image_data)
        
        # Encode images to base64
        env_image_base64 = base64.b64encode(env_image_data).decode('utf-8')
        character_image_base64 = base64.b64encode(character_image_data).decode('utf-8')
        
        # Prepare prompt for hint generation
        hint_prompt = """Look at these two images. The first image is an environment scene, and the second image is a character.

Your task is to describe where the character (from the second image) is located in the environment scene (first image). 

Provide a clear, helpful hint that describes the location of the character in the environment. Be specific about:
- The area or region of the scene (e.g., "top left", "center", "bottom right")
- Nearby objects or landmarks that can help locate the character
- Any distinctive features or colors in that area

Keep the hint concise (1-2 sentences) and child-friendly. Do not reveal the exact position, but give enough information to guide the search.

Example format: "Look in the upper right area near the colorful flowers" or "Check the bottom left corner where the trees are"."""
        
        # Send request to Gemini API with both images using text model (not pro)
        main.logger.info("Sending images to Gemini API for hint generation...")
        response = main.gemini_client.models.generate_content(
            model=main.GEMINI_TEXT_MODEL,  # Use text model (gemini-2.5-flash), not pro
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": hint_prompt},
                        {
                            "inline_data": {
                                "mime_type": env_mime_type,
                                "data": env_image_base64
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": character_mime_type,
                                "data": character_image_base64
                            }
                        }
                    ]
                }
            ],
            config=main.types.GenerateContentConfig(
                response_modalities=['TEXT']
            )
        )
        
        # Extract text response
        hint_text = ""
        for part in response.parts:
            if part.text:
                hint_text += part.text
        
        # Clean up the hint text (remove extra whitespace)
        hint_text = hint_text.strip()
        
        main.logger.info(f"Generated hint text: {hint_text}")
        
        if not hint_text:
            raise HTTPException(status_code=500, detail="Failed to generate hint text from Gemini API")
        
        return SearchGameHintResponse(
            success=True,
            hint_text=hint_text,
            message="Hint generated successfully"
        )
        
    except HTTPException as e:
        main.logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in search_game_hint_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
