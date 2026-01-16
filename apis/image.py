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
    image_url1: HttpUrl
    image_url2: HttpUrl


class CompareSimilarityResponse(BaseModel):
    success: bool
    score: float
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
        image_url1_str = str(body.image_url1)
        image_url2_str = str(body.image_url2)
        
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
