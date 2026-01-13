"""
Image API routes
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from io import BytesIO
from datetime import datetime
import uuid
from rate_limiter import limiter

router = APIRouter()


class ImageRequest(BaseModel):
    image_url: HttpUrl
    prompt: str


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
