"""
Story API routes
"""
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import uuid
import requests
import time
from pydantic import HttpUrl, BaseModel
from rate_limiter import limiter
from story_lib import generate_story
from audio_generator import AudioGenerator
from pdf_generator import create_book_pdf_with_cover
from .models import StoryRequest, SearchGameResultRequest

if TYPE_CHECKING:
    import main

router = APIRouter()


@router.post("/api/books/generate")
@limiter.limit("10/minute")
async def create_book_generation_job(request: Request, body):
    """Create a new book generation job"""
    import main  # Import here to avoid circular import
    try:
        if not main.queue_manager:
            raise HTTPException(
                status_code=500,
                detail="Queue manager not initialized"
            )
        
        # Validate job_type
        if body.job_type not in ["interactive_search", "story_adventure"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid job_type. Must be 'interactive_search' or 'story_adventure'"
            )
        
        # Validate age_group
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if body.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {body.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )
        
        # Validate priority
        if body.priority < 1 or body.priority > 10:
            raise HTTPException(
                status_code=400,
                detail="Priority must be between 1 and 10 (1 is highest)"
            )
        
        # Prepare job data
        job_data = {
            "character_name": body.character_name,
            "character_type": body.character_type,
            "special_ability": body.special_ability,
            "age_group": body.age_group,
            "story_world": body.story_world,
            "adventure_type": body.adventure_type,
            "occasion_theme": body.occasion_theme,
            "character_image_url": str(body.character_image_url) if body.character_image_url else None
        }
        
        # Create job
        job = main.queue_manager.create_job(
            job_type=body.job_type,
            job_data=job_data,
            user_id=body.user_id,
            child_profile_id=body.child_profile_id,
            priority=body.priority
        )
        
        return main.JobResponse(
            success=True,
            job_id=job["id"],
            message=f"Job {job['id']} created successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


@router.get("/api/books/{book_id}/status")
@limiter.limit("60/minute")
async def get_book_status(request: Request, book_id: int):
    """Get the status of a book generation job"""
    import main  # Import here to avoid circular import
    try:
        if not main.queue_manager:
            raise HTTPException(
                status_code=500,
                detail="Queue manager not initialized"
            )
        
        job_status = main.queue_manager.get_job_status(book_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail=f"Job {book_id} not found"
            )
        
        job = job_status["job"]
        
        return main.JobStatusResponse(
            job_id=book_id,
            status=job["status"],
            overall_progress=job_status["overall_progress"],
            stages=job_status["stages"],
            error_message=job.get("error_message"),
            result_data=job.get("result_data")
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")


@router.get("/api/books/")
@limiter.limit("60/minute")
async def list_all_books(request: Request, parent_id: Optional[str] = None):
    """
    Get all story data from the stories table
    
    Args:
        parent_id: Optional parent user ID to filter stories by parent's children
    
    Returns:
        List of all story/book data, optionally filtered by parent
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        # If parent_id is provided, filter by parent's children
        if parent_id:
            # First, get all child profile IDs for this parent
            child_profiles_response = main.supabase.table("child_profiles").select("*").eq("parent_id", parent_id).execute()
            
            if child_profiles_response.data is None or len(child_profiles_response.data) == 0:
                main.logger.info(f"No child profiles found for parent {parent_id}")
                return []
            
            # Extract child profile IDs
            child_profile_ids = [profile["id"] for profile in child_profiles_response.data]
            
            # Get user data for parent
            user_response = main.supabase.table("users").select("*").eq("id", parent_id).execute()
            user_data = user_response.data[0] if user_response.data and len(user_response.data) > 0 else None
            
            # Get all stories for these child profiles
            response = main.supabase.table("stories").select("*").in_("child_profile_id", child_profile_ids).order("created_at", desc=True).execute()
            
            if response.data is None:
                main.logger.warning("No stories found or query returned None")
                return []
            
            # Merge child profile data with stories
            stories_with_child_data = []
            for story in response.data:
                child_profile = next((cp for cp in child_profiles_response.data if cp["id"] == story["child_profile_id"]), None)
                user_name = "Unknown"
                if user_data:
                    first_name = user_data.get('first_name', '')
                    last_name = user_data.get('last_name', '')
                    user_name = f"{first_name} {last_name}".strip() or "Unknown"
                story_with_data = {
                    **story,
                    "user_name": user_name,
                    "child_profiles": child_profile
                }
                stories_with_child_data.append(story_with_data)
            
            main.logger.info(f"Retrieved {len(stories_with_child_data)} stories for parent {parent_id}")
            return stories_with_child_data
        else:
            # Query all stories from the stories table
            response = main.supabase.table("stories").select("*").execute()
            
            if response.data is None:
                main.logger.warning("No stories found or query returned None")
                return []
            
            main.logger.info(f"Retrieved {len(response.data)} stories")
            return response.data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error listing all books: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error listing all books: {str(e)}")


@router.get("/api/books/{id}/preview")
@limiter.limit("60/minute")
async def get_book_preview(request: Request, id: str):
    """
    Get book data from the stories table by ID or UID
    
    Args:
        id: Book ID (integer) or UID (string)
    
    Returns:
        Book data from the stories table
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        # Try to find book by uid first (in case id is a string uid)
        story_response = main.supabase.table("stories").select("*").eq("uid", id).execute()
        
        # If no result with uid, try id (in case id is an integer)
        if not story_response.data or len(story_response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Book {id} not found (tried both uid and id)"
            )
        
        book_data = story_response.data[0]
        main.logger.info(f"Retrieved book preview for id={id}")
        
        return book_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error getting book preview: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting book preview: {str(e)}")


@router.delete("/api/books/{id}")
@limiter.limit("30/minute")
async def delete_book(request: Request, id: str):
    """
    Delete a book from the stories table by ID or UID
    
    Args:
        id: Book ID (integer) or UID (string)
    
    Returns:
        Success message with deleted book information
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        # First, try to find the book by uid (in case id is a string uid)
        story_response = main.supabase.table("stories").select("*").eq("uid", id).execute()
        
        # If no result with uid, try id (in case id is an integer)
        if not story_response.data or len(story_response.data) == 0:
            # Try by numeric id
            try:
                numeric_id = int(id)
                story_response = main.supabase.table("stories").select("*").eq("id", numeric_id).execute()
            except ValueError:
                pass  # id is not numeric, continue with error
        
        if not story_response.data or len(story_response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Book {id} not found"
            )
        
        book_data = story_response.data[0]
        book_id = book_data.get("id")
        book_uid = book_data.get("uid")
        
        # Delete the book - try by id first (more reliable)
        if book_id:
            delete_response = main.supabase.table("stories").delete().eq("id", book_id).execute()
        elif book_uid:
            delete_response = main.supabase.table("stories").delete().eq("uid", book_uid).execute()
        else:
            raise HTTPException(
                status_code=400,
                detail="Book has no valid identifier (id or uid)"
            )
        
        main.logger.info(f"Deleted book with id={id} (db_id={book_id}, uid={book_uid})")
        
        return {
            "success": True,
            "message": f"Book {id} deleted successfully",
            "deleted_book": {
                "id": book_id,
                "uid": book_uid,
                "title": book_data.get("story_title", "Unknown")
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error deleting book: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error deleting book: {str(e)}")


@router.post("/generate-story/")
@limiter.limit("10/minute")
async def generate_story_endpoint(request: Request, body: StoryRequest):
    """Generate a 5-page children's story based on the provided parameters"""
    import main  # Import here to avoid circular import
    try:
        # Validate age_group
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if body.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {body.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )
        
        main.logger.info(f"Generating story for character: {body.character_name}")
        main.logger.info(f"Age group: {body.age_group}, Adventure: {body.adventure_type}")
        
        # Validate API keys
        if not main.OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        if not main.GEMINI_API_KEY or not main.gemini_client:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key not configured or client not initialized. Please set GEMINI_API_KEY environment variable."
            )
        
        # Generate the story using OpenAI (GPT-4) via the story library
        main.logger.info("Generating story with OpenAI GPT-4...")
        story_result = generate_story(
            character_name=body.character_name,
            character_type=body.character_type,
            special_ability=body.special_ability,
            age_group=body.age_group,
            story_world=body.story_world,
            adventure_type=body.adventure_type,
            occasion_theme=body.occasion_theme,
            use_api=True,  # Use OpenAI API for story generation
            api_key=main.OPENAI_API_KEY,
            story_text_prompt=body.story_text_prompt  # Use prompt from frontend if provided
        )
        
        main.logger.info(f"Story generated successfully. Word count: {story_result['word_count']}")
        
        # Generate dedication image FIRST before story images (if dedication info is provided)
        dedication_image_url = None
        if body.dedication_text and body.dedication_scene_prompt:
            main.logger.info("Generating dedication page image...")
            try:
                # Create blank base image for dedication (typically portrait format for dedication pages)
                dedication_base_image = main.create_blank_base_image(width=768, height=1024)  # Portrait format
                
                # Use edit_image function to generate the dedication image
                main.logger.info("Calling edit_image function for dedication page...")
                dedication_image_bytes = main.edit_image(dedication_base_image, body.dedication_scene_prompt, None)
                
                # Optimize image to JPG format
                main.logger.info("Optimizing dedication image to JPG format...")
                optimized_dedication_image = main.optimize_image_to_jpg(dedication_image_bytes)
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                dedication_filename = f"dedication_{timestamp}_{unique_id}.jpg"
                
                # Upload to Supabase and get URL
                dedication_storage_result = main.upload_to_supabase(optimized_dedication_image, dedication_filename)
                
                if dedication_storage_result.get("uploaded") and dedication_storage_result.get("url"):
                    dedication_image_url = dedication_storage_result['url']
                    main.logger.info(f"✅ Dedication image generated and uploaded: {dedication_image_url}")
                else:
                    main.logger.warning("Failed to upload dedication image")
            except Exception as e:
                main.logger.error(f"Error generating dedication image: {e}")
                import traceback
                main.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Generate scene images for each page using Gemini Pro image preview model
        main.logger.info("Generating scene images with Gemini Pro image preview model for each story page...")
        reference_image_url = str(body.character_image_url) if body.character_image_url else None
        
        # Download reference image once for consistency validation
        reference_image_data = None
        if reference_image_url:
            try:
                main.logger.info(f"Downloading reference image for consistency validation: {reference_image_url}")
                reference_image_data = main.download_image_from_url(reference_image_url)
                main.logger.info(f"✅ Reference image downloaded for validation ({len(reference_image_data)} bytes)")
            except Exception as e:
                main.logger.warning(f"Failed to download reference image for validation: {e}")
                reference_image_data = None
        
        story_pages = []
        consistency_results = []
        flagged_pages = []
        
        for i, page_text in enumerate(story_result['pages'], 1):
            main.logger.info(f"Generating scene image for page {i}/5...")
            # Use scene prompt from frontend if available, otherwise use None (will generate from params)
            scene_prompt = None
            if body.scene_prompts and len(body.scene_prompts) >= i:
                scene_prompt = body.scene_prompts[i - 1]  # i is 1-indexed, list is 0-indexed
                # Replace placeholder with actual page text
                scene_prompt = scene_prompt.replace(
                    f"[Page {i} text will be inserted here after story generation]",
                    page_text
                )
                main.logger.info(f"Using scene prompt from frontend for page {i} (with actual page text)")
            
            scene_url = main.generate_story_scene_image(
                story_page_text=page_text,
                page_number=i,
                character_name=body.character_name,
                character_type=body.character_type,
                story_world=body.story_world,
                reference_image_url=reference_image_url,
                scene_prompt=scene_prompt
            )
            # Convert string URL to HttpUrl if not empty, otherwise None
            scene_http_url = None
            scene_image_data = None
            consistency_validation = None
            
            if scene_url:
                try:
                    scene_http_url = HttpUrl(scene_url)
                    # Download scene image for consistency validation
                    try:
                        scene_image_data = main.download_image_from_url(scene_url)
                        main.logger.info(f"✅ Scene image downloaded for validation ({len(scene_image_data)} bytes)")
                    except Exception as e:
                        main.logger.warning(f"Failed to download scene image for validation: {e}")
                except Exception as e:
                    main.logger.warning(f"Invalid scene URL for page {i}: {e}")
                    scene_http_url = None
            
            # Perform character consistency validation if both images are available
            if reference_image_data and scene_image_data:
                main.logger.info(f"Performing character consistency validation for page {i}...")
                try:
                    consistency_validation = main.validate_character_consistency(
                        scene_image_data=scene_image_data,
                        reference_image_data=reference_image_data,
                        page_number=i,
                        timeout_seconds=15
                    )
                    consistency_results.append(consistency_validation)
                    
                    if consistency_validation.flagged:
                        flagged_pages.append(i)
                        main.logger.warning(f"⚠️ Page {i} flagged as INCONSISTENT (similarity: {consistency_validation.similarity_score:.3f})")
                    else:
                        main.logger.info(f"✅ Page {i} validated as CONSISTENT (similarity: {consistency_validation.similarity_score:.3f})")
                except Exception as e:
                    main.logger.error(f"Error during consistency validation for page {i}: {e}")
                    import traceback
                    main.logger.debug(f"Traceback: {traceback.format_exc()}")
            elif not reference_image_data:
                main.logger.info(f"Skipping consistency validation for page {i} - no reference image available")
            elif not scene_image_data:
                main.logger.warning(f"Skipping consistency validation for page {i} - scene image not available")
            
            story_pages.append(main.StoryPage(
                text=page_text, 
                scene=scene_http_url,
                consistency_validation=consistency_validation
            ))
        
        main.logger.info("All scene images generated successfully")
        
        # Generate audio for all story pages
        main.logger.info("Generating audio for story pages...")
        audio_urls = []
        audio_generator = None
        
        if main.supabase:
            try:
                audio_generator = AudioGenerator()
                if audio_generator.available:
                    # Generate audio for all pages
                    audio_data_list = audio_generator.generate_audio_for_story(
                        story_pages=story_result['pages'],
                        age_group=body.age_group,
                        timeout_per_page=60
                    )
                    
                    # Upload audio files to Supabase storage
                    for i, audio_data in enumerate(audio_data_list, 1):
                        if audio_data is None:
                            main.logger.warning(f"⚠️ No audio generated for page {i}, skipping upload")
                            audio_urls.append(None)
                            continue
                        
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"story_audio_page{i}_{timestamp}_{unique_id}.mp3"
                        
                        # Upload to Supabase storage (try audio bucket first, fallback to images)
                        storage_bucket = "audio"
                        audio_url = None
                        
                        try:
                            # Try audio bucket first
                            try:
                                response = main.supabase.storage.from_(storage_bucket).upload(
                                    filename,
                                    audio_data,
                                    {
                                        'content-type': 'audio/mpeg',
                                        'upsert': 'true'
                                    }
                                )
                            except Exception as e:
                                # If audio bucket doesn't exist, use images bucket
                                main.logger.warning(f"Audio bucket not found, using images bucket: {e}")
                                storage_bucket = "images"
                                response = main.supabase.storage.from_(storage_bucket).upload(
                                    filename,
                                    audio_data,
                                    {
                                        'content-type': 'audio/mpeg',
                                        'upsert': 'true'
                                    }
                                )
                            
                            if hasattr(response, 'full_path') and response.full_path:
                                audio_url = main.supabase.storage.from_(storage_bucket).get_public_url(filename)
                                audio_urls.append(audio_url)
                                main.logger.info(f"✅ Uploaded audio for page {i}: {audio_url}")
                            else:
                                main.logger.error(f"❌ Failed to upload audio for page {i}: Unexpected response")
                                audio_urls.append(None)
                        except Exception as e:
                            main.logger.error(f"❌ Error uploading audio for page {i} to Supabase: {e}")
                            audio_urls.append(None)
                        
                    successful_uploads = sum(1 for url in audio_urls if url is not None)
                    if successful_uploads > 0:
                        main.logger.info(f"✅ Generated and uploaded {successful_uploads}/5 audio files")
                    else:
                        main.logger.warning("⚠️ Failed to generate/upload any audio files")
                    
                    # Update StoryPage objects with audio URLs (recreate since Pydantic models are immutable)
                    updated_story_pages = []
                    for idx, page in enumerate(story_pages):
                        audio_http_url = None
                        if idx < len(audio_urls) and audio_urls[idx]:
                            try:
                                audio_http_url = HttpUrl(audio_urls[idx])
                            except Exception as e:
                                main.logger.warning(f"Failed to create HttpUrl for audio on page {idx + 1}: {e}")
                        
                        updated_story_pages.append(main.StoryPage(
                            text=page.text,
                            scene=page.scene,
                            audio=audio_http_url,
                            consistency_validation=page.consistency_validation
                        ))
                    story_pages = updated_story_pages
                else:
                    main.logger.warning("⚠️ Audio generator not available. Install: pip install gtts>=2.5.0")
            except Exception as e:
                main.logger.error(f"Error during audio generation: {e}")
                import traceback
                main.logger.debug(f"Traceback: {traceback.format_exc()}")
        else:
            main.logger.warning("⚠️ Supabase not configured, skipping audio generation")
        
        # Create consistency summary
        consistency_summary = None
        if consistency_results:
            avg_score = sum(r.similarity_score for r in consistency_results) / len(consistency_results)
            min_score = min(r.similarity_score for r in consistency_results)
            max_score = max(r.similarity_score for r in consistency_results)
            total_validation_time = sum(r.validation_time_seconds for r in consistency_results)
            consistent_count = sum(1 for r in consistency_results if r.is_consistent)
            
            consistency_summary = {
                "total_pages_validated": len(consistency_results),
                "consistent_pages": consistent_count,
                "inconsistent_pages": len(consistency_results) - consistent_count,
                "flagged_pages": flagged_pages,
                "average_similarity_score": round(avg_score, 3),
                "min_similarity_score": round(min_score, 3),
                "max_similarity_score": round(max_score, 3),
                "total_validation_time_seconds": round(total_validation_time, 2),
                "average_validation_time_seconds": round(total_validation_time / len(consistency_results), 2),
                "all_consistent": len(flagged_pages) == 0
            }
            
            main.logger.info("=" * 60)
            main.logger.info("CHARACTER CONSISTENCY VALIDATION SUMMARY")
            main.logger.info("=" * 60)
            main.logger.info(f"Total pages validated: {consistency_summary['total_pages_validated']}")
            main.logger.info(f"Consistent pages: {consistency_summary['consistent_pages']}")
            main.logger.info(f"Inconsistent pages: {consistency_summary['inconsistent_pages']}")
            if flagged_pages:
                main.logger.warning(f"⚠️ Flagged pages (inconsistent): {flagged_pages}")
            main.logger.info(f"Average similarity score: {avg_score:.3f}")
            main.logger.info(f"Score range: {min_score:.3f} - {max_score:.3f}")
            main.logger.info(f"Total validation time: {total_validation_time:.2f}s")
            main.logger.info(f"Average validation time per page: {total_validation_time / len(consistency_results):.2f}s")
            main.logger.info("=" * 60)
        
        # Story saving is now handled on the frontend
        main.logger.info("Story generation completed. Frontend will handle saving to database.")
        
        return main.StoryResponse(
            success=True,
            pages=story_pages,
            full_story=story_result['full_story'],
            word_count=story_result['word_count'],
            page_word_counts=story_result['page_word_counts'],
            consistency_summary=consistency_summary,
            audio_urls=audio_urls if audio_urls else None,
            dedication_image_url=dedication_image_url
        )
        
    except ValueError as e:
        main.logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        main.logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in generate_story_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.get("/api/books/{book_id}/pdf")
@limiter.limit("10/minute")
async def download_book_pdf(
    request: Request,
    book_id: int,
    authorization: Optional[str] = Header(None)
):
    """
    Download PDF for a book/story with purchase verification
    
    Args:
        book_id: Story/Book ID
        authorization: Bearer token (required for purchase verification)
    
    Returns:
        PDF file stream
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(status_code=500, detail="Storage service not available")
        
        # Extract user ID from authorization header
        user_id = main.extract_user_from_token(authorization)
        
        # In production, require authentication
        if main.IS_PRODUCTION and not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to download PDF"
            )
        
        # Get story/book information
        story_response = main.supabase.table("stories").select("*").eq("id", book_id).execute()
        
        if not story_response.data or len(story_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Book {book_id} not found")
        
        story = story_response.data[0]
        pdf_url = story.get("pdf_url")
        
        if not pdf_url:
            raise HTTPException(
                status_code=404,
                detail=f"PDF not available for book {book_id}. PDF may still be generating."
            )
        
        # Verify purchase before allowing download
        if not main.verify_purchase(book_id, user_id):
            raise HTTPException(
                status_code=403,
                detail="Purchase verification failed. Please purchase this book to download the PDF."
            )
        
        # Download PDF from storage
        main.logger.info(f"Downloading PDF from: {pdf_url}")
        
        # Extract filename from URL or generate one
        filename = pdf_url.split("/")[-1].split("?")[0] or f"book_{book_id}.pdf"
        
        # Download PDF bytes
        pdf_response = requests.get(pdf_url, timeout=30)
        pdf_response.raise_for_status()
        pdf_bytes = pdf_response.content
        
        # Return PDF as streaming response
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except HTTPException as e:
        raise e
    except requests.exceptions.RequestException as e:
        main.logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        main.logger.error(f"Unexpected error in download_book_pdf: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/api/books/{book_id}/generate-pdf")
@limiter.limit("10/minute")
async def generate_book_pdf(request: Request, book_id: str):
    """
    Generate PDF on-demand for a book/story
    
    This endpoint generates a PDF from the story data and uploads it to Supabase storage.
    Returns the PDF URL for download.
    """
    import main  # Import here to avoid circular import
    try:
        start_time = time.time()
        main.logger.info(f"Generating PDF on-demand for book {book_id}")
        
        if not main.supabase:
            raise HTTPException(status_code=500, detail="Storage service not available")
        
        # Try uid first, then fallback to id
        story_response = main.supabase.table("stories").select("*").eq("uid", book_id).execute()
        
        # If no result with uid, try id (in case uid doesn't exist in database)
        if not story_response.data or len(story_response.data) == 0:
            main.logger.info(f"No story found with uid={book_id}, trying id...")
            try:
                # Try to convert to integer for id lookup
                book_id_int = int(book_id)
                story_response = main.supabase.table("stories").select("*").eq("id", book_id_int).execute()
            except (ValueError, TypeError):
                main.logger.warning(f"Could not convert {book_id} to integer for id lookup")
        
        if not story_response.data or len(story_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Book {book_id} not found (tried both uid and id)")
        
        story = story_response.data[0]
        
        # Check if PDF already exists
        if story.get("pdf_url"):
            main.logger.info(f"PDF already exists for book {book_id}: {story.get('pdf_url')}")
            return {
                "success": True,
                "pdf_url": story.get("pdf_url"),
                "message": "PDF already generated"
            }
        
        # Prepare data for PDF generation
        story_title = story.get("story_title") or "Untitled Story"
        story_cover = story.get("story_cover")
        scene_images = story.get("scene_images")
        
        # Check if we have at least cover or scene images
        if not story_cover and (not scene_images or len(scene_images) == 0):
            raise HTTPException(
                status_code=400,
                detail="No cover image or scene images found. Cannot generate PDF without images."
            )
        
        # Generate 6-page PDF: cover + up to 5 scene images
        main.logger.info(f"Generating PDF with cover and {len(scene_images)} scene images")
        
        output_buffer = BytesIO()
        success = create_book_pdf_with_cover(
            story_title=story_title,
            story_cover_url=story_cover,
            scene_urls=scene_images,  # Up to 5 scene images will be used
            output_buffer=output_buffer
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")
        
        pdf_bytes = output_buffer.getvalue()
        
        # Upload PDF to Supabase storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"book_{book_id}_{timestamp}_{unique_id}.pdf"
        
        main.logger.info(f"Uploading PDF to Supabase storage: {filename}")
        
        # Upload to 'pdfs' bucket, fallback to 'images' bucket
        storage_bucket = "pdfs"
        pdf_url = None
        
        try:
            response = main.supabase.storage.from_(storage_bucket).upload(
                filename,
                pdf_bytes,
                {
                    'content-type': 'application/pdf',
                    'upsert': 'true'
                }
            )
        except Exception as e:
            # Fallback to images bucket if pdfs bucket doesn't exist
            main.logger.warning(f"PDF bucket not found, using images bucket: {e}")
            storage_bucket = "images"
            response = main.supabase.storage.from_(storage_bucket).upload(
                filename,
                pdf_bytes,
                {
                    'content-type': 'application/pdf',
                    'upsert': 'true'
                }
            )
        
        if hasattr(response, 'full_path') and response.full_path:
            pdf_url = main.supabase.storage.from_(storage_bucket).get_public_url(filename)
            main.logger.info(f"✅ PDF uploaded successfully: {pdf_url}")
        else:
            raise HTTPException(status_code=500, detail="Failed to upload PDF to storage")
        
        # Update story record with PDF URL
        update_response = main.supabase.table("stories").update({"pdf_url": pdf_url}).eq("uid", book_id).execute()
        
        if not update_response.data:
            main.logger.warning(f"Failed to update story {book_id} with PDF URL")
        
        elapsed = time.time() - start_time
        main.logger.info(f"✅ PDF generated and uploaded successfully in {elapsed:.2f} seconds")
        
        return {
            "success": True,
            "pdf_url": pdf_url,
            "message": "PDF generated successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error generating PDF: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")


@router.post("/api/books/{book_id}/purchase")
@limiter.limit("20/minute")
async def record_book_purchase(
    request: Request,
    book_id: int,
    user_id: Optional[str] = None,
    transaction_id: Optional[str] = None,
    amount_paid: Optional[float] = None,
    payment_method: Optional[str] = None
):
    """
    Record a book purchase (for purchase verification)
    
    This endpoint should be called after a successful payment
    """
    try:
        if not main.supabase:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Check if purchase already exists
        existing = main.supabase.table("book_purchases").select("*").eq("story_id", book_id).eq("user_id", user_id).execute()
        
        if existing.data and len(existing.data) > 0:
            main.logger.info(f"Purchase already exists for story {book_id}, user {user_id}")
            return {
                "success": True,
                "message": "Purchase already recorded",
                "purchase_id": existing.data[0]["id"]
            }
        
        # Create new purchase record
        purchase_data = {
            "story_id": book_id,
            "user_id": user_id,
            "purchase_status": "completed",
            "transaction_id": transaction_id,
            "amount_paid": amount_paid,
            "payment_method": payment_method or "free"
        }
        
        response = main.supabase.table("book_purchases").insert(purchase_data).execute()
        
        if response.data:
            main.logger.info(f"Purchase recorded for story {book_id}, user {user_id}")
            return {
                "success": True,
                "message": "Purchase recorded successfully",
                "purchase_id": response.data[0]["id"]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record purchase")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error recording purchase: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording purchase: {str(e)}")


@router.post("/api/search-game-results")
@limiter.limit("20/minute")
async def save_search_game_results(request: Request, body: SearchGameResultRequest):
    """
    Save search game results for an interactive search story
    
    This endpoint stores the results of a completed search game, including
    scene-by-scene results and summary statistics.
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        if not body.character_id:
            raise HTTPException(status_code=400, detail="character_id is required")
        
        # Convert scene results to JSON format for storage
        result_array = []
        for scene_result in body.result:
            result_array.append({
                "scene_index": scene_result.scene_index,
                "scene_title": scene_result.scene_title,
                "time": scene_result.time,
                "hint_used": scene_result.hint_used,
                "star_rate": scene_result.star_rate
            })
        
        # Prepare data for insertion
        result_data = {
            "character_id": body.character_id,
            "story_id": body.story_id,
            "result": result_array,  # This will be stored as JSONB
            "total_time": body.total_time,
            "avg_stars": float(body.avg_stars),
            "hints_used": body.hints_used,
            "best_scene": body.best_scene,
            "user_id": body.user_id,
            "child_profile_id": body.child_profile_id
        }
        
        # Insert into database
        response = main.supabase.table("search_game_results").insert(result_data).execute()
        
        if response.data:
            main.logger.info(f"Search game results saved for character {body.character_id}")
            
            # Update hints count in stories table if story_id is provided
            if body.story_id and body.hints_used > 0:
                try:
                    # Get current hints count
                    story_response = main.supabase.table("stories").select("hints").eq("id", body.story_id).execute()
                    
                    if story_response.data and len(story_response.data) > 0:
                        current_hints = story_response.data[0].get("hints", 3)
                        if current_hints is not None:
                            # Decrement hints by the number used, but don't go below 0
                            new_hints = max(0, current_hints - body.hints_used)
                            
                            # Update stories table
                            update_response = main.supabase.table("stories").update({
                                "hints": new_hints
                            }).eq("id", body.story_id).execute()
                            
                            main.logger.info(f"Updated hints count for story {body.story_id}: {current_hints} -> {new_hints} (used {body.hints_used})")
                        else:
                            main.logger.warning(f"Story {body.story_id} has NULL hints, skipping update")
                    else:
                        main.logger.warning(f"Story {body.story_id} not found, skipping hints update")
                except Exception as e:
                    # Don't fail the entire operation if hints update fails
                    main.logger.error(f"Error updating hints count for story {body.story_id}: {e}")
            
            return {
                "success": True,
                "message": "Search game results saved successfully",
                "result_id": response.data[0]["id"]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save search game results")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error saving search game results: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error saving search game results: {str(e)}")
