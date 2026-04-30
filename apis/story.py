"""
Story API routes
"""
from fastapi import APIRouter, HTTPException, Request, Header, Body
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime
from typing import Optional, TYPE_CHECKING, List, Dict, Any
import uuid
import requests
import time
import json
from pydantic import HttpUrl, BaseModel
from rate_limiter import limiter
from story_lib import generate_story
from audio_generator import AudioGenerator
from pdf_generator import create_book_pdf_with_cover
from .models import StoryRequest, StoryGenerateWithProgressRequest, StoryScenesRequest, StoryAudioRequest, SearchGameResultRequest, StoryTitlesRequest, SaveStoryDraftRequest, SetStoryGeneratingRequest


class CheckPointRequest(BaseModel):
    templateId: Optional[str] = None
    storyUid: Optional[str] = None
    pageNumber: int
    x: float
    y: float


if TYPE_CHECKING:
    import main

router = APIRouter()
PROMPT_DOCUMENTS_TABLE = "ai_prompt_documents"
PROMPT_IMAGE_FILE_KEY = "prompt_image"


def _load_prompt_image_content() -> Dict[str, Any]:
    try:
        import main
        if main.supabase:
            response = (
                main.supabase
                .table(PROMPT_DOCUMENTS_TABLE)
                .select("content")
                .eq("file_key", PROMPT_IMAGE_FILE_KEY)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            content = rows[0].get("content") if rows else None
            if isinstance(content, dict):
                return content
    except Exception:
        pass

    from pathlib import Path
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "drawtopia_frontend"
        / "src"
        / "lib"
        / "prompt_image.json"
    )
    if not prompt_path.exists():
        return {}
    with prompt_path.open("r", encoding="utf-8") as prompt_file:
        return json.load(prompt_file)


def _require_generation_prompt(value: Optional[str], field_name: str) -> str:
    prompt = (value or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return prompt


def _require_scene_prompts(scene_prompts: Optional[List[str]], count: int) -> List[str]:
    if not scene_prompts or len(scene_prompts) < count:
        raise HTTPException(status_code=400, detail=f"scene_prompts must include {count} prompts")

    prompts = [prompt.strip() for prompt in scene_prompts[:count]]
    if any(not prompt for prompt in prompts):
        raise HTTPException(status_code=400, detail="scene_prompts cannot contain empty prompts")
    return prompts


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
            "character_image_url": str(body.character_image_url) if body.character_image_url else None,
            "story_text_prompt": getattr(body, "story_text_prompt", None),
            "scene_prompts": getattr(body, "scene_prompts", None),
        }

        if body.job_type == "story_adventure":
            _require_generation_prompt(job_data.get("story_text_prompt"), "story_text_prompt")
            _require_scene_prompts(job_data.get("scene_prompts"), 5)
        elif body.job_type == "interactive_search":
            _require_scene_prompts(job_data.get("scene_prompts"), 2)
        
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


@router.post("/api/game/check-point")
@limiter.limit("120/minute")
async def check_game_point(request: Request, body: CheckPointRequest):
    """
    Check whether a normalized (x,y) point falls within the page-specific position subset.
    Positions can come from:
      - `book_templates.positions` (using `templateId`)
      - `stories.positions` (using `storyUid`)
      - `stories.template_id` -> `book_templates.positions` fallback
    Returns JSON: { success: True, hit: int }
    Where `hit` is:
      - 0 if not found
      - 1..4 indicating which character point subset was hit
    """
    import main
    try:
        if not main.supabase:
            raise HTTPException(status_code=500, detail="Database service not available")

        if not body.templateId and not body.storyUid:
            raise HTTPException(status_code=400, detail="templateId or storyUid is required")
        if body.pageNumber < 1:
            raise HTTPException(status_code=400, detail="pageNumber must be >= 1")

        # Clamp and validate coordinates
        try:
            x = float(body.x)
            y = float(body.y)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid x/y values")

        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            raise HTTPException(status_code=400, detail="x and y must be between 0.0 and 1.0")

        def extract_positions(row: Any) -> Optional[List[Dict[str, float]]]:
            if isinstance(row, dict):
                p = row.get("positions")
                if isinstance(p, list) and len(p) > 0:
                    return p
            return None

        positions: Optional[List[Dict[str, float]]] = None
        story_row = None

        # Story ID is commonly passed by the client.
        # Accept either storyUid field or templateId field carrying story UID/ID.
        story_identifier = body.storyUid or body.templateId

        # 1) Primary: read story first, then use story.template_id -> book_templates.positions
        if story_identifier:
            try:
                story_resp = (
                    main.supabase.table("stories")
                    .select("id,uid,template_id,positions")
                    .eq("uid", str(story_identifier))
                    .limit(1)
                    .execute()
                )
                story_row = story_resp.data[0] if story_resp and getattr(story_resp, "data", None) and len(story_resp.data) > 0 else None
            except Exception:
                story_row = None

            # Optional fallback: story numeric id
            if not story_row:
                try:
                    numeric_story_id = int(str(story_identifier))
                    story_resp = (
                        main.supabase.table("stories")
                        .select("id,uid,template_id,positions")
                        .eq("id", numeric_story_id)
                        .limit(1)
                        .execute()
                    )
                    story_row = story_resp.data[0] if story_resp and getattr(story_resp, "data", None) and len(story_resp.data) > 0 else None
                except Exception:
                    story_row = story_row

        # Keep support for stories.positions when present.
        if isinstance(story_row, dict):
            positions = extract_positions(story_row)

        if not positions and isinstance(story_row, dict) and story_row.get("template_id"):
            try:
                tmpl_resp = (
                    main.supabase.table("book_templates")
                    .select("id,positions")
                    .eq("id", str(story_row.get("template_id")))
                    .limit(1)
                    .execute()
                )
                tmpl_row = tmpl_resp.data[0] if tmpl_resp and getattr(tmpl_resp, "data", None) and len(tmpl_resp.data) > 0 else None
                positions = extract_positions(tmpl_row)
            except Exception:
                positions = None

        # 2) Backward compatibility: if caller truly sends templateId, query template directly
        if not positions and body.templateId:
            try:
                tmpl_resp = (
                    main.supabase.table("book_templates")
                    .select("id,positions")
                    .eq("id", body.templateId)
                    .limit(1)
                    .execute()
                )
                tmpl_row = tmpl_resp.data[0] if tmpl_resp and getattr(tmpl_resp, "data", None) and len(tmpl_resp.data) > 0 else None
                positions = extract_positions(tmpl_row)
            except Exception:
                positions = None

        # If no positions found for the template, return hit=0
        if not positions or not isinstance(positions, list) or len(positions) == 0:
            return {"success": True, "hit": 0}

        # Resolve the current page subset.
        # Common layouts:
        # - 16 points across pages 3..6
        # - 16 points across pages 1..4
        # - 4 points total (single-page layout)
        page_positions: List[Dict[str, float]] = []
        total = len(positions)
        if total >= 16:
            # Try page 3..6 mapping first
            if 3 <= body.pageNumber <= 6:
                start_idx = (body.pageNumber - 3) * 4
            else:
                # Then page 1..4 mapping
                start_idx = (body.pageNumber - 1) * 4
            if start_idx < 0 or start_idx + 4 > total:
                start_idx = 0
            page_positions = positions[start_idx:start_idx + 4]
        elif total == 4:
            page_positions = positions
        else:
            # Graceful handling for non-standard data: pick page-sized chunk if possible.
            start_idx = (max(body.pageNumber, 1) - 1) * 4
            if start_idx < total:
                page_positions = positions[start_idx:start_idx + 4]
            if len(page_positions) == 0:
                page_positions = positions[:4]

        if len(page_positions) == 0:
            return {"success": True, "hit": 0}

        # Circle radius in normalized coordinates.
        R = 0.05

        hit_index = 0
        for idx, coord in enumerate(page_positions):
            try:
                cx = float(coord.get('x'))
                cy = float(coord.get('y'))
            except Exception:
                continue
            # Compare center-to-pointer distance in the same normalized coordinate space.
            dx = cx - x
            dy = cy - y
            if (dx*dx + dy*dy) <= (R * R):
                hit_index = idx + 1
                break

        return {"success": True, "hit": hit_index}

    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error in check_game_point: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking point: {str(e)}")


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
        parent_id: Optional parent user ID to filter stories by user_id
    
    Returns:
        List of all story/book data, optionally filtered by parent, with child profile information
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        # If parent_id is provided, filter stories by user_id
        if parent_id:
            # Step 1: Get all stories filtered by user_id from stories table
            stories_response = main.supabase.table("stories").select("*").eq("user_id", parent_id).order("created_at", desc=True).execute()
            
            if stories_response.data is None:
                main.logger.warning("No stories found or query returned None")
                return []
            
            if len(stories_response.data) == 0:
                main.logger.info(f"No stories found for user {parent_id}")
                return []
            
            # Step 2: Get unique child_profile_ids from the stories
            child_profile_ids = list(set([story["child_profile_id"] for story in stories_response.data if story.get("child_profile_id")]))
            
            # Step 3: Fetch child profile information for all child_profile_ids
            child_profiles_map = {}
            if child_profile_ids:
                child_profiles_response = main.supabase.table("child_profiles").select("*").in_("id", child_profile_ids).execute()
                
                if child_profiles_response.data:
                    # Create a map for quick lookup
                    child_profiles_map = {cp["id"]: cp for cp in child_profiles_response.data}
            
            # Step 4: Get user data for parent
            user_response = main.supabase.table("users").select("*").eq("id", parent_id).execute()
            user_data = user_response.data[0] if user_response.data and len(user_response.data) > 0 else None
            
            # Step 5: Merge child profile data with stories
            stories_with_child_data = []
            for story in stories_response.data:
                child_profile_id = story.get("child_profile_id")
                child_profile = child_profiles_map.get(child_profile_id) if child_profile_id else None
                
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
    Also deletes associated images from storage (except character images and enhancement images)
    
    Args:
        id: Book ID (integer) or UID (string)
    
    Returns:
        Success message with deleted book information
    """
    import main  # Import here to avoid circular import
    from storage_utils import delete_story_images, collect_book_template_image_urls
    
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
        character_id = book_data.get("character_id")
        book_user_id = book_data.get("user_id")
        
        # Delete associated images from storage (exclude character images)
        main.logger.info(f"Deleting images for story {id} from storage...")
        try:
            protected_template_urls = set()
            try:
                protected_template_urls = collect_book_template_image_urls(main.supabase)
            except Exception as template_lookup_error:
                main.logger.warning(
                    f"Could not load shared template URLs for deletion protection: {template_lookup_error}"
                )

            deletion_result = delete_story_images(
                main.supabase, 
                book_data, 
                exclude_character_images=True,  # Keep character image and enhancement images
                protected_urls=protected_template_urls,
            )
            main.logger.info(f"Image deletion result: {deletion_result['success']} succeeded, {deletion_result['errors']} failed")
        except Exception as storage_error:
            # Log but don't fail the deletion if storage cleanup fails
            main.logger.error(f"Error deleting images from storage: {storage_error}")
        
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

        # Delete related character row if this story has one
        deleted_character = False
        if character_id:
            try:
                delete_character_query = main.supabase.table("characters").delete().eq("id", character_id)
                if book_user_id:
                    delete_character_query = delete_character_query.eq("user_id", book_user_id)

                delete_character_response = delete_character_query.execute()
                deleted_character = bool(delete_character_response.data and len(delete_character_response.data) > 0)

                if deleted_character:
                    main.logger.info(
                        f"Deleted related character id={character_id} for story id={book_id or book_uid}"
                    )
                else:
                    main.logger.warning(
                        f"No related character deleted for story id={book_id or book_uid}; "
                        f"character_id={character_id} may not exist or may already be deleted"
                    )
            except Exception as character_delete_error:
                # Log but do not fail story deletion if related character cleanup fails
                main.logger.error(
                    f"Failed to delete related character {character_id} for story id={book_id or book_uid}: "
                    f"{character_delete_error}"
                )
        
        return {
            "success": True,
            "message": f"Book {id} deleted successfully",
            "deleted_book": {
                "id": book_id,
                "uid": book_uid,
                "title": book_data.get("story_title", "Unknown")
            },
            "deleted_related_character": deleted_character,
            "deleted_character_id": character_id if deleted_character else None,
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error deleting book: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error deleting book: {str(e)}")


@router.post("/api/story/save-draft")
@limiter.limit("30/minute")
async def save_story_draft(request: Request, body: SaveStoryDraftRequest):
    """
    Save current story data as draft to Supabase (story-preview page).
    Called when user clicks "Save as Draft" or "Generate and Preview Story" before any other action.
    Story state (status) is set to "draft".
    Returns the created story with uid and id for use on the loading page.
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        # If client sent an existing story uid, check if it exists — then update instead of insert
        existing_uid = (body.story_uid or "").strip() or None
        if existing_uid:
            check = main.supabase.table("stories").select("uid").eq("uid", existing_uid).execute()
            if check.data and len(check.data) > 0:
                update_data = {
                    "user_id": body.user_id,
                    "child_profile_id": body.child_profile_id,
                    "character_id": body.character_id,
                    "character_name": body.character_name,
                    "character_type": body.character_type,
                    "special_ability": body.special_ability or "",
                    "character_style": body.character_style,
                    "story_world": body.story_world,
                    "adventure_type": body.adventure_type,
                    "difficulty": body.difficulty,
                    "original_image_url": body.original_image_url,
                    "enhanced_images": body.enhanced_images or [],
                    "story_title": body.story_title,
                    "story_cover": body.story_cover,
                    "cover_design": body.cover_design,
                    "template_id": body.template_id,
                    "status": "draft",
                    "story_type": body.story_type or "story",
                    "gift_id": body.gift_id,
                    "purchased": body.purchased or False,
                }
                # If this is an interactive/search story, try to populate character_for_finding
                try:
                    stype = (body.story_type or "").strip().lower()
                    if stype in ("search", "interactive", "search-and-find"):
                        # Normalize style/world
                        style = (body.character_style or "").strip().lower().replace("_", "-").replace(" ", "-")
                        world = (body.story_world or "").strip().lower()
                        q = main.supabase.table("book_templates").select("character_for_finding,story_format,story_style,story_world")
                        if world:
                            q = q.eq("story_world", world)
                        if style:
                            q = q.eq("story_style", style)
                        resp = q.execute()
                        rows = resp.data or []
                        urls = []
                        for r in rows:
                            fmt = (r.get("story_format") or "").strip().lower().replace("-", "_")
                            # Only include templates marked as interactive_story
                            if fmt != "interactive_story":
                                continue
                            cff = r.get("character_for_finding")
                            if isinstance(cff, list):
                                for u in cff:
                                    if isinstance(u, str) and u.strip() and u not in urls:
                                        urls.append(u)
                            elif isinstance(cff, str) and cff.strip() and cff not in urls:
                                urls.append(cff)
                        if urls:
                            update_data["character_for_finding"] = urls
                except Exception as e:
                    main.logger.warning(f"Could not populate character_for_finding for draft update: {e}")
                main.supabase.table("stories").update(update_data).eq("uid", existing_uid).execute()
                story_response = main.supabase.table("stories").select("*").eq("uid", existing_uid).execute()
                story = story_response.data[0] if story_response.data else None
                if story:
                    main.logger.info(f"Story draft updated (existing): uid={story.get('uid')}, id={story.get('id')}")
                    return story
        # New story or existing uid not found: insert
        story_uid = str(uuid.uuid4())
        insert_data = {
            "uid": story_uid,
            "user_id": body.user_id,
            "child_profile_id": body.child_profile_id,
            "character_id": body.character_id,
            "character_name": body.character_name,
            "character_type": body.character_type,
            "special_ability": body.special_ability or "",
            "character_style": body.character_style,
            "story_world": body.story_world,
            "adventure_type": body.adventure_type,
            "difficulty": body.difficulty,
            "original_image_url": body.original_image_url,
            "enhanced_images": body.enhanced_images or [],
            "story_title": body.story_title,
            "story_cover": body.story_cover,
            "cover_design": body.cover_design,
            "template_id": body.template_id,
            "story_content": None,
            "scene_images": [],
            "audio_url": [],
            "dedication_text": None,
            "dedication_image": None,
            "status": "draft",
            "story_type": body.story_type or "story",
            # character_for_finding will be populated for interactive/search stories below
            "hints": None,
            "gift_id": body.gift_id,
            "purchased": body.purchased or False,
        }
        # Populate character_for_finding for interactive/search stories when inserting
        try:
            stype = (body.story_type or "").strip().lower()
            if stype in ("search", "interactive", "search-and-find"):
                style = (body.character_style or "").strip().lower().replace("_", "-").replace(" ", "-")
                world = (body.story_world or "").strip().lower()
                q = main.supabase.table("book_templates").select("character_for_finding,story_format,story_style,story_world")
                if world:
                    q = q.eq("story_world", world)
                if style:
                    q = q.eq("story_style", style)
                resp = q.execute()
                rows = resp.data or []
                urls = []
                for r in rows:
                    fmt = (r.get("story_format") or "").strip().lower().replace("-", "_")
                    if fmt != "interactive_story":
                        continue
                    cff = r.get("character_for_finding")
                    if isinstance(cff, list):
                        for u in cff:
                            if isinstance(u, str) and u.strip() and u not in urls:
                                urls.append(u)
                    elif isinstance(cff, str) and cff.strip() and cff not in urls:
                        urls.append(cff)
                if urls:
                    insert_data["character_for_finding"] = urls
        except Exception as e:
            main.logger.warning(f"Could not populate character_for_finding for draft insert: {e}")
        response = main.supabase.table("stories").insert(insert_data).execute()
        if not response.data or len(response.data) == 0:
            # Some Supabase clients don't return inserted row; fetch by uid
            story_response = main.supabase.table("stories").select("*").eq("uid", story_uid).execute()
            if not story_response.data or len(story_response.data) == 0:
                raise HTTPException(status_code=500, detail="Failed to create story draft")
            story = story_response.data[0]
        else:
            story = response.data[0]
        main.logger.info(f"Story draft saved: uid={story.get('uid')}, id={story.get('id')}")
        return story
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error saving story draft: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error saving story draft: {str(e)}")


@router.post("/api/books/update-state")
@limiter.limit("60/minute")
async def set_story_generating(request: Request, body: SetStoryGeneratingRequest):
    """
    Set story status to "generating" in Supabase before generation starts.
    Called from the loading page before generating story text/images.
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        story_id = body.id
        if not story_id:
            raise HTTPException(status_code=400, detail="Story id is required")
        
        main.supabase.table("stories").update({"status": "generating"}).eq("uid", story_id).execute()
        main.logger.info(f"Story {story_id} status set to generating")
        return {"success": True, "message": "Story status set to generating"}
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error setting story generating: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error setting story generating: {str(e)}")


@router.post("/story/generate-text")
@limiter.limit("10/minute")
async def generate_story_text_endpoint(request: Request, body: StoryRequest):
    """Generate story text only (5-page children's story)"""
    import main  # Import here to avoid circular import
    try:
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if body.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {body.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )
        
        main.logger.info(f"Generating story text for character: {body.character_name}")
        
        if not main.OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )

        story_text_prompt = _require_generation_prompt(body.story_text_prompt, "story_text_prompt")

        main.logger.info("Generating story with OpenAI GPT-4...")
        story_result = generate_story(
            character_name=body.character_name,
            character_type=body.character_type,
            special_ability=body.special_ability,
            age_group=body.age_group,
            story_world=body.story_world,
            adventure_type=body.adventure_type,
            occasion_theme=body.occasion_theme,
            use_api=True,
            api_key=main.OPENAI_API_KEY,
            story_text_prompt=story_text_prompt
        )
        
        main.logger.info(f"Story text generated successfully. Word count: {story_result['word_count']}")
        
        return {
            "success": True,
            "pages": story_result["pages"],
            "full_story": story_result["full_story"],
            "word_count": story_result["word_count"],
            "page_word_counts": story_result["page_word_counts"],
        }
        
    except ValueError as e:
        main.logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in generate_story_text_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/story/generate")
@limiter.limit("10/minute")
async def generate_story_full_endpoint(request: Request, body: StoryRequest):
    """Generate story text, audio, and scene images in one request."""
    import main  # Import here to avoid circular import
    try:
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if body.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {body.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )

        main.logger.info(f"Generating full story (text + audio + scenes) for character: {body.character_name}")

        if not main.OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )

        story_text_prompt = _require_generation_prompt(body.story_text_prompt, "story_text_prompt")

        # ——— Step 1: Generate story text ———
        main.logger.info("Step 1/3: Generating story text...")
        story_result = generate_story(
            character_name=body.character_name,
            character_type=body.character_type,
            special_ability=body.special_ability,
            age_group=body.age_group,
            story_world=body.story_world,
            adventure_type=body.adventure_type,
            occasion_theme=body.occasion_theme,
            use_api=True,
            api_key=main.OPENAI_API_KEY,
            story_text_prompt=story_text_prompt
        )
        pages_text = story_result.get("pages") or []
        main.logger.info(f"Story text generated. Word count: {story_result.get('word_count', 0)}")

        if not pages_text:
            raise HTTPException(status_code=500, detail="No story pages generated")
        scene_prompts = _require_scene_prompts(body.scene_prompts, len(pages_text[:5]))

        # ——— Step 2: Generate audio ———
        main.logger.info("Step 2/3: Generating story audio...")
        audio_urls = []
        if main.supabase:
            try:
                audio_generator = AudioGenerator()
                if audio_generator.available:
                    audio_data_list = audio_generator.generate_audio_for_story(
                        story_pages=pages_text,
                        age_group=body.age_group,
                        timeout_per_page=60
                    )
                    for i, audio_data in enumerate(audio_data_list, 1):
                        if audio_data is None:
                            audio_urls.append(None)
                            continue
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"story_audio_page{i}_{timestamp}_{unique_id}.mp3"
                        storage_bucket = "audio"
                        audio_url = None
                        try:
                            try:
                                main.supabase.storage.from_(storage_bucket).upload(
                                    filename, audio_data, {"content-type": "audio/mpeg", "upsert": "true"}
                                )
                            except Exception:
                                storage_bucket = "images"
                                main.supabase.storage.from_(storage_bucket).upload(
                                    filename, audio_data, {"content-type": "audio/mpeg", "upsert": "true"}
                                )
                            if storage_bucket:
                                audio_url = main.supabase.storage.from_(storage_bucket).get_public_url(filename)
                        except Exception as e:
                            main.logger.error(f"Error uploading audio for page {i}: {e}")
                        audio_urls.append(audio_url)
            except Exception as e:
                main.logger.error(f"Error during audio generation: {e}")

        # ——— Step 3: Generate scene images ———
        main.logger.info("Step 3/3: Generating story scenes...")
        reference_image_url = str(body.character_image_url) if body.character_image_url else None
        reference_image_data = None
        if reference_image_url:
            try:
                reference_image_data = main.download_image_from_url(reference_image_url)
            except Exception as e:
                main.logger.warning(f"Failed to download reference image: {e}")

        story_pages_out = []
        consistency_results = []
        flagged_pages = []
        for i, page_text in enumerate(pages_text[:5], 1):
            main.logger.info(f"Generating scene image for page {i}/5...")
            scene_prompt = scene_prompts[i - 1]

            scene_url = main.generate_story_scene_image(
                story_page_text=page_text,
                page_number=i,
                character_name=body.character_name,
                character_type=body.character_type,
                story_world=body.story_world,
                reference_image_url=reference_image_url,
                scene_prompt=scene_prompt
            )
            scene_http_url = None
            scene_image_data = None
            consistency_validation = None
            if scene_url:
                try:
                    scene_http_url = HttpUrl(scene_url)
                    try:
                        scene_image_data = main.download_image_from_url(scene_url)
                    except Exception:
                        pass
                except Exception as e:
                    main.logger.warning(f"Invalid scene URL for page {i}: {e}")
            if reference_image_data and scene_image_data:
                try:
                    consistency_validation = main.validate_character_consistency(
                        scene_image_data=scene_image_data,
                        reference_image_data=reference_image_data,
                        page_number=i,
                        timeout_seconds=15,
                        scene_image_url=scene_url,
                        reference_image_url=reference_image_url,
                    )
                    consistency_results.append(consistency_validation)
                    if consistency_validation.flagged:
                        flagged_pages.append(i)
                except Exception as e:
                    main.logger.error(f"Error during consistency validation for page {i}: {e}")
            story_pages_out.append(main.StoryPage(
                text=page_text,
                scene=scene_http_url,
                consistency_validation=consistency_validation
            ))

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

        return {
            "success": True,
            "pages": [
                {
                    "text": p.text,
                    "scene": str(p.scene) if p.scene else None,
                    "consistency_validation": p.consistency_validation.model_dump() if p.consistency_validation is not None else None
                }
                for p in story_pages_out
            ],
            "audio_urls": audio_urls,
            "full_story": story_result.get("full_story"),
            "word_count": story_result.get("word_count"),
            "page_word_counts": story_result.get("page_word_counts"),
            "consistency_summary": consistency_summary,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in generate_story_full_endpoint: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


async def _send_progress(session_id: str, percentage: int) -> None:
    """Send progress percentage to client via WebSocket."""
    if main.story_progress_manager:
        await main.story_progress_manager.send_progress(session_id, percentage)


@router.post("/story/generate-with-progress")
@limiter.limit("10/minute")
async def generate_story_with_progress_endpoint(request: Request, body: StoryGenerateWithProgressRequest):
    """Generate story text, audio, and scene images; send percentage progress via WebSocket (session_id from ws/story-progress)."""
    import main  # Import here to avoid circular import
    session_id = body.session_id
    try:
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if body.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {body.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )

        if not main.OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )

        story_text_prompt = _require_generation_prompt(body.story_text_prompt, "story_text_prompt")

        await _send_progress(session_id, 2)
        main.logger.info(f"Generating full story (with progress) for character: {body.character_name}")

        # ——— Step 1: Generate story text ———
        main.logger.info("Step 1/3: Generating story text...")
        story_result = generate_story(
            character_name=body.character_name,
            character_type=body.character_type,
            special_ability=body.special_ability,
            age_group=body.age_group,
            story_world=body.story_world,
            adventure_type=body.adventure_type,
            occasion_theme=body.occasion_theme,
            use_api=True,
            api_key=main.OPENAI_API_KEY,
            story_text_prompt=story_text_prompt
        )
        pages_text = story_result.get("pages") or []
        main.logger.info(f"Story text generated. Word count: {story_result.get('word_count', 0)}")
        await _send_progress(session_id, 20)

        if not pages_text:
            raise HTTPException(status_code=500, detail="No story pages generated")
        scene_prompts = _require_scene_prompts(body.scene_prompts, len(pages_text[:5]))

        # ——— Step 2: Generate audio ———
        main.logger.info("Step 2/3: Generating story audio...")
        audio_urls = []
        if main.supabase:
            try:
                audio_generator = AudioGenerator()
                if audio_generator.available:
                    audio_data_list = audio_generator.generate_audio_for_story(
                        story_pages=pages_text,
                        age_group=body.age_group,
                        timeout_per_page=60
                    )
                    for i, audio_data in enumerate(audio_data_list, 1):
                        if audio_data is None:
                            audio_urls.append(None)
                            continue
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"story_audio_page{i}_{timestamp}_{unique_id}.mp3"
                        storage_bucket = "audio"
                        audio_url = None
                        try:
                            try:
                                main.supabase.storage.from_(storage_bucket).upload(
                                    filename, audio_data, {"content-type": "audio/mpeg", "upsert": "true"}
                                )
                            except Exception:
                                storage_bucket = "images"
                                main.supabase.storage.from_(storage_bucket).upload(
                                    filename, audio_data, {"content-type": "audio/mpeg", "upsert": "true"}
                                )
                            if storage_bucket:
                                audio_url = main.supabase.storage.from_(storage_bucket).get_public_url(filename)
                        except Exception as e:
                            main.logger.error(f"Error uploading audio for page {i}: {e}")
                        audio_urls.append(audio_url)
            except Exception as e:
                main.logger.error(f"Error during audio generation: {e}")
        await _send_progress(session_id, 40)

        # ——— Step 3: Generate scene images ———
        main.logger.info("Step 3/3: Generating story scenes...")
        await _send_progress(session_id, 45)
        reference_image_url = str(body.character_image_url) if body.character_image_url else None
        reference_image_data = None
        if reference_image_url:
            try:
                reference_image_data = main.download_image_from_url(reference_image_url)
            except Exception as e:
                main.logger.warning(f"Failed to download reference image: {e}")

        story_pages_out = []
        consistency_results = []
        flagged_pages = []
        for i, page_text in enumerate(pages_text[:5], 1):
            main.logger.info(f"Generating scene image for page {i}/5...")
            scene_prompt = scene_prompts[i - 1]

            scene_url = main.generate_story_scene_image(
                story_page_text=page_text,
                page_number=i,
                character_name=body.character_name,
                character_type=body.character_type,
                story_world=body.story_world,
                reference_image_url=reference_image_url,
                scene_prompt=scene_prompt
            )
            scene_http_url = None
            scene_image_data = None
            consistency_validation = None
            if scene_url:
                try:
                    scene_http_url = HttpUrl(scene_url)
                    try:
                        scene_image_data = main.download_image_from_url(scene_url)
                    except Exception:
                        pass
                except Exception as e:
                    main.logger.warning(f"Invalid scene URL for page {i}: {e}")
            if reference_image_data and scene_image_data:
                try:
                    consistency_validation = main.validate_character_consistency(
                        scene_image_data=scene_image_data,
                        reference_image_data=reference_image_data,
                        page_number=i,
                        timeout_seconds=15,
                        scene_image_url=scene_url,
                        reference_image_url=reference_image_url,
                    )
                    consistency_results.append(consistency_validation)
                    if consistency_validation.flagged:
                        flagged_pages.append(i)
                except Exception as e:
                    main.logger.error(f"Error during consistency validation for page {i}: {e}")
            story_pages_out.append(main.StoryPage(
                text=page_text,
                scene=scene_http_url,
                consistency_validation=consistency_validation
            ))
            await _send_progress(session_id, 45 + (i * 11))

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

        await _send_progress(session_id, 100)
        return {
            "success": True,
            "pages": [
                {
                    "text": p.text,
                    "scene": str(p.scene) if p.scene else None,
                    "consistency_validation": p.consistency_validation.model_dump() if p.consistency_validation is not None else None
                }
                for p in story_pages_out
            ],
            "audio_urls": audio_urls,
            "full_story": story_result.get("full_story"),
            "word_count": story_result.get("word_count"),
            "page_word_counts": story_result.get("page_word_counts"),
            "consistency_summary": consistency_summary,
        }

    except HTTPException as e:
        await _send_progress(session_id, 100)
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in generate_story_with_progress_endpoint: {e}")
        await _send_progress(session_id, 100)
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/story/generate-titles")
@limiter.limit("5/minute")
async def generate_story_titles_endpoint(request: Request, body: StoryTitlesRequest):
    """Generate story title suggestions using OpenAI with the frontend-supplied prompt."""
    import main  # Import here to avoid circular import
    try:
        if not main.OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )

        main.logger.info(f"Generating story titles for character: {body.character_name}")
        prompt = _require_generation_prompt(body.title_prompt, "title_prompt")

        from openai import OpenAI
        client = OpenAI(api_key=main.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=120,
            n=1
        )

        titles = []
        content = (response.choices[0].message.content or "").strip() if response.choices else ""
        if content:
            import re
            numbered_titles = re.findall(
                r"(?:^|\n)\s*\d+[\).\:-]\s*(.+?)(?=(?:\n\s*\d+[\).\:-])|\Z)",
                content,
                flags=re.DOTALL,
            )

            raw_titles = numbered_titles or content.splitlines()
            for raw_title in raw_titles:
                cleaned_title = re.sub(r"\s+", " ", raw_title).strip()
                cleaned_title = re.sub(r"^\d+[\).\:-]\s*", "", cleaned_title)
                cleaned_title = cleaned_title.strip().strip("**").strip('"').strip("'")
                if cleaned_title and cleaned_title not in titles:
                    titles.append(cleaned_title)
                if len(titles) == 3:
                    break

        if not titles:
            raise HTTPException(status_code=500, detail="No titles returned from OpenAI")

        main.logger.info(f"Generated story titles: {titles}")

        return {
            "success": True,
            "titles": titles
        }

    except json.JSONDecodeError as e:
        main.logger.error(f"Failed to parse OpenAI response as JSON: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse generated titles")
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error in generate_story_titles_endpoint: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/story/generate-scenes")
@limiter.limit("10/minute")
async def generate_story_scenes_endpoint(request: Request, body: StoryScenesRequest):
    """Generate scene images for story pages (dedication + 5 story pages)"""
    import main  # Import here to avoid circular import
    try:
        if not main.GEMINI_API_KEY or not main.gemini_client:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key not configured or client not initialized. Please set GEMINI_API_KEY environment variable."
            )
        
        main.logger.info(f"Generating story scenes for character: {body.character_name}")
        scene_prompts = _require_scene_prompts(body.scene_prompts, len(body.pages[:5]))
        
        # Generate scene images for each page
        main.logger.info("Generating scene images for each story page...")
        reference_image_url = str(body.character_image_url) if body.character_image_url else None
        
        reference_image_data = None
        if reference_image_url:
            try:
                reference_image_data = main.download_image_from_url(reference_image_url)
            except Exception as e:
                main.logger.warning(f"Failed to download reference image: {e}")
        
        story_pages = []
        consistency_results = []
        flagged_pages = []
        
        for i, page_text in enumerate(body.pages[:5], 1):  # Max 5 pages
            main.logger.info(f"Generating scene image for page {i}/5...")
            scene_prompt = scene_prompts[i - 1]
            
            scene_url = main.generate_story_scene_image(
                story_page_text=page_text,
                page_number=i,
                character_name=body.character_name,
                character_type=body.character_type,
                story_world=body.story_world,
                reference_image_url=reference_image_url,
                scene_prompt=scene_prompt
            )
            
            scene_http_url = None
            scene_image_data = None
            consistency_validation = None
            
            if scene_url:
                try:
                    scene_http_url = HttpUrl(scene_url)
                    try:
                        scene_image_data = main.download_image_from_url(scene_url)
                    except Exception:
                        pass
                except Exception as e:
                    main.logger.warning(f"Invalid scene URL for page {i}: {e}")
            
            if reference_image_data and scene_image_data:
                try:
                    consistency_validation = main.validate_character_consistency(
                        scene_image_data=scene_image_data,
                        reference_image_data=reference_image_data,
                        page_number=i,
                        timeout_seconds=15,
                        scene_image_url=scene_url,
                        reference_image_url=reference_image_url,
                    )
                    consistency_results.append(consistency_validation)
                    if consistency_validation.flagged:
                        flagged_pages.append(i)
                except Exception as e:
                    main.logger.error(f"Error during consistency validation for page {i}: {e}")
            
            story_pages.append(main.StoryPage(
                text=page_text,
                scene=scene_http_url,
                consistency_validation=consistency_validation
            ))
        
        main.logger.info("All scene images generated successfully")
        
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
        
        return {
            "success": True,
            "pages": [
                {
                    "text": p.text,
                    "scene": str(p.scene) if p.scene else None,
                    "consistency_validation": p.consistency_validation.model_dump() if p.consistency_validation is not None else None
                }
                for p in story_pages
            ],
            "consistency_summary": consistency_summary,
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in generate_story_scenes_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/story/generate-audio")
@limiter.limit("10/minute")
async def generate_story_audio_endpoint(request: Request, body: StoryAudioRequest):
    """Generate audio for story pages"""
    import main  # Import here to avoid circular import
    try:
        valid_age_groups = ["3-6", "7-10", "11-12"]
        if body.age_group not in valid_age_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: {body.age_group}. Must be one of: {', '.join(valid_age_groups)}"
            )
        
        main.logger.info("Generating audio for story pages...")
        
        audio_urls = []
        
        if main.supabase:
            try:
                audio_generator = AudioGenerator()
                if audio_generator.available:
                    audio_data_list = audio_generator.generate_audio_for_story(
                        story_pages=body.pages,
                        age_group=body.age_group,
                        timeout_per_page=60
                    )
                    
                    for i, audio_data in enumerate(audio_data_list, 1):
                        if audio_data is None:
                            audio_urls.append(None)
                            continue
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"story_audio_page{i}_{timestamp}_{unique_id}.mp3"
                        storage_bucket = "audio"
                        audio_url = None
                        
                        try:
                            try:
                                response = main.supabase.storage.from_(storage_bucket).upload(
                                    filename, audio_data, {"content-type": "audio/mpeg", "upsert": "true"}
                                )
                            except Exception:
                                storage_bucket = "images"
                                response = main.supabase.storage.from_(storage_bucket).upload(
                                    filename, audio_data, {"content-type": "audio/mpeg", "upsert": "true"}
                                )
                            
                            if hasattr(response, "full_path") and response.full_path:
                                audio_url = main.supabase.storage.from_(storage_bucket).get_public_url(filename)
                                main.logger.info(f"✅ Uploaded audio for page {i}: {audio_url}")
                        except Exception as e:
                            main.logger.error(f"Error uploading audio for page {i}: {e}")
                        
                        audio_urls.append(audio_url)
                    
                    if sum(1 for u in audio_urls if u) > 0:
                        main.logger.info(f"✅ Generated and uploaded {sum(1 for u in audio_urls if u)}/5 audio files")
                    else:
                        main.logger.warning("⚠️ Failed to generate/upload any audio files")
                else:
                    main.logger.warning("⚠️ Audio generator not available")
            except Exception as e:
                main.logger.error(f"Error during audio generation: {e}")
                import traceback
                main.logger.debug(f"Traceback: {traceback.format_exc()}")
        else:
            main.logger.warning("⚠️ Supabase not configured, skipping audio generation")
        
        return {
            "success": True,
            "audio_urls": audio_urls,
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Unexpected error in generate_story_audio_endpoint: {e}")
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
        # Some stories may store preview-page image URLs with either legacy keys
        # (e.g., dedication_image) or template-style keys (e.g., dedication_page_image).
        # Resolve both to ensure pages are not skipped during PDF generation.
        def _first_non_empty(*values):
            for value in values:
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned:
                        return cleaned
                elif value:
                    return value
            return None

        story_title = story.get("story_title") or "Untitled Story"
        story_cover = story.get("story_cover")
        scene_images = story.get("scene_images")
        copyright_image = _first_non_empty(
            story.get("copyright_image"),
            story.get("copyright_page_image"),
        )
        dedication_image = _first_non_empty(
            story.get("dedication_image"),
            story.get("dedication_page_image"),
        )
        last_word_page_image = _first_non_empty(
            story.get("last_word_page_image"),
            story.get("last_words_page_image"),
        )
        last_admin_page_image = _first_non_empty(
            story.get("last_admin_page_image"),
            story.get("last_story_page_image"),
        )
        back_cover_image = _first_non_empty(
            story.get("back_cover_image"),
            story.get("back_page_image"),
        )
        dedication_text = story.get("dedication_text") or ""
        character_name = story.get("character_name") or "[CHARACTER_NAME]"
        story_content = story.get("story_content")
        story_page_texts = []
        
        # Resolve child first name for copyright/dedication/last-words text (same as preview)
        copyright_child_name = "[CHILD_NAME]"
        child_profile_id = story.get("child_profile_id")
        if child_profile_id and main.supabase:
            try:
                child_resp = main.supabase.table("child_profiles").select("first_name").eq("id", child_profile_id).execute()
                if child_resp.data and len(child_resp.data) > 0 and child_resp.data[0].get("first_name"):
                    copyright_child_name = child_resp.data[0]["first_name"]
            except Exception as e:
                main.logger.warning(f"Could not fetch child name for PDF: {e}")
        
        # Parse dedication into body and signature (same logic as preview)
        import re
        dedication_raw = (dedication_text or "").strip()
        dedication_body = ""
        dedication_signature = ""
        if dedication_raw:
            dash_match = re.search(r"\s+[—–\-]\s+(.+)$", dedication_raw)
            if dash_match:
                dedication_body = dedication_raw[: dash_match.start()].strip()
                sig = (dash_match.group(1) or "").strip()
                dedication_signature = f"— {sig}" if sig else ""
            else:
                dedication_body = dedication_raw

        # Parse story page text list for PDF overlay (match /preview/default page text source)
        try:
            parsed_content = story_content
            if isinstance(parsed_content, str) and parsed_content.strip():
                parsed_content = json.loads(parsed_content)

            content_pages = []
            if isinstance(parsed_content, list):
                content_pages = parsed_content
            elif isinstance(parsed_content, dict):
                pages_value = parsed_content.get("pages")
                if isinstance(pages_value, list):
                    content_pages = pages_value
            elif isinstance(parsed_content, str) and parsed_content.strip():
                content_pages = [{"text": parsed_content.strip()}]

            style_keys = {
                "x", "y", "fontSize", "font_size", "color", "color_hex",
                "fontFamily", "font_family", "fontWeight", "font_weight",
                "fontStyle", "font_style", "strokeColor", "stroke_color",
                "strokeWidth", "stroke_width", "shadow", "align", "alignment",
            }
            for page in content_pages[:5]:
                if isinstance(page, dict):
                    text_value = (page.get("text") or "").strip()
                    if text_value and any(key in page for key in style_keys):
                        story_page_texts.append(page)
                        continue
                elif isinstance(page, str):
                    text_value = page.strip()
                else:
                    text_value = ""
                story_page_texts.append(text_value)
        except Exception as e:
            main.logger.warning(f"Could not parse story_content for PDF story text overlay: {e}")

        def _normalize_prompt_style(value: Any) -> str:
            return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")

        def _normalize_prompt_world(value: Any) -> str:
            world = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
            if "space" in world:
                return "outerspace"
            if "underwater" in world:
                return "underwater"
            if "forest" in world:
                return "forest"
            return world

        def _replace_pdf_text_placeholders(value: Any) -> Any:
            if isinstance(value, list):
                return [_replace_pdf_text_placeholders(item) for item in value]
            if not isinstance(value, dict):
                return value
            item = dict(value)
            text_value = str(item.get("text") or "")
            if text_value:
                text_value = re.sub(r"\[Character\s+Name\]", character_name, text_value, flags=re.IGNORECASE)
                text_value = re.sub(r"\[CHARACTER\s+NAME\]", character_name, text_value, flags=re.IGNORECASE)
                special_ability = str(story.get("special_ability") or "").strip()
                if special_ability:
                    text_value = re.sub(r"\[SPECIAL_ABILITY\]", special_ability, text_value, flags=re.IGNORECASE)
                item["text"] = text_value
            return item

        def _load_interactive_page_text_styles() -> List[Any]:
            try:
                prompt_image_data = _load_prompt_image_content()

                style_key = _normalize_prompt_style(story.get("character_style")) or "cartoon"
                world_key = _normalize_prompt_world(story.get("story_world"))
                page_text_content = (
                    prompt_image_data
                    .get("interactiveStoryStyleWorldPagePrompts", {})
                    .get(style_key, {})
                    .get(world_key, {})
                    .get("pageTextContent", {})
                )
                if not isinstance(page_text_content, dict):
                    return []

                scene_count = len(scene_images) if isinstance(scene_images, list) else len(story_page_texts)
                styled_pages: List[Any] = []
                for page_number in range(1, max(scene_count, len(story_page_texts), 0) + 1):
                    content = page_text_content.get(str(page_number)) or page_text_content.get(page_number)
                    if content:
                        styled_pages.append(_replace_pdf_text_placeholders(content))
                    elif page_number - 1 < len(story_page_texts):
                        styled_pages.append(story_page_texts[page_number - 1])
                return styled_pages
            except Exception as e:
                main.logger.warning(f"Could not load interactive PDF text styles: {e}")
                return []

        story_type = str(story.get("story_type") or "").strip().lower()
        story_format = str(story.get("story_format") or "").strip().lower()
        if story_type in {"search", "interactive", "search-and-find"} or story_format == "interactive_story":
            styled_interactive_texts = _load_interactive_page_text_styles()
            if styled_interactive_texts:
                story_page_texts = styled_interactive_texts
        
        # Check if we have at least cover or scene images (or other page images)
        has_cover = bool(story_cover)
        has_scenes = scene_images and (len(scene_images) if isinstance(scene_images, list) else True)
        has_other = bool(copyright_image or dedication_image or last_word_page_image or last_admin_page_image or back_cover_image)
        if not has_cover and not has_scenes and not has_other:
            raise HTTPException(
                status_code=400,
                detail="No cover image, scene images, or other page images found. Cannot generate PDF without images."
            )
        
        # Generate full PDF: cover, copyright, dedication, story pages, last words, last admin, back cover (with text overlays)
        main.logger.info(
            "Generating PDF: cover, copyright, dedication, story pages, last words, last admin, back cover"
        )
        
        output_buffer = BytesIO()
        success = create_book_pdf_with_cover(
            story_title=story_title,
            story_cover_url=story_cover,
            scene_urls=scene_images,
            output_buffer=output_buffer,
            copyright_image_url=copyright_image,
            dedication_image_url=dedication_image,
            last_word_page_image_url=last_word_page_image,
            last_admin_page_image_url=last_admin_page_image,
            back_cover_image_url=back_cover_image,
            copyright_child_name=copyright_child_name,
            copyright_character_name=character_name,
            dedication_body=dedication_body,
            dedication_signature=dedication_signature,
            story_page_texts=story_page_texts,
            story_world=story.get("story_world"),
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
                            main.logger.info(f"Story {body.story_id} has {current_hints} hints remaining (used {body.hints_used} in this game)")
                        else:
                            main.logger.warning(f"Story {body.story_id} has NULL hints")
                    else:
                        main.logger.warning(f"Story {body.story_id} not found")
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
