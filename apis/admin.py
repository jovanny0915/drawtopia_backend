"""
Admin API routes for book template management
Handles all admin operations including:
- Template CRUD operations
- Image uploads to Supabase storage
- Storage bucket file management
- Image optimization before upload
"""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
from rate_limiter import limiter
from datetime import datetime, timedelta
from collections import defaultdict
import random
import os
import logging
from uuid import uuid4
from image_optimizer import TemplateImageOptimizer
from storage_utils import (
    delete_story_images,
    delete_character_images,
    delete_files_from_storage,
    collect_book_template_image_urls,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize image optimizer for template images
image_optimizer = TemplateImageOptimizer()


# ==================== Pydantic Models ====================

class BookTemplateCreate(BaseModel):
    """Request model for creating a new book template"""
    name: str
    story_world: Optional[str] = None  # 'forest', 'underwater', or 'outerspace'


class BookTemplateUpdate(BaseModel):
    """Request model for updating book template metadata"""
    name: Optional[str] = None
    story_world: Optional[str] = None  # 'forest', 'underwater', or 'outerspace'
    cover_image: Optional[str] = None
    copyright_page_image: Optional[str] = None
    dedication_page_image: Optional[str] = None
    story_page_images: Optional[List[str]] = None
    last_words_page_image: Optional[str] = None
    last_story_page_image: Optional[str] = None
    back_cover_image: Optional[str] = None


class BookTemplateResponse(BaseModel):
    """Response model for book template"""
    id: str
    name: str
    story_world: Optional[str] = None  # 'forest', 'underwater', or 'outerspace'
    cover_image: Optional[str] = None
    copyright_page_image: Optional[str] = None
    dedication_page_image: Optional[str] = None
    story_page_images: Optional[List[str]] = None
    last_words_page_image: Optional[str] = None
    last_story_page_image: Optional[str] = None
    back_cover_image: Optional[str] = None
    created_at: Optional[str] = None


class AdminUserCreate(BaseModel):
    """Request model for creating a user profile from admin panel"""
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[str] = "user"
    subscription_status: Optional[str] = None
    credit: Optional[int] = 0


class AdminUserUpdate(BaseModel):
    """Request model for updating user profile fields from admin panel"""
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[str] = None
    subscription_status: Optional[str] = None
    credit: Optional[int] = None


# ==================== Helper Functions ====================

def get_supabase_client():
    """Get Supabase client from main module"""
    import main
    if not main.supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
    return main.supabase


def sanitize_template_name(name: str) -> str:
    """Sanitize template name for folder paths"""
    import re
    return re.sub(r'^-+|-+$', '', re.sub(r'[^a-z0-9]+', '-', name.lower().strip()))


async def upload_to_storage(file: UploadFile, bucket_name: str, file_path: str) -> str:
    """
    Upload file to Supabase storage with optimization and return public URL.
    Images are automatically optimized to WebP format before upload.
    """
    supabase = get_supabase_client()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Optimize image before upload
        logger.info(f"🔧 Optimizing image before upload: {file.filename}")
        try:
            optimized_content, content_type, extension = image_optimizer.optimize_image(
                file_content,
                filename=file.filename
            )
            
            # Update file path to use optimized extension
            if not file_path.endswith(f".{extension}"):
                # Replace original extension with optimized extension
                base_path = file_path.rsplit(".", 1)[0] if "." in file_path else file_path
                file_path = f"{base_path}.{extension}"
            
            logger.info(
                f"✅ Image optimized: {len(file_content) / 1024:.1f}KB → "
                f"{len(optimized_content) / 1024:.1f}KB "
                f"({content_type})"
            )
            
            # Use optimized content
            upload_content = optimized_content
            upload_content_type = content_type
            
        except Exception as opt_error:
            logger.warning(f"⚠️ Image optimization failed, uploading original: {opt_error}")
            # Fallback to original if optimization fails
            upload_content = file_content
            upload_content_type = file.content_type or "image/jpeg"
        
        # Upload to storage with upsert (overwrites existing file)
        response = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=upload_content,
            file_options={
                "content-type": upload_content_type,
                "upsert": "true"
            }
        )
        
        # Get public URL
        public_url_response = supabase.storage.from_(bucket_name).get_public_url(file_path)
        public_url = public_url_response
        
        logger.info(f"✅ Uploaded optimized file to storage: {file_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Error uploading file to storage: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file to storage: {str(e)}"
        )


async def delete_folder_from_storage(bucket_name: str, folder_path: str) -> None:
    """Delete all files in a folder from Supabase storage"""
    supabase = get_supabase_client()
    
    try:
        # List all files in the folder
        files_response = supabase.storage.from_(bucket_name).list(folder_path)
        
        if files_response and len(files_response) > 0:
            # Build file paths to delete
            file_paths = [f"{folder_path}/{file['name']}" for file in files_response]
            
            # Delete all files
            delete_response = supabase.storage.from_(bucket_name).remove(file_paths)
            
            logger.info(f"✅ Deleted {len(file_paths)} files from storage folder: {folder_path}")
        else:
            logger.info(f"ℹ️ No files found in storage folder: {folder_path}")
            
    except Exception as e:
        logger.error(f"⚠️ Error deleting folder from storage: {e}")
        # Don't raise exception - continue with template deletion even if storage cleanup fails


def _delete_urls_or_raise(supabase_client, urls: List[str], context: str) -> None:
    """Delete storage URLs and raise when any deletion fails."""
    urls_to_delete = [url for url in urls if isinstance(url, str) and url.strip()]
    if not urls_to_delete:
        return

    deletion_stats = delete_files_from_storage(supabase_client, urls_to_delete)
    if deletion_stats.get("errors", 0) > 0:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to delete one or more files from storage for {context}. "
                "Database was not updated."
            )
        )


def _safe_delete_eq(supabase_client, table_name: str, column: str, value: Any) -> int:
    """Delete rows by equality and return deleted count (best-effort)."""
    try:
        response = supabase_client.table(table_name).delete().eq(column, value).execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        logger.warning(f"⚠️ Could not delete from {table_name} where {column}={value}: {e}")
        return 0


def _safe_delete_in(supabase_client, table_name: str, column: str, values: List[Any]) -> int:
    """Delete rows by IN list and return deleted count (best-effort)."""
    if not values:
        return 0
    try:
        response = supabase_client.table(table_name).delete().in_(column, values).execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        logger.warning(f"⚠️ Could not delete from {table_name} where {column} IN (...): {e}")
        return 0


# ==================== API Endpoints ====================

@router.get("/admin/analysis/story-counts-by-day")
@limiter.limit("60/minute")
async def get_story_counts_by_day(request: Request, days: int = Query(90, ge=7, le=365)):
    """
    Get counts of story generation per day from the stories table.
    Returns list of { date: "YYYY-MM-DD", count: number } for the last `days` days.
    """
    supabase = get_supabase_client()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        response = supabase.table("stories").select("created_at").gte("created_at", since).execute()
        rows = response.data if response.data else []
        # Group by date (day only)
        by_day = defaultdict(int)
        for row in rows:
            created = row.get("created_at")
            if not created:
                continue
            if isinstance(created, str):
                day = created[:10]  # "YYYY-MM-DD"
            else:
                day = datetime.fromisoformat(str(created).replace("Z", "+00:00")).strftime("%Y-%m-%d")
            by_day[day] += 1
        # Sort by date and return list
        result = [{"date": d, "count": c} for d, c in sorted(by_day.items())]
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error fetching story counts by day: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch story counts: {str(e)}")


@router.get("/admin/analysis/user-auth-counts-by-day")
@limiter.limit("60/minute")
async def get_user_auth_counts_by_day(request: Request, days: int = Query(90, ge=7, le=365)):
    """
    Get daily counts for login/register events from user_auth_history table.
    Returns:
    [
      { date: "YYYY-MM-DD", login_count: number, register_count: number, total_count: number }
    ]
    """
    supabase = get_supabase_client()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        response = (
            supabase
            .table("user_auth_history")
            .select("created_at,event_type")
            .gte("created_at", since)
            .execute()
        )
        rows = response.data if response.data else []

        # Group by day and auth event type.
        by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: {"login_count": 0, "register_count": 0})
        for row in rows:
            created = row.get("created_at")
            event_type = row.get("event_type")
            if not created or event_type not in ("login", "register"):
                continue

            if isinstance(created, str):
                day = created[:10]  # "YYYY-MM-DD"
            else:
                day = datetime.fromisoformat(str(created).replace("Z", "+00:00")).strftime("%Y-%m-%d")

            if event_type == "login":
                by_day[day]["login_count"] += 1
            elif event_type == "register":
                by_day[day]["register_count"] += 1

        result = []
        for day, counts in sorted(by_day.items()):
            login_count = counts["login_count"]
            register_count = counts["register_count"]
            result.append({
                "date": day,
                "login_count": login_count,
                "register_count": register_count,
                "total_count": login_count + register_count
            })

        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error fetching user auth counts by day: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user auth counts: {str(e)}")


@router.get("/admin/templates")
@limiter.limit("30/minute")
async def get_templates(request: Request):
    """Get all book templates"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table("book_templates").select("*").order("created_at", desc=True).execute()
        
        return {
            "success": True,
            "data": response.data
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch templates: {str(e)}")


@router.get("/templates/random")
@limiter.limit("60/minute")
async def get_random_template_by_story_world(
    request: Request,
    story_world: str = Query(..., description="Story world: forest, underwater, or outerspace")
):
    """Get one random template by story world (public endpoint for cover generation)."""
    supabase = get_supabase_client()

    valid_story_worlds = ["forest", "underwater", "outerspace"]
    normalized_world = (story_world or "").strip().lower()
    if normalized_world not in valid_story_worlds:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid story_world. Must be one of: {', '.join(valid_story_worlds)}"
        )

    try:
        response = (
            supabase
            .table("book_templates")
            .select("*")
            .eq("story_world", normalized_world)
            .not_.is_("cover_image", "null")
            .execute()
        )

        templates = response.data or []
        if len(templates) == 0:
            return {
                "success": False,
                "error": f"No templates found for story world: {normalized_world}"
            }

        return {
            "success": True,
            "data": random.choice(templates)
        }
    except Exception as e:
        logger.error(f"❌ Error fetching random template for {normalized_world}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch random template: {str(e)}")


@router.get("/admin/users")
@limiter.limit("30/minute")
async def get_users(request: Request):
    """Get users list for admin user management table"""
    supabase = get_supabase_client()

    try:
        response = (
            supabase
            .table("users")
            .select("id,email,first_name,last_name,avatar_url,role,subscription_status,credit,created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return {"success": True, "data": response.data or []}
    except Exception as e:
        logger.error(f"❌ Error fetching users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")


@router.post("/admin/users")
@limiter.limit("10/minute")
async def create_user(request: Request, body: AdminUserCreate):
    """Create user record from admin panel"""
    supabase = get_supabase_client()

    email = (body.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    try:
        existing = (
            supabase
            .table("users")
            .select("id")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="User with this email already exists")

        insert_data = {
            "id": str(uuid4()),
            "email": email,
            "first_name": (body.first_name or "").strip() or None,
            "last_name": (body.last_name or "").strip() or None,
            "role": (body.role or "user").strip() or "user",
            "subscription_status": (body.subscription_status or "").strip() or None,
            "credit": max(0, body.credit or 0),
        }

        response = supabase.table("users").insert(insert_data).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create user")

        return {"success": True, "data": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")


@router.patch("/admin/users/{user_id}")
@limiter.limit("20/minute")
async def update_user(request: Request, user_id: str, body: AdminUserUpdate):
    """Update user record from admin panel"""
    supabase = get_supabase_client()

    try:
        update_data: Dict[str, Any] = {}
        if body.email is not None:
            update_data["email"] = body.email.strip().lower()
        if body.first_name is not None:
            update_data["first_name"] = body.first_name.strip() or None
        if body.last_name is not None:
            update_data["last_name"] = body.last_name.strip() or None
        if body.role is not None:
            update_data["role"] = body.role.strip() or "user"
        if body.subscription_status is not None:
            update_data["subscription_status"] = body.subscription_status.strip() or None
        if body.credit is not None:
            update_data["credit"] = max(0, body.credit)

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        response = supabase.table("users").update(update_data).eq("id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="User not found")

        return {"success": True, "data": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")


@router.delete("/admin/users/{user_id}")
@limiter.limit("10/minute")
async def delete_user(request: Request, user_id: str):
    """Delete user and all related data/storage assets from admin panel"""
    supabase = get_supabase_client()

    try:
        # Verify existence for cleaner error messages
        existing = supabase.table("users").select("id,email").eq("id", user_id).single().execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="User not found")

        user_email = existing.data.get("email")

        # 1) Collect child profiles for this parent
        child_profiles = (
            supabase.table("child_profiles")
            .select("id,avatar_url")
            .eq("parent_id", user_id)
            .execute()
        )
        child_rows = child_profiles.data or []
        child_ids = [row.get("id") for row in child_rows if row.get("id") is not None]
        child_avatar_urls = [row.get("avatar_url") for row in child_rows if row.get("avatar_url")]

        # 2) Collect characters for this user
        characters_response = (
            supabase.table("characters")
            .select("id,original_image_url,enhanced_images")
            .eq("user_id", user_id)
            .execute()
        )
        character_rows = characters_response.data or []
        character_ids = [row.get("id") for row in character_rows if row.get("id") is not None]

        # 3) Collect stories related to this user via user/child/character relations
        story_map: Dict[Any, Dict[str, Any]] = {}

        user_stories = (
            supabase.table("stories")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        for row in (user_stories.data or []):
            story_map[row.get("id")] = row

        if child_ids:
            child_stories = (
                supabase.table("stories")
                .select("*")
                .in_("child_profile_id", child_ids)
                .execute()
            )
            for row in (child_stories.data or []):
                story_map[row.get("id")] = row

        if character_ids:
            character_stories = (
                supabase.table("stories")
                .select("*")
                .in_("character_id", character_ids)
                .execute()
            )
            for row in (character_stories.data or []):
                story_map[row.get("id")] = row

        story_rows = list(story_map.values())
        story_ids = [row.get("id") for row in story_rows if row.get("id") is not None]

        # 4) Delete related storage files (best-effort)
        storage_files_deleted = 0
        storage_files_failed = 0

        protected_template_urls: Set[str] = set()
        try:
            protected_template_urls = collect_book_template_image_urls(supabase)
        except Exception as template_lookup_error:
            logger.warning(
                f"⚠️ Could not load shared template URLs for user-delete cleanup protection: {template_lookup_error}"
            )

        for story in story_rows:
            try:
                # Full cleanup for admin delete: include character/enhanced images too.
                deletion_result = delete_story_images(
                    supabase,
                    story,
                    exclude_character_images=False,
                    protected_urls=protected_template_urls,
                )
                storage_files_deleted += deletion_result.get("success", 0)
                storage_files_failed += deletion_result.get("errors", 0)
            except Exception as e:
                logger.warning(f"⚠️ Story storage cleanup failed for story_id={story.get('id')}: {e}")

        for character in character_rows:
            try:
                deletion_result = delete_character_images(supabase, character)
                storage_files_deleted += deletion_result.get("success", 0)
                storage_files_failed += deletion_result.get("errors", 0)
            except Exception as e:
                logger.warning(f"⚠️ Character storage cleanup failed for character_id={character.get('id')}: {e}")

        if child_avatar_urls:
            try:
                child_avatar_cleanup = delete_files_from_storage(supabase, child_avatar_urls)
                storage_files_deleted += child_avatar_cleanup.get("success", 0)
                storage_files_failed += child_avatar_cleanup.get("errors", 0)
            except Exception as e:
                logger.warning(f"⚠️ Child avatar cleanup failed for user_id={user_id}: {e}")

        # 5) Delete related rows from all known tables (best-effort per table)
        deleted_counts: Dict[str, int] = {}

        # Gifts and user activity
        deleted_counts["gifts_by_user_id"] = _safe_delete_eq(supabase, "gifts", "user_id", user_id)
        deleted_counts["gifts_by_from_user_id"] = _safe_delete_eq(supabase, "gifts", "from_user_id", user_id)
        deleted_counts["gifts_by_to_user_id"] = _safe_delete_eq(supabase, "gifts", "to_user_id", user_id)
        deleted_counts["user_auth_history"] = _safe_delete_eq(supabase, "user_auth_history", "user_id", user_id)
        deleted_counts["push_subscriptions"] = _safe_delete_eq(supabase, "push_subscriptions", "user_id", user_id)
        deleted_counts["subscriptions"] = _safe_delete_eq(supabase, "subscriptions", "user_id", user_id)
        deleted_counts["book_generation_jobs_by_user"] = _safe_delete_eq(supabase, "book_generation_jobs", "user_id", user_id)
        deleted_counts["book_purchases_by_user"] = _safe_delete_eq(supabase, "book_purchases", "user_id", user_id)
        deleted_counts["search_game_results_by_user"] = _safe_delete_eq(supabase, "search_game_results", "user_id", user_id)

        # Story-linked records
        deleted_counts["stories"] = _safe_delete_in(supabase, "stories", "id", story_ids)
        deleted_counts["book_purchases_by_story"] = _safe_delete_in(supabase, "book_purchases", "story_id", story_ids)
        deleted_counts["search_game_results_by_story"] = _safe_delete_in(supabase, "search_game_results", "story_id", story_ids)
        deleted_counts["gifts_by_story"] = _safe_delete_in(supabase, "gifts", "story_id", story_ids)

        # Character-linked and child-linked records
        deleted_counts["characters"] = _safe_delete_in(supabase, "characters", "id", character_ids)
        deleted_counts["search_game_results_by_character"] = _safe_delete_in(supabase, "search_game_results", "character_id", character_ids)
        deleted_counts["child_profiles"] = _safe_delete_in(supabase, "child_profiles", "id", child_ids)
        deleted_counts["search_game_results_by_child"] = _safe_delete_in(supabase, "search_game_results", "child_profile_id", child_ids)
        deleted_counts["book_generation_jobs_by_child"] = _safe_delete_in(supabase, "book_generation_jobs", "child_profile_id", child_ids)
        deleted_counts["gifts_by_child"] = _safe_delete_in(supabase, "gifts", "child_profile_id", child_ids)

        # 6) Delete user row in custom users table
        deleted_counts["users"] = _safe_delete_eq(supabase, "users", "id", user_id)
        if deleted_counts["users"] == 0:
            # Safety: if row wasn't deleted, return error because this is primary target.
            raise HTTPException(status_code=500, detail="Failed to delete user row from users table")

        # 7) Try deleting auth user as final step (best-effort)
        auth_user_deleted = False
        try:
            # Supabase admin SDKs differ by method name across versions.
            if hasattr(supabase.auth.admin, "delete_user"):
                supabase.auth.admin.delete_user(user_id)
                auth_user_deleted = True
            elif hasattr(supabase.auth.admin, "deleteUser"):
                supabase.auth.admin.deleteUser(user_id)
                auth_user_deleted = True
        except Exception as e:
            logger.warning(f"⚠️ Could not delete auth user {user_id}: {e}")

        return {
            "success": True,
            "message": "User and related data deleted successfully",
            "data": {
                "user_id": user_id,
                "email": user_email,
                "auth_user_deleted": auth_user_deleted,
                "related_story_count": len(story_ids),
                "related_character_count": len(character_ids),
                "related_child_profile_count": len(child_ids),
                "storage_cleanup": {
                    "files_deleted": storage_files_deleted,
                    "files_failed": storage_files_failed
                },
                "deleted_counts": deleted_counts
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")


@router.post("/admin/templates")
@limiter.limit("10/minute")
async def create_template(request: Request, body: BookTemplateCreate):
    """Create a new book template"""
    supabase = get_supabase_client()
    
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="Template name is required")
    
    # Validate story_world if provided
    valid_story_worlds = ['forest', 'underwater', 'outerspace']
    if body.story_world and body.story_world not in valid_story_worlds:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid story_world. Must be one of: {', '.join(valid_story_worlds)}"
        )
    
    try:
        insert_data = {"name": body.name.strip()}
        if body.story_world:
            insert_data["story_world"] = body.story_world
        
        response = supabase.table("book_templates").insert(insert_data).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create template")
        
        logger.info(f"✅ Created template: {body.name} (story_world: {body.story_world or 'none'})")
        
        return {
            "success": True,
            "data": response.data[0]
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@router.delete("/admin/templates/{template_id}")
@limiter.limit("10/minute")
async def delete_template(request: Request, template_id: str):
    """Delete a book template and all associated images from storage"""
    supabase = get_supabase_client()
    
    try:
        # Get template to get its name for storage deletion
        response = supabase.table("book_templates").select("name").eq("id", template_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template_name = response.data["name"]
        sanitized_name = sanitize_template_name(template_name)
        folder_path = f"book-templates/{sanitized_name}"
        
        # Delete files from storage bucket
        await delete_folder_from_storage("book-images", folder_path)
        
        # Delete template from database
        delete_response = supabase.table("book_templates").delete().eq("id", template_id).execute()
        
        logger.info(f"✅ Deleted template: {template_name} (ID: {template_id})")
        
        return {
            "success": True,
            "message": f"Template '{template_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-image")
@limiter.limit("20/minute")
async def upload_template_image(
    request: Request,
    template_id: str,
    file: UploadFile = File(...),
    field_key: str = Form(...),
    template_name: str = Form(...)
):
    """
    Upload a single image for a book template field.
    Images are automatically optimized to WebP format before upload to save storage space.
    
    Args:
        template_id: ID of the template
        file: Image file to upload (will be optimized to WebP)
        field_key: Database field name (cover_image, back_cover_image)
        template_name: Name of the template (for folder path)
    
    Returns:
        JSON with success status, updated template data, and optimized image URL
    """
    supabase = get_supabase_client()
    
    # Validate field_key
    valid_fields = [
        "cover_image",
        "copyright_page_image",
        "dedication_page_image",
        "back_cover_image",
        "last_words_page_image",
        "last_story_page_image",
    ]
    if field_key not in valid_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field_key. Must be one of: {', '.join(valid_fields)}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Build storage path (extension will be updated by optimizer to .webp)
        sanitized_name = sanitize_template_name(template_name)
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_path = f"book-templates/{sanitized_name}/{field_key}.{file_ext}"
        
        # Upload to storage (will be optimized to WebP automatically)
        public_url = await upload_to_storage(file, "book-images", file_path)
        
        # Update database
        update_data = {field_key: public_url}
        response = supabase.table("book_templates").update(update_data).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")
        
        logger.info(f"✅ Uploaded {field_key} for template: {template_name}")
        
        return {
            "success": True,
            "data": response.data[0],
            "image_url": public_url,
            "optimized": True,
            "format": "WebP"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-story-page")
@limiter.limit("30/minute")
async def upload_single_story_page(
    request: Request,
    template_id: str,
    file: UploadFile = File(...),
    template_name: str = Form(...),
    page_index: int = Form(...)  # Index position for this page (0-based)
):
    """
    Upload a single story page image for a book template.
    Images are automatically optimized to WebP format before upload.
    This endpoint should be called multiple times (once per image) to avoid 413 errors.
    
    Args:
        template_id: ID of the template
        file: Single image file to upload (will be optimized to WebP)
        template_name: Name of the template (for folder path)
        page_index: Index position for this page in the story_page_images array (0-based)
    
    Returns:
        JSON with success status, updated template data, and image URL
    """
    supabase = get_supabase_client()
    
    # Validate file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")
    
    try:
        # Get current template to retrieve existing story pages
        template_response = supabase.table("book_templates").select("story_page_images").eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        existing_urls = template_response.data.get("story_page_images") or []
        
        # Upload new file (will be optimized to WebP automatically)
        sanitized_name = sanitize_template_name(template_name)
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_path = f"book-templates/{sanitized_name}/story-page-{page_index + 1}.{file_ext}"
        
        # Upload to storage (with automatic optimization)
        public_url = await upload_to_storage(file, "book-images", file_path)
        
        # Insert or update the URL at the specified index
        if page_index < len(existing_urls):
            # Update existing position
            existing_urls[page_index] = public_url
        else:
            # Append to the end (fill gaps if needed)
            while len(existing_urls) < page_index:
                existing_urls.append(None)  # Placeholder for gaps
            existing_urls.append(public_url)
        
        # Remove any None placeholders
        existing_urls = [url for url in existing_urls if url is not None]
        
        # Update database with new array
        response = supabase.table("book_templates").update({
            "story_page_images": existing_urls
        }).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")
        
        logger.info(f"✅ Uploaded story page {page_index + 1} for template: {template_name}")
        
        return {
            "success": True,
            "data": response.data[0],
            "image_url": public_url,
            "page_index": page_index,
            "total_pages": len(existing_urls),
            "optimized": True,
            "format": "WebP"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading story page: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload story page: {str(e)}")


@router.post("/admin/templates/{template_id}/upload-story-pages")
@limiter.limit("20/minute")
async def upload_story_pages(
    request: Request,
    template_id: str,
    files: List[UploadFile] = File(...),
    template_name: str = Form(...),
    existing_images: str = Form(default="[]")  # JSON string of existing image URLs
):
    """
    Upload multiple story page images for a book template (DEPRECATED - use upload-story-page instead).
    All images are automatically optimized to WebP format before upload to save storage space.
    
    NOTE: This endpoint may cause 413 errors with many/large files. 
    Use POST /upload-story-page endpoint instead to upload one image at a time.
    
    Args:
        template_id: ID of the template
        files: List of image files to upload (will be optimized to WebP)
        template_name: Name of the template (for folder path)
        existing_images: JSON string array of existing image URLs to preserve
    
    Returns:
        JSON with success status, updated template data, upload count, and optimization info
    """
    supabase = get_supabase_client()
    
    try:
        import json
        existing_urls = json.loads(existing_images) if existing_images else []
        
        if not isinstance(existing_urls, list):
            raise HTTPException(status_code=400, detail="existing_images must be a JSON array")
        
        # Validate all files are images
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image")
        
        # Upload new files (will be optimized to WebP automatically)
        sanitized_name = sanitize_template_name(template_name)
        new_urls = []
        
        for idx, file in enumerate(files):
            current_index = len(existing_urls) + idx
            file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            file_path = f"book-templates/{sanitized_name}/story-page-{current_index + 1}.{file_ext}"
            
            # Upload to storage (with automatic optimization)
            public_url = await upload_to_storage(file, "book-images", file_path)
            new_urls.append(public_url)
        
        # Combine existing and new URLs
        all_urls = existing_urls + new_urls
        
        # Update database
        response = supabase.table("book_templates").update({
            "story_page_images": all_urls
        }).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update template in database")
        
        logger.info(f"✅ Uploaded {len(new_urls)} story pages for template: {template_name}")
        
        return {
            "success": True,
            "data": response.data[0],
            "uploaded_count": len(new_urls),
            "total_count": len(all_urls),
            "optimized": True,
            "format": "WebP"
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid existing_images JSON format")
    except Exception as e:
        logger.error(f"❌ Error uploading story pages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload story pages: {str(e)}")


@router.delete("/admin/templates/{template_id}/image")
@limiter.limit("30/minute")
async def delete_template_image(
    request: Request,
    template_id: str,
    field_key: str = Query(...)
):
    """Delete a single template image from storage and clear its DB field."""
    supabase = get_supabase_client()

    valid_fields = [
        "cover_image",
        "copyright_page_image",
        "dedication_page_image",
        "back_cover_image",
        "last_words_page_image",
        "last_story_page_image",
    ]
    if field_key not in valid_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field_key. Must be one of: {', '.join(valid_fields)}"
        )

    try:
        template_response = (
            supabase
            .table("book_templates")
            .select(field_key)
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        existing_url = template_response.data.get(field_key)
        _delete_urls_or_raise(
            supabase,
            [existing_url] if existing_url else [],
            f"template field '{field_key}'"
        )

        response = (
            supabase
            .table("book_templates")
            .update({field_key: None})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Template not found")

        logger.info(f"✅ Deleted template image '{field_key}' for template {template_id}")
        return {
            "success": True,
            "data": response.data[0],
            "deleted_url": existing_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting template image '{field_key}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template image: {str(e)}")


@router.delete("/admin/templates/{template_id}/story-page/{page_index}")
@limiter.limit("30/minute")
async def delete_story_page_image(
    request: Request,
    template_id: str,
    page_index: int
):
    """Delete one story page image from storage and remove it from DB array."""
    supabase = get_supabase_client()

    if page_index < 0:
        raise HTTPException(status_code=400, detail="page_index must be >= 0")

    try:
        template_response = (
            supabase
            .table("book_templates")
            .select("story_page_images")
            .eq("id", template_id)
            .single()
            .execute()
        )

        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")

        story_page_images = template_response.data.get("story_page_images") or []
        if not isinstance(story_page_images, list):
            story_page_images = []

        if page_index >= len(story_page_images):
            raise HTTPException(status_code=400, detail="Invalid story page index")

        removed_url = story_page_images[page_index]
        _delete_urls_or_raise(
            supabase,
            [removed_url] if removed_url else [],
            f"story page index {page_index}"
        )

        next_story_page_images = [url for idx, url in enumerate(story_page_images) if idx != page_index]
        response = (
            supabase
            .table("book_templates")
            .update({"story_page_images": next_story_page_images})
            .eq("id", template_id)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Template not found")

        logger.info(f"✅ Deleted story page {page_index + 1} for template {template_id}")
        return {
            "success": True,
            "data": response.data[0],
            "deleted_url": removed_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting story page {page_index} for template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete story page image: {str(e)}")


@router.patch("/admin/templates/{template_id}")
@limiter.limit("20/minute")
async def update_template(
    request: Request,
    template_id: str,
    body: BookTemplateUpdate
):
    """Update book template metadata (name, story_world, or image URLs)"""
    supabase = get_supabase_client()
    
    try:
        provided_fields = getattr(body, "__fields_set__", set())

        # Validate story_world if provided
        if "story_world" in provided_fields and body.story_world is not None:
            valid_story_worlds = ['forest', 'underwater', 'outerspace']
            # Empty string means clear the story_world
            if body.story_world and body.story_world not in valid_story_worlds:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid story_world. Must be one of: {', '.join(valid_story_worlds)}"
                )
        
        # Build update data from non-None fields
        update_data = {}
        if "name" in provided_fields and body.name is not None:
            update_data["name"] = body.name.strip()
        if "story_world" in provided_fields:
            # Empty string or null means clear the field
            update_data["story_world"] = body.story_world if body.story_world else None
        if "cover_image" in provided_fields:
            update_data["cover_image"] = body.cover_image
        if "story_page_images" in provided_fields:
            update_data["story_page_images"] = body.story_page_images
        if "copyright_page_image" in provided_fields:
            update_data["copyright_page_image"] = body.copyright_page_image
        if "dedication_page_image" in provided_fields:
            update_data["dedication_page_image"] = body.dedication_page_image
        if "last_words_page_image" in provided_fields:
            update_data["last_words_page_image"] = body.last_words_page_image
        if "last_story_page_image" in provided_fields:
            update_data["last_story_page_image"] = body.last_story_page_image
        if "back_cover_image" in provided_fields:
            update_data["back_cover_image"] = body.back_cover_image

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Update database
        response = supabase.table("book_templates").update(update_data).eq("id", template_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(f"✅ Updated template (ID: {template_id})")
        
        return {
            "success": True,
            "data": response.data[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")
