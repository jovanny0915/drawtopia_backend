"""
Storage utility functions for deleting files from Supabase S3 buckets
"""
import logging
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def extract_storage_path_from_url(url: str) -> Optional[tuple[str, str]]:
    """
    Extract bucket name and file path from a Supabase storage URL
    
    Args:
        url: Full Supabase storage URL (e.g., "https://...supabase.co/storage/v1/object/public/images/filename.jpg")
    
    Returns:
        Tuple of (bucket_name, file_path) or None if URL is invalid
    """
    if not url or not isinstance(url, str):
        return None
    
    try:
        # Parse URL
        parsed = urlparse(url)
        path = parsed.path
        
        # Expected format: /storage/v1/object/public/{bucket}/{file_path}
        if '/storage/v1/object/public/' in path:
            parts = path.split('/storage/v1/object/public/', 1)
            if len(parts) == 2:
                remaining = parts[1]
                # Split into bucket and file path
                path_parts = remaining.split('/', 1)
                if len(path_parts) >= 1:
                    bucket = path_parts[0]
                    file_path = path_parts[1] if len(path_parts) > 1 else ""
                    return (bucket, file_path)
        
        return None
    except Exception as e:
        logger.warning(f"Error parsing storage URL: {e}")
        return None


def delete_files_from_storage(supabase_client, urls: List[str]) -> dict:
    """
    Delete multiple files from Supabase storage
    
    Args:
        supabase_client: Supabase client instance
        urls: List of full storage URLs to delete
    
    Returns:
        Dict with success count and error count
    """
    if not urls:
        return {"success": 0, "errors": 0}
    
    success_count = 0
    error_count = 0
    
    # Group files by bucket for efficient deletion
    bucket_files = {}
    
    for url in urls:
        if not url:
            continue
            
        result = extract_storage_path_from_url(url)
        if result:
            bucket, file_path = result
            if bucket not in bucket_files:
                bucket_files[bucket] = []
            bucket_files[bucket].append(file_path)
        else:
            logger.warning(f"Could not extract storage path from URL: {url}")
            error_count += 1
    
    # Delete files bucket by bucket
    for bucket, file_paths in bucket_files.items():
        try:
            logger.info(f"Deleting {len(file_paths)} files from bucket '{bucket}'")
            
            # Supabase storage delete accepts list of file paths
            delete_response = supabase_client.storage.from_(bucket).remove(file_paths)
            
            success_count += len(file_paths)
            logger.info(f"✅ Successfully deleted {len(file_paths)} files from bucket '{bucket}'")
            
        except Exception as e:
            logger.error(f"❌ Error deleting files from bucket '{bucket}': {e}")
            error_count += len(file_paths)
    
    return {
        "success": success_count,
        "errors": error_count
    }


def delete_story_images(supabase_client, story_data: dict, exclude_character_images: bool = True) -> dict:
    """
    Delete story images from Supabase storage
    
    Args:
        supabase_client: Supabase client instance
        story_data: Story data dictionary from database
        exclude_character_images: If True, keep character image and enhancement images
    
    Returns:
        Dict with deletion statistics
    """
    urls_to_delete = []
    
    # Collect story-specific images (always delete these)
    if story_data.get("story_cover"):
        urls_to_delete.append(story_data["story_cover"])
    
    if story_data.get("scene_images"):
        scene_images = story_data["scene_images"]
        if isinstance(scene_images, list):
            urls_to_delete.extend([img for img in scene_images if img])
    
    if story_data.get("dedication_image"):
        urls_to_delete.append(story_data["dedication_image"])
    
    # Delete audio files
    if story_data.get("audio_url"):
        audio_urls = story_data["audio_url"]
        if isinstance(audio_urls, list):
            urls_to_delete.extend([audio for audio in audio_urls if audio])
    
    # Delete PDF if exists
    if story_data.get("pdf_url"):
        urls_to_delete.append(story_data["pdf_url"])
    
    # Character images (only delete if exclude_character_images is False)
    if not exclude_character_images:
        if story_data.get("original_image_url"):
            urls_to_delete.append(story_data["original_image_url"])
        
        if story_data.get("enhanced_images"):
            enhanced_images = story_data["enhanced_images"]
            if isinstance(enhanced_images, list):
                urls_to_delete.extend([img for img in enhanced_images if img])
    
    logger.info(f"Deleting {len(urls_to_delete)} files for story (exclude_character_images={exclude_character_images})")
    
    return delete_files_from_storage(supabase_client, urls_to_delete)


def delete_character_images(supabase_client, character_data: dict) -> dict:
    """
    Delete all character images from Supabase storage
    
    Args:
        supabase_client: Supabase client instance
        character_data: Character data dictionary from database
    
    Returns:
        Dict with deletion statistics
    """
    urls_to_delete = []
    
    # Collect character images
    if character_data.get("original_image_url"):
        urls_to_delete.append(character_data["original_image_url"])
    
    if character_data.get("enhanced_images"):
        enhanced_images = character_data["enhanced_images"]
        if isinstance(enhanced_images, list):
            urls_to_delete.extend([img for img in enhanced_images if img])
    
    logger.info(f"Deleting {len(urls_to_delete)} character image files")
    
    return delete_files_from_storage(supabase_client, urls_to_delete)
