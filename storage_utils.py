"""
Storage utility functions for deleting files from Supabase S3 buckets
"""
import logging
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


SHARED_TEMPLATE_PREFIXES = (
    "book-templates/",
    "book_templates/",
)


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
        
        # Handle common Supabase storage URL formats:
        # - /storage/v1/object/public/{bucket}/{file_path}
        # - /storage/v1/object/sign/{bucket}/{file_path}
        # - /storage/v1/object/authenticated/{bucket}/{file_path}
        marker = "/storage/v1/object/"
        if marker in path:
            parts = path.split(marker, 1)
            if len(parts) == 2 and parts[1]:
                remaining = parts[1]
                visibility_split = remaining.split("/", 1)
                if len(visibility_split) == 2 and visibility_split[1]:
                    bucket_and_path = visibility_split[1]
                    path_parts = bucket_and_path.split("/", 1)
                    if len(path_parts) >= 1:
                        bucket = path_parts[0]
                        file_path = path_parts[1] if len(path_parts) > 1 else ""
                        if bucket:
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


def _is_shared_template_asset(url: str) -> bool:
    """
    Return True when URL points to a shared book-template asset.

    Shared template files are reused by many stories and must not be deleted
    during per-story cleanup.
    """
    parsed = extract_storage_path_from_url(url)
    if not parsed:
        return False

    _, file_path = parsed
    normalized_path = (file_path or "").lstrip("/").lower()
    return any(normalized_path.startswith(prefix) for prefix in SHARED_TEMPLATE_PREFIXES)


def _normalized_storage_target(url: str) -> Optional[Tuple[str, str]]:
    """
    Convert a storage URL to a comparable (bucket, normalized_path) tuple.
    """
    parsed = extract_storage_path_from_url(url)
    if not parsed:
        return None

    bucket, file_path = parsed
    normalized_bucket = (bucket or "").strip().lower()
    normalized_path = (file_path or "").lstrip("/")
    if not normalized_bucket or not normalized_path:
        return None
    return (normalized_bucket, normalized_path)


def _build_protected_storage_targets(urls: Optional[Set[str]]) -> Set[Tuple[str, str]]:
    """
    Build normalized storage targets from protected URL strings.
    """
    if not urls:
        return set()

    targets: Set[Tuple[str, str]] = set()
    for url in urls:
        if not isinstance(url, str) or not url.strip():
            continue
        target = _normalized_storage_target(url)
        if target:
            targets.add(target)
    return targets


def _collect_urls(value) -> List[str]:
    """Normalize a URL-like field into a list of URL strings."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def collect_book_template_image_urls(supabase_client) -> Set[str]:
    """
    Load all image URLs referenced by book templates.
    These URLs are shared assets and must never be deleted by story cleanup.
    """
    template_columns = (
        "cover_image,story_page_images,copyright_page_image,"
        "dedication_page_image,last_words_page_image,last_story_page_image,back_cover_image"
    )
    response = supabase_client.table("book_templates").select(template_columns).execute()
    template_rows = response.data or []

    urls: Set[str] = set()
    for row in template_rows:
        if not isinstance(row, dict):
            continue

        urls.update(_collect_urls(row.get("cover_image")))
        urls.update(_collect_urls(row.get("story_page_images")))
        urls.update(_collect_urls(row.get("copyright_page_image")))
        urls.update(_collect_urls(row.get("dedication_page_image")))
        urls.update(_collect_urls(row.get("last_words_page_image")))
        urls.update(_collect_urls(row.get("last_story_page_image")))
        urls.update(_collect_urls(row.get("back_cover_image")))
    return urls


def delete_story_images(
    supabase_client,
    story_data: dict,
    exclude_character_images: bool = True,
    protected_urls: Optional[Set[str]] = None,
) -> dict:
    """
    Delete story images from Supabase storage
    
    Args:
        supabase_client: Supabase client instance
        story_data: Story data dictionary from database
        exclude_character_images: If True, keep character image and enhancement images
        protected_urls: Optional set of shared URLs that must never be deleted
    
    Returns:
        Dict with deletion statistics
    """
    urls_to_delete = []
    
    # Collect story-specific files.
    urls_to_delete.extend(_collect_urls(story_data.get("story_cover")))
    urls_to_delete.extend(_collect_urls(story_data.get("scene_images")))
    urls_to_delete.extend(_collect_urls(story_data.get("dedication_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("copyright_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("last_word_page_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("last_admin_page_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("back_cover_image")))

    # Collect alias/template-style keys that may be present on stories.
    urls_to_delete.extend(_collect_urls(story_data.get("dedication_page_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("copyright_page_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("last_words_page_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("last_story_page_image")))
    urls_to_delete.extend(_collect_urls(story_data.get("story_page_images")))

    # Delete audio files.
    urls_to_delete.extend(_collect_urls(story_data.get("audio_url")))
    urls_to_delete.extend(_collect_urls(story_data.get("audio_urls")))

    # Delete PDF if exists.
    urls_to_delete.extend(_collect_urls(story_data.get("pdf_url")))
    
    # Character images (only delete if exclude_character_images is False)
    if not exclude_character_images:
        if story_data.get("original_image_url"):
            urls_to_delete.append(story_data["original_image_url"])
        
        if story_data.get("enhanced_images"):
            enhanced_images = story_data["enhanced_images"]
            if isinstance(enhanced_images, list):
                urls_to_delete.extend([img for img in enhanced_images if img])

    # Never delete shared template assets (e.g. dedication page from book_templates).
    protected_targets = _build_protected_storage_targets(protected_urls)
    shared_urls = []
    deletable_urls = []
    for url in urls_to_delete:
        target = _normalized_storage_target(url)
        is_protected_shared = bool(target and target in protected_targets)
        if _is_shared_template_asset(url) or is_protected_shared:
            shared_urls.append(url)
        else:
            deletable_urls.append(url)

    # Deduplicate while preserving order.
    deduped_urls = list(dict.fromkeys(deletable_urls))

    logger.info(
        "Deleting %s files for story (exclude_character_images=%s, skipped_shared_templates=%s)",
        len(deduped_urls),
        exclude_character_images,
        len(shared_urls),
    )
    if shared_urls:
        logger.info("Skipped shared template assets during story cleanup: %s", len(shared_urls))

    deletion_result = delete_files_from_storage(supabase_client, deduped_urls)
    deletion_result["skipped_shared_template_assets"] = len(shared_urls)
    return deletion_result


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
