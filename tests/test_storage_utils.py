import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage_utils import delete_story_images, extract_storage_path_from_url


def _public_url(bucket: str, path: str) -> str:
    return f"https://example.supabase.co/storage/v1/object/public/{bucket}/{path}"


class _MockBucket:
    def __init__(self):
        self.removed_paths = []

    def remove(self, file_paths):
        self.removed_paths.extend(file_paths)
        return {"deleted": file_paths}


class _MockStorage:
    def __init__(self):
        self._buckets = {}

    def from_(self, bucket_name):
        if bucket_name not in self._buckets:
            self._buckets[bucket_name] = _MockBucket()
        return self._buckets[bucket_name]


class _MockSupabase:
    def __init__(self):
        self.storage = _MockStorage()


def test_delete_story_images_skips_shared_template_assets():
    client = _MockSupabase()
    story_data = {
        "story_cover": _public_url("images", "story_cover_1.jpg"),
        "scene_images": [
            _public_url("images", "story_scene_1.jpg"),
            _public_url("images", "book-templates/my-template/story-page-2.jpg"),
        ],
        "dedication_image": _public_url("images", "book-templates/my-template/dedication_page_image.jpg"),
        "copyright_image": _public_url("images", "book-templates/my-template/copyright_page_image.jpg"),
        "last_word_page_image": _public_url("images", "book-templates/my-template/last_words_page_image.jpg"),
        "last_admin_page_image": _public_url("images", "book-templates/my-template/last_story_page_image.jpg"),
        "back_cover_image": _public_url("images", "book-templates/my-template/back_cover_image.jpg"),
        "audio_url": [_public_url("audio", "story_audio_page1.mp3")],
        "pdf_url": _public_url("pdfs", "book_1.pdf"),
    }

    result = delete_story_images(client, story_data, exclude_character_images=True)

    # Only true story-owned assets should be removed.
    assert result["success"] == 4
    assert result["errors"] == 0
    assert result["skipped_shared_template_assets"] == 6

    image_deletes = client.storage.from_("images").removed_paths
    assert "story_cover_1.jpg" in image_deletes
    assert "story_scene_1.jpg" in image_deletes
    assert not any(path.startswith("book-templates/") for path in image_deletes)


def test_delete_story_images_respects_exclude_character_images_flag():
    story_data = {
        "story_cover": _public_url("images", "story_cover_2.jpg"),
        "original_image_url": _public_url("images", "character_original.jpg"),
        "enhanced_images": [_public_url("images", "character_enhanced_1.jpg")],
    }

    client_excluded = _MockSupabase()
    result_excluded = delete_story_images(client_excluded, story_data, exclude_character_images=True)
    assert result_excluded["success"] == 1
    assert client_excluded.storage.from_("images").removed_paths == ["story_cover_2.jpg"]

    client_included = _MockSupabase()
    result_included = delete_story_images(client_included, story_data, exclude_character_images=False)
    assert result_included["success"] == 3
    assert "character_original.jpg" in client_included.storage.from_("images").removed_paths
    assert "character_enhanced_1.jpg" in client_included.storage.from_("images").removed_paths


def test_extract_storage_path_from_signed_url():
    signed_url = (
        "https://example.supabase.co/storage/v1/object/sign/book-images/"
        "book-templates/my-template/dedication_page_image.webp?token=abc123"
    )
    parsed = extract_storage_path_from_url(signed_url)
    assert parsed == ("book-images", "book-templates/my-template/dedication_page_image.webp")


def test_delete_story_images_skips_protected_shared_urls_even_without_template_prefix():
    client = _MockSupabase()
    shared_dedication_url = _public_url("book-images", "legacy-template-assets/dedication_page_image.jpg")
    story_data = {
        "dedication_page_image": shared_dedication_url,
        "story_cover": _public_url("images", "story_cover_3.jpg"),
    }

    result = delete_story_images(
        client,
        story_data,
        exclude_character_images=True,
        protected_urls={shared_dedication_url},
    )

    assert result["success"] == 1
    assert result["errors"] == 0
    assert result["skipped_shared_template_assets"] == 1
    assert client.storage.from_("images").removed_paths == ["story_cover_3.jpg"]
    assert client.storage.from_("book-images").removed_paths == []
