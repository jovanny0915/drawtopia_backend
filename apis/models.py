"""
Pydantic models for API requests/responses
"""
from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class StoryRequest(BaseModel):
    character_name: str
    character_type: str
    special_ability: str
    age_group: str  # Must be "3-6", "7-10", or "11-12"
    story_world: str
    adventure_type: str
    occasion_theme: Optional[str] = None
    character_image_url: Optional[HttpUrl] = None  # Supabase URL of the character reference image
    story_text_prompt: Optional[str] = None  # Full prompt for story text generation (from frontend)
    scene_prompts: Optional[List[str]] = None  # List of 5 scene prompts, one for each page (from frontend)
    reading_level: Optional[str] = None  # Reading level (early_reader / developing_reader / independent_reader)
    story_title: Optional[str] = None  # Story title
    user_id: Optional[str] = None  # User ID for email notification
    child_profile_id: Optional[int] = None  # Child profile ID for database record
    character_style: Optional[str] = None  # Character style (3d/cartoon/anime)
    enhanced_images: Optional[List[str]] = None  # Enhanced character images
    dedication_text: Optional[str] = None  # Dedication page text
    dedication_scene_prompt: Optional[str] = None  # Dedication scene prompt according to story environment
    
    class Config:
        json_schema_extra = {
            "example": {
                "character_name": "Luna",
                "character_type": "a brave dragon",
                "special_ability": "fly through clouds",
                "age_group": "7-10",
                "story_world": "the Enchanted Forest",
                "adventure_type": "treasure hunt",
                "occasion_theme": None,
                "character_image_url": "https://your-project.supabase.co/storage/v1/object/public/images/character_reference.jpg",
                "story_text_prompt": "Create a personalized 5-page children's storybook...",
                "scene_prompts": ["Scene prompt for page 1...", "Scene prompt for page 2...", ],
                "reading_level": "developing_reader",
                "story_title": "The Great Adventure of Luna"
            }
        }
