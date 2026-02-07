"""
Pydantic models for API requests/responses
"""
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any


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


class StoryGenerateWithProgressRequest(StoryRequest):
    """Story generation request with session_id for WebSocket progress updates (percentage only)."""
    session_id: str


class SceneResult(BaseModel):
    """Individual scene result within search game results"""
    scene_index: int
    scene_title: str
    time: str  # Format: "M:SS" or seconds as string
    hint_used: int
    star_rate: int  # 0-3 stars


class StoryScenesRequest(BaseModel):
    """Request model for generating story scene images"""
    pages: List[str]  # Story page texts (output from generate-story-text)
    character_name: str
    character_type: str
    story_world: str
    character_image_url: Optional[HttpUrl] = None
    scene_prompts: Optional[List[str]] = None  # List of 5 scene prompts, one per page
    dedication_text: Optional[str] = None
    dedication_scene_prompt: Optional[str] = None


class StoryAudioRequest(BaseModel):
    """Request model for generating story audio"""
    pages: List[str]  # Story page texts (output from generate-story-text)
    age_group: str  # Must be "3-6", "7-10", or "11-12"


class StoryTitlesRequest(BaseModel):
    """Request model for generating story titles"""
    character_name: str
    special_ability: str
    story_world: str  # forest, outerspace, underwater
    adventure_type: str  # treasure, helping
    character_type: Optional[str] = "person"  # person, animal, magical
    character_style: Optional[str] = "cartoon"  # 3d, cartoon, anime
    story_format: Optional[str] = "story"  # story, interactive
    age_group: Optional[str] = "7-10"


class SaveStoryDraftRequest(BaseModel):
    """Request model for saving a story as draft (story-preview page)"""
    story_uid: Optional[str] = None  # If set and exists in DB, update instead of insert
    user_id: Optional[str] = None
    child_profile_id: str
    character_id: Optional[int] = None
    character_name: str
    character_type: str  # person, animal, magical_creature
    special_ability: Optional[str] = None
    character_style: str  # 3d, cartoon, anime
    story_world: str  # forest, space, underwater
    adventure_type: str  # treasure_hunt, helping_friend
    original_image_url: str
    enhanced_images: Optional[List[str]] = None
    story_title: Optional[str] = None
    story_cover: Optional[str] = None
    cover_design: Optional[str] = None
    story_type: Optional[str] = "story"  # story or search
    gift_id: Optional[str] = None
    purchased: Optional[bool] = False


class SetStoryGeneratingRequest(BaseModel):
    """Request model for setting story status to 'generating' (before generation starts)"""
    id: str  # Story ID (uid or numeric id)


class SearchGameResultRequest(BaseModel):
    """Request model for saving search game results"""
    character_id: int
    story_id: Optional[int] = None
    result: List[SceneResult]  # Array of scene results
    total_time: int  # Total time in seconds
    avg_stars: float  # Average stars (0.00 to 3.00)
    hints_used: int  # Total hints used
    best_scene: str  # Best scene title
    user_id: Optional[str] = None
    child_profile_id: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "character_id": 1,
                "story_id": 123,
                "result": [
                    {
                        "scene_index": 0,
                        "scene_title": "The Magical Forest",
                        "time": "2:30",
                        "hint_used": 1,
                        "star_rate": 3
                    },
                    {
                        "scene_index": 1,
                        "scene_title": "The Enchanted Castle",
                        "time": "3:15",
                        "hint_used": 2,
                        "star_rate": 2
                    }
                ],
                "total_time": 345,
                "avg_stars": 2.5,
                "hints_used": 3,
                "best_scene": "The Magical Forest",
                "user_id": "user-uuid-here",
                "child_profile_id": 1
            }
        }
