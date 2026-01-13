"""
Character Database Models and Helper Functions
Python models and utilities for working with the characters table
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, HttpUrl


class CharacterExtractionData(BaseModel):
    """Model for character extraction data stored in JSONB field"""
    facial_features: Optional[Dict[str, Any]] = None
    clothing: Optional[List[str]] = None
    distinctive_features: Optional[List[str]] = None
    pose: Optional[str] = None
    extraction_model: Optional[str] = None
    extraction_timestamp: Optional[str] = None


class Character(BaseModel):
    """Character model matching the database schema"""
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Ownership
    user_id: str
    child_profile_id: Optional[int] = None
    
    # Basic info
    character_name: str
    character_type: str  # 'person' | 'animal' | 'magical_creature'
    special_ability: Optional[str] = None
    
    # Appearance
    character_style: str  # '3d' | 'cartoon' | 'anime'
    
    # Images
    original_image_url: str
    enhanced_images: Optional[List[str]] = []
    thumbnail_url: Optional[str] = None
    
    # Metadata
    age_group: Optional[str] = None  # '3-6' | '7-10' | '11-12'
    description: Optional[str] = None
    
    # Usage stats
    last_used_at: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    is_favorite: bool = False
    
    # AI data
    extraction_data: Optional[CharacterExtractionData] = None
    
    # Tags
    tags: Optional[List[str]] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "child_profile_id": 1,
                "character_name": "Super Sam",
                "character_type": "person",
                "special_ability": "Flying",
                "character_style": "cartoon",
                "original_image_url": "https://example.com/image.jpg",
                "enhanced_images": ["https://example.com/enhanced1.jpg"],
                "age_group": "7-10",
                "description": "A brave superhero",
                "tags": ["superhero", "brave"],
                "is_active": True,
                "is_favorite": False
            }
        }


class CharacterCreate(BaseModel):
    """Model for creating a new character"""
    user_id: str
    child_profile_id: Optional[int] = None
    character_name: str
    character_type: str
    special_ability: Optional[str] = None
    character_style: str
    original_image_url: str
    enhanced_images: Optional[List[str]] = []
    thumbnail_url: Optional[str] = None
    age_group: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = []
    extraction_data: Optional[Dict[str, Any]] = None


class CharacterUpdate(BaseModel):
    """Model for updating a character"""
    character_name: Optional[str] = None
    special_ability: Optional[str] = None
    description: Optional[str] = None
    enhanced_images: Optional[List[str]] = None
    thumbnail_url: Optional[str] = None
    is_active: Optional[bool] = None
    is_favorite: Optional[bool] = None
    tags: Optional[List[str]] = None
    extraction_data: Optional[Dict[str, Any]] = None


class CharacterResponse(BaseModel):
    """Response model for character operations"""
    success: bool
    data: Optional[Character] = None
    error: Optional[str] = None


class CharacterListResponse(BaseModel):
    """Response model for character list operations"""
    success: bool
    data: Optional[List[Character]] = []
    count: int = 0
    error: Optional[str] = None


class CharacterService:
    """Service class for character database operations"""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.table_name = "characters"
    
    def create_character(self, character_data: CharacterCreate) -> Dict[str, Any]:
        """
        Create a new character
        
        Args:
            character_data: Character creation data
            
        Returns:
            Dict with 'success', 'data', and optional 'error'
        """
        try:
            result = self.supabase.table(self.table_name).insert(
                character_data.model_dump(exclude_none=True)
            ).execute()
            
            if result.data:
                return {
                    "success": True,
                    "data": result.data[0]
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create character"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_character_by_id(self, character_id: int) -> Dict[str, Any]:
        """Get character by ID"""
        try:
            result = self.supabase.table(self.table_name).select("*").eq(
                "id", character_id
            ).single().execute()
            
            if result.data:
                return {
                    "success": True,
                    "data": result.data
                }
            else:
                return {
                    "success": False,
                    "error": "Character not found"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_characters_by_user(
        self,
        user_id: str,
        child_profile_id: Optional[int] = None,
        is_active: Optional[bool] = None,
        is_favorite: Optional[bool] = None,
        character_type: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get characters for a user with optional filters
        
        Args:
            user_id: User ID
            child_profile_id: Optional child profile filter
            is_active: Optional active status filter
            is_favorite: Optional favorite status filter
            character_type: Optional character type filter
            limit: Maximum number of results
            
        Returns:
            Dict with 'success', 'data', and optional 'error'
        """
        try:
            query = self.supabase.table(self.table_name).select("*").eq("user_id", user_id)
            
            if child_profile_id is not None:
                query = query.eq("child_profile_id", child_profile_id)
            
            if is_active is not None:
                query = query.eq("is_active", is_active)
            
            if is_favorite is not None:
                query = query.eq("is_favorite", is_favorite)
            
            if character_type is not None:
                query = query.eq("character_type", character_type)
            
            result = query.order("created_at", desc=True).limit(limit).execute()
            
            return {
                "success": True,
                "data": result.data or [],
                "count": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "count": 0
            }
    
    def update_character(
        self,
        character_id: int,
        updates: CharacterUpdate
    ) -> Dict[str, Any]:
        """Update a character"""
        try:
            result = self.supabase.table(self.table_name).update(
                updates.model_dump(exclude_none=True)
            ).eq("id", character_id).execute()
            
            if result.data:
                return {
                    "success": True,
                    "data": result.data[0]
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to update character"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_character_usage(self, character_id: int) -> Dict[str, Any]:
        """
        Increment times_used and update last_used_at
        This is called when a character is used in a new story
        """
        try:
            # Get current usage count
            char_result = self.get_character_by_id(character_id)
            if not char_result["success"]:
                return char_result
            
            # current_times_used = char_result["data"].get("times_used", 0)
            
            result = self.supabase.table(self.table_name).update({
                # "times_used": current_times_used + 1,
                "last_used_at": datetime.utcnow().isoformat()
            }).eq("id", character_id).execute()
            
            return {
                "success": True,
                "data": result.data[0] if result.data else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_extraction_data(
        self,
        character_id: int,
        extraction_data: Dict[str, Any],
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        Update character extraction data after AI processing
        
        Args:
            character_id: Character ID
            extraction_data: Extracted features from AI
            model_name: Name of the AI model used
            
        Returns:
            Dict with 'success', 'data', and optional 'error'
        """
        try:
            # Add metadata to extraction data
            extraction_data["extraction_model"] = model_name
            extraction_data["extraction_timestamp"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table(self.table_name).update({
                "extraction_data": extraction_data
            }).eq("id", character_id).execute()
            
            if result.data:
                return {
                    "success": True,
                    "data": result.data[0]
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to update extraction data"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_enhanced_image(
        self,
        character_id: int,
        image_url: str
    ) -> Dict[str, Any]:
        """
        Add an enhanced image to character's enhanced_images array
        
        Args:
            character_id: Character ID
            image_url: URL of the enhanced image
            
        Returns:
            Dict with 'success', 'data', and optional 'error'
        """
        try:
            # Get current enhanced images
            char_result = self.get_character_by_id(character_id)
            if not char_result["success"]:
                return char_result
            
            enhanced_images = char_result["data"].get("enhanced_images", [])
            enhanced_images.append(image_url)
            
            result = self.supabase.table(self.table_name).update({
                "enhanced_images": enhanced_images
            }).eq("id", character_id).execute()
            
            return {
                "success": True,
                "data": result.data[0] if result.data else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def soft_delete_character(self, character_id: int) -> Dict[str, Any]:
        """Soft delete a character (set is_active to False)"""
        return self.update_character(
            character_id,
            CharacterUpdate(is_active=False)
        )
    
    def hard_delete_character(self, character_id: int) -> Dict[str, Any]:
        """Hard delete a character (permanent)"""
        try:
            result = self.supabase.table(self.table_name).delete().eq(
                "id", character_id
            ).execute()
            
            return {
                "success": True,
                "data": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_characters(
        self,
        user_id: str,
        search_term: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Search characters by name or tags
        
        Args:
            user_id: User ID
            search_term: Search term for name or tags
            limit: Maximum number of results
            
        Returns:
            Dict with 'success', 'data', and optional 'error'
        """
        try:
            # Note: Supabase Python client might need different syntax for OR queries
            # This is a simplified version - adjust based on your Supabase client version
            result = self.supabase.table(self.table_name).select("*").eq(
                "user_id", user_id
            ).eq("is_active", True).ilike(
                "character_name", f"%{search_term}%"
            ).limit(limit).execute()
            
            return {
                "success": True,
                "data": result.data or [],
                "count": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "count": 0
            }
    
    def get_most_used_characters(
        self,
        user_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get most used characters for a user"""
        try:
            result = self.supabase.table(self.table_name).select("*").eq(
                "user_id", user_id
            ).eq("is_active", True).order(
                "times_used", desc=True
            ).limit(limit).execute()
            
            return {
                "success": True,
                "data": result.data or [],
                "count": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "count": 0
            }
    
    def get_recently_used_characters(
        self,
        user_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get recently used characters for a user"""
        try:
            result = self.supabase.table(self.table_name).select("*").eq(
                "user_id", user_id
            ).eq("is_active", True).not_.is_(
                "last_used_at", "null"
            ).order(
                "last_used_at", desc=True
            ).limit(limit).execute()
            
            return {
                "success": True,
                "data": result.data or [],
                "count": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "count": 0
            }


# Example usage in main.py or other files:
"""
from character_models import CharacterService, CharacterCreate

# Initialize service
character_service = CharacterService(supabase_client)

# Create character
new_character = CharacterCreate(
    user_id="user-123",
    child_profile_id=1,
    character_name="Super Sam",
    character_type="person",
    character_style="cartoon",
    original_image_url="https://example.com/image.jpg"
)

result = character_service.create_character(new_character)
if result["success"]:
    character_id = result["data"]["id"]
    
    # Update usage when used in a story
    character_service.update_character_usage(character_id)
    
    # Store extraction data after AI processing
    extraction_data = {
        "facial_features": {"eye_color": "blue"},
        "clothing": ["red cape"]
    }
    character_service.update_extraction_data(character_id, extraction_data)
"""

