"""
Character API routes
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from rate_limiter import limiter

router = APIRouter()


@router.get("/api/characters")
@limiter.limit("60/minute")
async def list_characters(request: Request, parent_id: Optional[str] = None):
    """
    List all created characters from the characters table with associated stories
    
    Args:
        parent_id: Required parent user ID to filter characters by parent
    
    Returns:
        List of character data with associated story information
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        # If parent_id is provided, filter by parent_id
        if parent_id:
            response = main.supabase.table("characters").select("*").eq("user_id", parent_id).execute()
        else:
            response = main.supabase.table("characters").select("*").execute()
        
        if not response or response.data is None:
            main.logger.warning("No characters found or query returned None")
            return []
        
        characters = response.data
        main.logger.info(f"Retrieved {len(characters)} characters for parent {parent_id}")
        
        # Enrich each character with associated story information
        enriched_characters = []
        for character in characters:
            character_id = character.get("id")
            
            # Get stories associated with this character
            if character_id:
                stories_response = main.supabase.table("stories").select("*").eq("user_id", parent_id).eq("character_id", character_id).execute()
                
                # Add stories to character data
                character_with_stories = {
                    **character,
                    "stories": stories_response.data if stories_response.data else []
                }
                enriched_characters.append(character_with_stories)
            else:
                # If no character_id, add empty stories list
                enriched_characters.append({
                    **character,
                    "stories": []
                })
        
        main.logger.info(f"Enriched {len(enriched_characters)} characters with story information")
        
        return enriched_characters
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error listing characters: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error listing characters: {str(e)}")


@router.delete("/api/characters/{character_id}")
@limiter.limit("30/minute")
async def delete_character(request: Request, character_id: str, user_id: Optional[str] = None):
    """
    Delete a character and update related stories
    
    Args:
        character_id: ID of the character to delete
        user_id: Optional user ID for authorization check
    
    Returns:
        Success message with details of the deletion
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        main.logger.info(f"Attempting to delete character {character_id} for user {user_id}")
        
        # First, verify the character exists and belongs to the user (if user_id provided)
        if user_id:
            character_response = main.supabase.table("characters").select("*").eq("id", character_id).eq("user_id", user_id).execute()
        else:
            character_response = main.supabase.table("characters").select("*").eq("id", character_id).execute()
        
        if not character_response.data or len(character_response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="Character not found or you don't have permission to delete it"
            )
        
        # Update all stories that reference this character - set character_id to null
        stories_update_response = main.supabase.table("stories").update({"character_id": None}).eq("character_id", character_id).execute()
        
        updated_stories_count = len(stories_update_response.data) if stories_update_response.data else 0
        main.logger.info(f"Updated {updated_stories_count} stories by removing character reference")
        
        # Delete the character
        delete_response = main.supabase.table("characters").delete().eq("id", character_id).execute()
        
        if not delete_response.data or len(delete_response.data) == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete character"
            )
        
        main.logger.info(f"Successfully deleted character {character_id}")
        
        return {
            "success": True,
            "message": "Character deleted successfully",
            "character_id": character_id,
            "stories_updated": updated_stories_count
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error deleting character: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error deleting character: {str(e)}")
