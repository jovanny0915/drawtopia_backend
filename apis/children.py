"""
Children API routes
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from rate_limiter import limiter

router = APIRouter()


@router.get("/api/users/children")
@limiter.limit("120/minute")
async def list_child_profiles(request: Request, parent_id: Optional[str] = None):
    """
    List child profiles from the child_profiles table
    
    Args:
        parent_id: Optional parent user ID to filter children by parent
    
    Returns:
        List of child profile data, optionally filtered by parent
    """
    import main  # Import here to avoid circular import
    try:
        if not main.supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        # If parent_id is provided, filter by parent
        if parent_id:
            response = main.supabase.table("child_profiles").select("*").eq("parent_id", parent_id).execute()
        else:
            # Query all child profiles if no parent_id provided
            response = main.supabase.table("child_profiles").select("*").execute()
        
        if response.data is None:
            main.logger.warning("No child profiles found or query returned None")
            return []
        
        main.logger.info(f"Retrieved {len(response.data)} child profiles" + (f" for parent {parent_id}" if parent_id else ""))
        
        return response.data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        main.logger.error(f"Error listing child profiles: {e}")
        import traceback
        main.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error listing child profiles: {str(e)}")
