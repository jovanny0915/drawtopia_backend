"""
Supabase-based Queue Manager
Replaces Bull Queue with Supabase as the backend storage
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from supabase import Client
from enum import Enum
import ssl

logger = logging.getLogger(__name__)

# Maximum retries for SSL/connection errors
MAX_RETRIES = 3
RETRY_DELAY = 1  # Initial delay in seconds
MAX_RETRY_DELAY = 10  # Maximum delay in seconds


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageName(str, Enum):
    CHARACTER_EXTRACTION = "character_extraction"
    ENHANCEMENT = "enhancement"
    STORY_GENERATION = "story_generation"
    SCENE_CREATION = "scene_creation"
    CONSISTENCY_VALIDATION = "consistency_validation"
    AUDIO_GENERATION = "audio_generation"
    PDF_CREATION = "pdf_creation"


class QueueManager:
    """Manages job queue using Supabase as the backend"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def create_job(
        self,
        job_type: str,
        job_data: Dict[str, Any],
        user_id: Optional[str] = None,
        child_profile_id: Optional[int] = None,
        priority: int = 5,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Create a new job in the queue
        
        Args:
            job_type: 'interactive_search' or 'story_adventure'
            job_data: Dictionary containing job parameters
            user_id: User ID (optional)
            child_profile_id: Child profile ID (optional)
            priority: Job priority (1-10, 1 is highest)
            max_retries: Maximum number of retries
        
        Returns:
            Created job record
        """
        def _create():
            job_record = {
                "job_type": job_type,
                "status": JobStatus.PENDING.value,
                "priority": priority,
                "max_retries": max_retries,
                "retry_count": 0,
                "job_data": job_data,
            }
            
            if user_id:
                job_record["user_id"] = user_id
            if child_profile_id:
                job_record["child_profile_id"] = child_profile_id
            
            result = self.supabase.table("book_generation_jobs").insert(job_record).execute()
            
            if result.data and len(result.data) > 0:
                job = result.data[0]
                logger.info(f"Created job {job['id']} of type {job_type} with priority {priority}")
                return job
            else:
                raise Exception("Failed to create job: No data returned")
        
        try:
            return self._retry_on_ssl_error(_create)
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            raise
    
    def _retry_on_ssl_error(self, func, *args, **kwargs):
        """
        Retry a function call on SSL/connection errors with exponential backoff
        
        Args:
            func: Function to retry
            *args, **kwargs: Arguments to pass to the function
        
        Returns:
            Function result (can be None if function legitimately returns None)
        
        Raises:
            Exception: If all retries fail or if a non-SSL exception occurs
        """
        last_exception = None
        
        def _is_retryable_error(error_str: str) -> bool:
            """Check if the error is a retryable connection/SSL error"""
            retryable_keywords = [
                'ssl', 'eof', 'connection', 'unexpected_eof', 
                'disconnected', 'server disconnected', 'reset',
                'broken pipe', 'timed out', 'timeout', 'network'
            ]
            return any(keyword in error_str for keyword in retryable_keywords)
        
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (ssl.SSLError, ConnectionError, OSError, BrokenPipeError, TimeoutError) as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                if _is_retryable_error(error_str):
                    if attempt < MAX_RETRIES - 1:
                        delay = min(RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                        logger.warning(
                            f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Connection error after {MAX_RETRIES} attempts: {e}")
                        # Re-raise the exception after all retries fail
                        raise
                else:
                    # Not a retryable error, re-raise immediately
                    raise
            except Exception as e:
                # Check if it's a retryable error in the exception message
                error_str = str(e).lower()
                if _is_retryable_error(error_str):
                    last_exception = e
                    if attempt < MAX_RETRIES - 1:
                        delay = min(RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                        logger.warning(
                            f"Connection error in exception (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Connection error after {MAX_RETRIES} attempts: {e}")
                        raise
                else:
                    # Not a retryable error, re-raise immediately
                    raise
        
        # If we get here, all retries failed (shouldn't happen due to raise above)
        if last_exception:
            logger.error(f"Failed after {MAX_RETRIES} retries: {last_exception}")
            raise last_exception
    
    def get_next_job(self, job_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next pending job from the queue (highest priority first)
        
        Args:
            job_type: Optional filter by job type
        
        Returns:
            Job record or None if no jobs available
        """
        def _execute_query():
            query = self.supabase.table("book_generation_jobs").select("*")
            
            if job_type:
                query = query.eq("job_type", job_type)
            
            query = query.eq("status", JobStatus.PENDING.value)
            query = query.order("priority", desc=False)  # Lower priority number = higher priority
            query = query.order("created_at", desc=False)  # FIFO for same priority
            query = query.limit(1)
            
            result = query.execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        
        try:
            return self._retry_on_ssl_error(_execute_query)
        except (ssl.SSLError, ConnectionError, OSError) as e:
            logger.error(f"SSL/Connection error getting next job after retries: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting next job: {e}")
            return None
    
    def claim_job(self, job_id: int) -> bool:
        """
        Claim a job for processing (atomically update status to processing)
        
        Args:
            job_id: Job ID to claim
        
        Returns:
            True if successfully claimed, False otherwise
        """
        def _claim():
            result = self.supabase.table("book_generation_jobs").update({
                "status": JobStatus.PROCESSING.value,
                "started_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).eq("status", JobStatus.PENDING.value).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Claimed job {job_id} for processing")
                return True
            return False
        
        try:
            return self._retry_on_ssl_error(_claim)
        except (ssl.SSLError, ConnectionError, OSError) as e:
            logger.error(f"SSL/Connection error claiming job {job_id} after retries: {e}")
            return False
        except Exception as e:
            logger.error(f"Error claiming job {job_id}: {e}")
            return False
    
    def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job status
        
        Args:
            job_id: Job ID
            status: New status
            error_message: Optional error message
            result_data: Optional result data
        
        Returns:
            True if successful
        """
        def _update():
            update_data = {"status": status.value}
            
            if error_message:
                update_data["error_message"] = error_message
            
            if result_data:
                update_data["result_data"] = result_data
            
            if status == JobStatus.COMPLETED:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table("book_generation_jobs").update(update_data).eq("id", job_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Updated job {job_id} status to {status.value}")
                return True
            return False
        
        try:
            return self._retry_on_ssl_error(_update)
        except (ssl.SSLError, ConnectionError, OSError) as e:
            logger.error(f"SSL/Connection error updating job {job_id} status after retries: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating job {job_id} status: {e}")
            return False
    
    def increment_retry_count(self, job_id: int) -> bool:
        """Increment retry count for a job"""
        def _increment():
            # Get current retry count
            job = self.supabase.table("book_generation_jobs").select("retry_count, max_retries").eq("id", job_id).execute()
            
            if not job.data or len(job.data) == 0:
                return False
            
            current_retry = job.data[0]["retry_count"]
            max_retries = job.data[0]["max_retries"]
            
            new_retry_count = current_retry + 1
            
            # If exceeded max retries, mark as failed
            if new_retry_count >= max_retries:
                return self.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"Job failed after {max_retries} retries"
                )
            
            result = self.supabase.table("book_generation_jobs").update({
                "retry_count": new_retry_count,
                "status": JobStatus.PENDING.value  # Reset to pending for retry
            }).eq("id", job_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Incremented retry count for job {job_id} to {new_retry_count}")
                return True
            return False
        
        try:
            return self._retry_on_ssl_error(_increment)
        except (ssl.SSLError, ConnectionError, OSError) as e:
            logger.error(f"SSL/Connection error incrementing retry count for job {job_id} after retries: {e}")
            return False
        except Exception as e:
            logger.error(f"Error incrementing retry count for job {job_id}: {e}")
            return False
    
    def create_stage(
        self,
        job_id: int,
        stage_name: str,
        scene_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a job stage
        
        Args:
            job_id: Job ID
            stage_name: Stage name
            scene_index: Optional scene index for scene-specific stages
        
        Returns:
            Created stage record
        """
        def _create():
            stage_record = {
                "job_id": job_id,
                "stage_name": stage_name,
                "status": StageStatus.PENDING.value,
                "progress_percentage": 0
            }
            
            if scene_index is not None:
                stage_record["scene_index"] = scene_index
            
            result = self.supabase.table("job_stages").insert(stage_record).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                raise Exception("Failed to create stage: No data returned")
        
        try:
            return self._retry_on_ssl_error(_create)
        except Exception as e:
            logger.error(f"Error creating stage: {e}")
            raise
    
    def update_stage_status(
        self,
        stage_id: int,
        status: StageStatus,
        progress_percentage: Optional[int] = None,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update stage status
        
        Args:
            stage_id: Stage ID
            status: New status
            progress_percentage: Optional progress percentage (0-100)
            error_message: Optional error message
            result_data: Optional result data
        
        Returns:
            True if successful
        """
        def _update():
            update_data = {"status": status.value}
            
            if progress_percentage is not None:
                update_data["progress_percentage"] = progress_percentage
            
            if error_message:
                update_data["error_message"] = error_message
            
            if result_data:
                update_data["result_data"] = result_data
            
            if status == StageStatus.PROCESSING and "started_at" not in update_data:
                update_data["started_at"] = datetime.utcnow().isoformat()
            
            if status in [StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED]:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table("job_stages").update(update_data).eq("id", stage_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Updated stage {stage_id} status to {status.value}")
                return True
            return False
        
        try:
            return self._retry_on_ssl_error(_update)
        except (ssl.SSLError, ConnectionError, OSError) as e:
            logger.error(f"SSL/Connection error updating stage {stage_id} status after retries: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating stage {stage_id} status: {e}")
            return False
    
    def get_job_stages(self, job_id: int) -> List[Dict[str, Any]]:
        """Get all stages for a job"""
        def _get_stages():
            result = self.supabase.table("job_stages").select("*").eq("job_id", job_id).order("created_at", desc=False).execute()
            return result.data if result.data else []
        
        try:
            return self._retry_on_ssl_error(_get_stages)
        except (ssl.SSLError, ConnectionError, OSError) as e:
            logger.error(f"SSL/Connection error getting job stages after retries: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting job stages: {e}")
            return []
    
    def get_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get job status with all stages"""
        def _get_job():
            # Get job
            job_result = self.supabase.table("book_generation_jobs").select("*").eq("id", job_id).execute()
            
            if not job_result.data or len(job_result.data) == 0:
                return None
            
            job = job_result.data[0]
            
            # Get stages
            stages = self.get_job_stages(job_id)
            
            # Calculate overall progress
            if stages:
                completed_stages = sum(1 for s in stages if s["status"] == StageStatus.COMPLETED.value)
                total_stages = len(stages)
                overall_progress = int((completed_stages / total_stages) * 100) if total_stages > 0 else 0
            else:
                overall_progress = 0
            
            return {
                "job": job,
                "stages": stages,
                "overall_progress": overall_progress
            }
        
        try:
            return self._retry_on_ssl_error(_get_job)
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None

