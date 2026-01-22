"""
Batch Processing Worker
Handles processing of book generation jobs with format-specific parallelization
"""

import logging
import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from queue_manager import QueueManager, JobStatus, StageStatus, StageName
from story_lib import generate_story
import requests
import base64
from io import BytesIO
from PIL import Image as PILImage
from pdf_generator import generate_pdf
from datetime import datetime
import uuid
import time
import httpx

logger = logging.getLogger(__name__)

# Helper function to call email API endpoints internally
async def call_email_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to call email API endpoints internally.
    This allows all email flows to go through the API layer.
    """
    try:
        # Get the base URL for internal API calls
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        api_url = f"{base_url}/api{endpoint}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling email API {endpoint}: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error calling email API {endpoint}: {e}")
        return {"success": False, "error": str(e)}


class BatchProcessor:
    """Processes book generation jobs in batches"""
    
    def __init__(
        self,
        queue_manager: QueueManager,
        gemini_client,
        openai_api_key: str,
        supabase_client,
        gemini_text_model: str = "gemini-2.5-flash"
    ):
        self.queue_manager = queue_manager
        self.gemini_client = gemini_client
        self.openai_api_key = openai_api_key
        self.supabase = supabase_client
        self.storage_bucket = "images"
        self.gemini_text_model = gemini_text_model
    
    async def process_job(self, job_id: int) -> bool:
        """
        Process a single job
        
        Args:
            job_id: Job ID to process
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Claim the job
            if not self.queue_manager.claim_job(job_id):
                logger.warning(f"Could not claim job {job_id}")
                return False
            
            # Get job details
            job_status = self.queue_manager.get_job_status(job_id)
            if not job_status:
                logger.error(f"Job {job_id} not found")
                return False
            
            job = job_status["job"]
            job_type = job["job_type"]
            job_data = job["job_data"]
            
            logger.info(f"Processing job {job_id} of type {job_type}")
            
            # Process based on job type
            if job_type == "interactive_search":
                success = await self._process_interactive_search(job_id, job_data)
            elif job_type == "story_adventure":
                success = await self._process_story_adventure(job_id, job_data)
            else:
                logger.error(f"Unknown job type: {job_type}")
                self.queue_manager.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"Unknown job type: {job_type}"
                )
                return False
            
            if success:
                self.queue_manager.update_job_status(job_id, JobStatus.COMPLETED)
                logger.info(f"Job {job_id} completed successfully")
                
                # Send book completion email
                await self._send_book_completion_email(job_id, job, job_data)
            else:
                # Check if we should retry
                if job["retry_count"] < job["max_retries"]:
                    self.queue_manager.increment_retry_count(job_id)
                    logger.info(f"Job {job_id} will be retried (attempt {job['retry_count'] + 1}/{job['max_retries']})")
                else:
                    self.queue_manager.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message="Job failed after all retries"
                    )
                    logger.error(f"Job {job_id} failed after all retries")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            self.queue_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(e)
            )
            return False
    
    async def _process_interactive_search(self, job_id: int, job_data: Dict[str, Any]) -> bool:
        """
        Process Interactive Search format (2 scenes simultaneously)
        
        Stages:
        1. character_extraction (M1)
        2. enhancement (M1)
        3. scene_creation (2 scenes in parallel)
        4. consistency_validation (2 scenes)
        5. pdf_creation (M3)
        """
        try:
            # Stage 1: Character Extraction
            stage_char_ext = self.queue_manager.create_stage(job_id, StageName.CHARACTER_EXTRACTION.value)
            self.queue_manager.update_stage_status(stage_char_ext["id"], StageStatus.PROCESSING)
            
            character_data = await self._extract_character(job_data)
            if not character_data:
                self.queue_manager.update_stage_status(
                    stage_char_ext["id"],
                    StageStatus.FAILED,
                    error_message="Character extraction failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_char_ext["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data=character_data
            )
            
            # Stage 2: Enhancement
            stage_enhance = self.queue_manager.create_stage(job_id, StageName.ENHANCEMENT.value)
            self.queue_manager.update_stage_status(stage_enhance["id"], StageStatus.PROCESSING)
            
            enhanced_images = await self._enhance_character(character_data, job_data)
            if not enhanced_images:
                self.queue_manager.update_stage_status(
                    stage_enhance["id"],
                    StageStatus.FAILED,
                    error_message="Character enhancement failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_enhance["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"enhanced_images": enhanced_images}
            )
            
            # Stage 3: Scene Creation (2 scenes in parallel)
            scene_stages = []
            for i in range(2):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.SCENE_CREATION.value,
                    scene_index=i
                )
                scene_stages.append(stage)
            
            # Process 2 scenes simultaneously
            scene_tasks = []
            for i, stage in enumerate(scene_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                task = self._create_scene(job_id, stage["id"], i, character_data, enhanced_images, job_data)
                scene_tasks.append(task)
            
            scene_results = await asyncio.gather(*scene_tasks, return_exceptions=True)
            
            scene_urls = []
            for i, (stage, result) in enumerate(zip(scene_stages, scene_results)):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                    return False
                else:
                    scene_urls.append(result)
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data={"scene_url": result}
                    )
            
            # Stage 4: Consistency Validation (2 scenes)
            validation_stages = []
            for i in range(2):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.CONSISTENCY_VALIDATION.value,
                    scene_index=i
                )
                validation_stages.append(stage)
            
            validation_tasks = []
            for i, stage in enumerate(validation_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                task = self._validate_consistency(job_id, stage["id"], i, scene_urls[i], enhanced_images[0] if enhanced_images else None)
                validation_tasks.append(task)
            
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for stage, result in zip(validation_stages, validation_results):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                else:
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data=result
                    )
            
            # Stage 5: PDF Creation (M3)
            stage_pdf = self.queue_manager.create_stage(job_id, StageName.PDF_CREATION.value)
            self.queue_manager.update_stage_status(stage_pdf["id"], StageStatus.PROCESSING)
            
            pdf_url = await self._create_pdf(job_id, job_data, scene_urls, enhanced_images=enhanced_images)
            if not pdf_url:
                self.queue_manager.update_stage_status(
                    stage_pdf["id"],
                    StageStatus.FAILED,
                    error_message="PDF creation failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_pdf["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"pdf_url": pdf_url}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing interactive search job {job_id}: {e}")
            return False
    
    async def _process_story_adventure(self, job_id: int, job_data: Dict[str, Any]) -> bool:
        """
        Process Story Adventure format (5 scenes in parallel)
        
        Stages:
        1. character_extraction (M1)
        2. enhancement (M1)
        3. story_generation (Story Adventure only)
        4. scene_creation (5 scenes in parallel)
        5. consistency_validation (5 scenes)
        6. audio_generation (Story Adventure - M3)
        7. pdf_creation (M3)
        """
        try:
            # Stage 1: Character Extraction
            stage_char_ext = self.queue_manager.create_stage(job_id, StageName.CHARACTER_EXTRACTION.value)
            self.queue_manager.update_stage_status(stage_char_ext["id"], StageStatus.PROCESSING)
            
            character_data = await self._extract_character(job_data)
            if not character_data:
                self.queue_manager.update_stage_status(
                    stage_char_ext["id"],
                    StageStatus.FAILED,
                    error_message="Character extraction failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_char_ext["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data=character_data
            )
            
            # Stage 2: Enhancement
            stage_enhance = self.queue_manager.create_stage(job_id, StageName.ENHANCEMENT.value)
            self.queue_manager.update_stage_status(stage_enhance["id"], StageStatus.PROCESSING)
            
            enhanced_images = await self._enhance_character(character_data, job_data)
            if not enhanced_images:
                self.queue_manager.update_stage_status(
                    stage_enhance["id"],
                    StageStatus.FAILED,
                    error_message="Character enhancement failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_enhance["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"enhanced_images": enhanced_images}
            )
            
            # Stage 3: Story Generation
            stage_story = self.queue_manager.create_stage(job_id, StageName.STORY_GENERATION.value)
            self.queue_manager.update_stage_status(stage_story["id"], StageStatus.PROCESSING)
            
            story_result = generate_story(
                character_name=job_data.get("character_name"),
                character_type=job_data.get("character_type"),
                special_ability=job_data.get("special_ability"),
                age_group=job_data.get("age_group"),
                story_world=job_data.get("story_world"),
                adventure_type=job_data.get("adventure_type"),
                occasion_theme=job_data.get("occasion_theme"),
                use_api=True,
                api_key=self.openai_api_key
            )
            
            if not story_result:
                self.queue_manager.update_stage_status(
                    stage_story["id"],
                    StageStatus.FAILED,
                    error_message="Story generation failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_story["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data=story_result
            )
            
            # Stage 4: Scene Creation (5 scenes in parallel)
            scene_stages = []
            for i in range(5):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.SCENE_CREATION.value,
                    scene_index=i
                )
                scene_stages.append(stage)
            
            # Process 5 scenes simultaneously
            scene_tasks = []
            for i, stage in enumerate(scene_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                page_text = story_result["pages"][i] if i < len(story_result["pages"]) else ""
                task = self._create_story_scene(
                    job_id,
                    stage["id"],
                    i,
                    page_text,
                    character_data,
                    enhanced_images,
                    job_data
                )
                scene_tasks.append(task)
            
            scene_results = await asyncio.gather(*scene_tasks, return_exceptions=True)
            
            scene_urls = []
            for i, (stage, result) in enumerate(zip(scene_stages, scene_results)):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                    return False
                else:
                    scene_urls.append(result)
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data={"scene_url": result}
                    )
            
            # Stage 5: Consistency Validation (5 scenes)
            validation_stages = []
            for i in range(5):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.CONSISTENCY_VALIDATION.value,
                    scene_index=i
                )
                validation_stages.append(stage)
            
            validation_tasks = []
            for i, stage in enumerate(validation_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                task = self._validate_consistency(job_id, stage["id"], i, scene_urls[i], enhanced_images[0] if enhanced_images else None)
                validation_tasks.append(task)
            
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for stage, result in zip(validation_stages, validation_results):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                else:
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data=result
                    )
            
            # Stage 6: Audio Generation (Story Adventure only)
            stage_audio = self.queue_manager.create_stage(job_id, StageName.AUDIO_GENERATION.value)
            self.queue_manager.update_stage_status(stage_audio["id"], StageStatus.PROCESSING)
            
            audio_result = await self._generate_audio(job_id, story_result, job_data)
            if not audio_result:
                self.queue_manager.update_stage_status(
                    stage_audio["id"],
                    StageStatus.FAILED,
                    error_message="Audio generation failed"
                )
                # Don't fail the entire job if audio fails, just log it
                logger.warning("Audio generation failed, but continuing with story generation")
            else:
                self.queue_manager.update_stage_status(
                    stage_audio["id"],
                    StageStatus.COMPLETED,
                    progress_percentage=100,
                    result_data=audio_result
                )
                # Store audio URLs in story_result for later use
                # audio_urls is a list of 5 URLs (one per page) or None for failed pages
                # When saving the story, include audioUrl in each page object:
                # story_content = {
                #   "pages": [
                #     {
                #       "pageNumber": 1,
                #       "text": "...",
                #       "sceneImage": "...",
                #       "audioUrl": audio_urls[0]  # or null if None
                #     },
                #     ...
                #   ]
                # }
                if "audio_urls" in audio_result:
                    story_result["audio_urls"] = audio_result["audio_urls"]
            
            # Stage 7: PDF Creation (M3)
            stage_pdf = self.queue_manager.create_stage(job_id, StageName.PDF_CREATION.value)
            self.queue_manager.update_stage_status(stage_pdf["id"], StageStatus.PROCESSING)
            
            pdf_url = await self._create_pdf(job_id, job_data, scene_urls, story_result, enhanced_images=enhanced_images)
            if not pdf_url:
                self.queue_manager.update_stage_status(
                    stage_pdf["id"],
                    StageStatus.FAILED,
                    error_message="PDF creation failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_pdf["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"pdf_url": pdf_url}
            )
            
            # Store final result data with all generated content including audio URLs
            final_result = {
                "story_result": story_result,
                "scene_urls": scene_urls,
                "audio_urls": story_result.get("audio_urls", []),
                "pdf_url": pdf_url
            }
            
            # Update job with final result data (audio URLs will be available for story saving)
            self.queue_manager.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result_data=final_result
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing story adventure job {job_id}: {e}")
            return False
    
    # Helper methods for processing stages
    async def _extract_character(self, job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract character from input (M1)"""
        # Placeholder - implement character extraction logic
        return {
            "character_name": job_data.get("character_name"),
            "character_type": job_data.get("character_type"),
            "original_image_url": job_data.get("character_image_url")
        }
    
    async def _enhance_character(self, character_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Enhance character images (M1)"""
        # Placeholder - implement enhancement logic
        # For now, return the original image URL
        if character_data.get("original_image_url"):
            return [character_data["original_image_url"]]
        return []
    
    
    async def _create_scene(
        self,
        job_id: int,
        stage_id: int,
        scene_index: int,
        character_data: Dict[str, Any],
        enhanced_images: List[str],
        job_data: Dict[str, Any]
    ) -> str:
        """Create a scene image for Interactive Search"""
        try:
            from image_utils import generate_story_scene_image
            
            # Create a simple scene description for interactive search
            scene_description = f"Scene {scene_index + 1} featuring {job_data.get('character_name', 'the character')} in {job_data.get('story_world', 'a magical world')}"
            
            reference_image_url = enhanced_images[0] if enhanced_images else character_data.get("original_image_url")
            
            scene_url = generate_story_scene_image(
                story_page_text=scene_description,
                page_number=scene_index + 1,
                character_name=job_data.get("character_name", ""),
                character_type=job_data.get("character_type", ""),
                story_world=job_data.get("story_world", ""),
                reference_image_url=reference_image_url,
                gemini_client=self.gemini_client,
                supabase_client=self.supabase,
                storage_bucket=self.storage_bucket
            )
            
            # Update progress
            self.queue_manager.update_stage_status(stage_id, StageStatus.PROCESSING, progress_percentage=100)
            
            return scene_url
            
        except Exception as e:
            logger.error(f"Error creating scene {scene_index} for job {job_id}: {e}")
            raise
    
    async def _create_story_scene(
        self,
        job_id: int,
        stage_id: int,
        scene_index: int,
        page_text: str,
        character_data: Dict[str, Any],
        enhanced_images: List[str],
        job_data: Dict[str, Any]
    ) -> str:
        """Create a story scene image"""
        try:
            from image_utils import generate_story_scene_image
            
            reference_image_url = enhanced_images[0] if enhanced_images else character_data.get("original_image_url")
            
            scene_url = generate_story_scene_image(
                story_page_text=page_text,
                page_number=scene_index + 1,
                character_name=job_data.get("character_name", ""),
                character_type=job_data.get("character_type", ""),
                story_world=job_data.get("story_world", ""),
                reference_image_url=reference_image_url,
                gemini_client=self.gemini_client,
                supabase_client=self.supabase,
                storage_bucket=self.storage_bucket
            )
            
            # Update progress
            self.queue_manager.update_stage_status(stage_id, StageStatus.PROCESSING, progress_percentage=100)
            
            return scene_url
            
        except Exception as e:
            logger.error(f"Error creating story scene {scene_index} for job {job_id}: {e}")
            raise
    
    async def _validate_consistency(
        self,
        job_id: int,
        stage_id: int,
        scene_index: int,
        scene_url: str,
        reference_image_url: Optional[str]
    ) -> Dict[str, Any]:
        """Validate character consistency"""
        try:
            if not reference_image_url:
                # Skip validation if no reference image
                return {
                    "is_consistent": True,
                    "similarity_score": 0.5,
                    "validation_available": False,
                    "message": "No reference image provided"
                }
            
            from image_utils import download_image_from_url
            from validation_utils import validate_character_consistency
            
            # Download images
            scene_image_data = download_image_from_url(scene_url)
            reference_image_data = download_image_from_url(reference_image_url)
            
            # Validate consistency
            validation_result = validate_character_consistency(
                scene_image_data=scene_image_data,
                reference_image_data=reference_image_data,
                page_number=scene_index + 1,
                gemini_client=self.gemini_client,
                gemini_text_model=self.gemini_text_model,
                timeout_seconds=15
            )
            
            # Update progress
            self.queue_manager.update_stage_status(stage_id, StageStatus.PROCESSING, progress_percentage=100)
            
            return {
                "is_consistent": validation_result.is_consistent,
                "similarity_score": validation_result.similarity_score,
                "validation_time_seconds": validation_result.validation_time_seconds,
                "flagged": validation_result.flagged,
                "details": validation_result.details
            }
            
        except Exception as e:
            logger.error(f"Error validating consistency for scene {scene_index} in job {job_id}: {e}")
            # Return a default result on error
            return {
                "is_consistent": True,
                "similarity_score": 0.5,
                "validation_available": False,
                "error": str(e)
            }
    
    async def _generate_audio(
        self,
        job_id: int,
        story_result: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> Optional[Dict[str, List[str]]]:
        """
        Generate audio for all story pages using Google Cloud Text-to-Speech
        
        Args:
            job_id: Job ID
            story_result: Story generation result with pages
            job_data: Job data containing age_group
        
        Returns:
            Dictionary with audio_urls list or None if failed
        """
        try:
            from audio_generator import AudioGenerator
            from datetime import datetime
            import uuid
            
            # Only generate audio for Story Adventure (not Interactive Search)
            # This check is already done at the job level, but keeping for safety
            if job_data.get("job_type") == "interactive_search":
                logger.info("Skipping audio generation for Interactive Search")
                return None
            
            age_group = job_data.get("age_group", "7-10")
            story_pages = story_result.get("pages", [])
            
            if not story_pages or len(story_pages) != 5:
                logger.error(f"Invalid story pages: expected 5, got {len(story_pages) if story_pages else 0}")
                return None
            
            # Initialize audio generator (gTTS - no API keys required)
            audio_generator = AudioGenerator()
            if not audio_generator.available:
                logger.error("Audio generator not available. Install: pip install gtts>=2.5.0")
                return None
            
            # Generate audio for all pages
            logger.info(f"Generating audio for {len(story_pages)} pages (age group: {age_group})...")
            audio_data_list = audio_generator.generate_audio_for_story(
                story_pages=story_pages,
                age_group=age_group,
                timeout_per_page=60
            )
            
            # Upload audio files to Supabase storage
            audio_urls = []
            
            for i, audio_data in enumerate(audio_data_list, 1):
                if audio_data is None:
                    logger.warning(f"⚠️ No audio generated for page {i}, skipping upload")
                    audio_urls.append(None)
                    continue
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"story_audio_page{i}_{timestamp}_{unique_id}.mp3"
                
                # Upload to Supabase storage
                # Create 'audio' bucket if it doesn't exist, or use 'images' bucket
                storage_bucket = "audio"  # You may need to create this bucket in Supabase
                try:
                    # Try audio bucket first, fallback to images bucket
                    try:
                        response = self.supabase.storage.from_(storage_bucket).upload(
                            filename,
                            audio_data,
                            {
                                'content-type': 'audio/mpeg',
                                'upsert': 'true'
                            }
                        )
                    except Exception as e:
                        # If audio bucket doesn't exist, use images bucket
                        logger.warning(f"Audio bucket not found, using images bucket: {e}")
                        storage_bucket = "images"
                        response = self.supabase.storage.from_(storage_bucket).upload(
                            filename,
                            audio_data,
                            {
                                'content-type': 'audio/mpeg',
                                'upsert': 'true'
                            }
                        )
                    
                    if hasattr(response, 'full_path') and response.full_path:
                        public_url = self.supabase.storage.from_(storage_bucket).get_public_url(filename)
                        audio_urls.append(public_url)
                        logger.info(f"✅ Uploaded audio for page {i}: {public_url}")
                    else:
                        logger.error(f"❌ Failed to upload audio for page {i}: Unexpected response")
                        audio_urls.append(None)
                except Exception as e:
                    logger.error(f"❌ Error uploading audio for page {i} to Supabase: {e}")
                    audio_urls.append(None)
            
            # Return audio URLs
            successful_uploads = sum(1 for url in audio_urls if url is not None)
            if successful_uploads > 0:
                logger.info(f"✅ Generated and uploaded {successful_uploads}/5 audio files")
                return {"audio_urls": audio_urls}
            else:
                logger.error("❌ Failed to generate/upload any audio files")
                return None
                
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def _create_pdf(
        self,
        job_id: int,
        job_data: Dict[str, Any],
        scene_urls: List[str],
        story_result: Optional[Dict[str, Any]] = None,
        enhanced_images: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create PDF (M3)
        
        For interactive_search: Creates PDF with cover + 4 scenes + back cover
        For story_adventure: Creates PDF with cover + 5 illustrated pages + audio info + back cover
        """
        try:
            start_time = time.time()
            logger.info(f"Starting PDF creation for job {job_id}")
            
            character_name = job_data.get("character_name", "Character")
            story_title = job_data.get("story_title") or f"{character_name}'s Adventure"
            character_image_url = job_data.get("character_image_url")
            
            # Determine PDF type based on whether story_result exists
            if story_result:
                # Story Adventure format
                pdf_type = "story_adventure"
                logger.info(f"Creating Story Adventure PDF with {len(scene_urls)} scenes")
                
                # Prepare story pages from story_result
                story_pages = []
                pages_data = story_result.get("pages", [])
                for i, page_data in enumerate(pages_data):
                    # Use scene URL from scene_urls if available, otherwise from page_data
                    scene_url = scene_urls[i] if i < len(scene_urls) else (page_data.get("scene") if isinstance(page_data, dict) else None)
                    page_text = page_data.get("text", "") if isinstance(page_data, dict) else str(page_data)
                    
                    story_pages.append({
                        "text": page_text,
                        "scene": scene_url
                    })
                
                audio_urls = story_result.get("audio_urls")
                
                # Generate PDF
                pdf_bytes = generate_pdf(
                    pdf_type=pdf_type,
                    character_name=character_name,
                    story_title=story_title,
                    character_image_url=character_image_url,
                    story_pages=story_pages,
                    audio_urls=audio_urls
                )
            else:
                # Interactive Search format
                pdf_type = "interactive_search"
                logger.info(f"Creating Interactive Search PDF with {len(scene_urls)} scenes")
                
                # For interactive search, we need cover image + 4 scenes
                # The first enhanced image is typically the cover, or use character_image_url
                # scene_urls should contain 4 scene URLs
                all_scene_urls = scene_urls[:4]  # Ensure we have exactly 4 scenes
                
                # Use first enhanced image as cover if available, otherwise use character_image_url
                cover_image_url = character_image_url
                if enhanced_images and len(enhanced_images) > 0:
                    cover_image_url = enhanced_images[0]
                
                # Generate PDF
                pdf_bytes = generate_pdf(
                    pdf_type=pdf_type,
                    character_name=character_name,
                    story_title=story_title,
                    character_image_url=cover_image_url,
                    scene_urls=all_scene_urls
                )
            
            if not pdf_bytes:
                logger.error(f"PDF generation failed for job {job_id}")
                return None
            
            # Upload PDF to Supabase storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"book_{pdf_type}_{job_id}_{timestamp}_{unique_id}.pdf"
            
            logger.info(f"Uploading PDF to Supabase storage: {filename}")
            pdf_url = await self._upload_pdf_to_storage(pdf_bytes, filename)
            
            if not pdf_url:
                logger.error(f"Failed to upload PDF for job {job_id}")
                return None
            
            elapsed = time.time() - start_time
            logger.info(f"✅ PDF created and uploaded successfully in {elapsed:.2f} seconds: {pdf_url}")
            
            # Ensure PDF creation completes within 30 seconds
            if elapsed > 30:
                logger.warning(f"PDF creation took {elapsed:.2f} seconds (exceeded 30s target)")
            
            # Update story/book record with PDF URL if book_id is available
            await self._update_story_with_pdf_url(job_id, pdf_url)
            
            return pdf_url
            
        except Exception as e:
            logger.error(f"Error creating PDF for job {job_id}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def _upload_pdf_to_storage(self, pdf_bytes: bytes, filename: str) -> Optional[str]:
        """Upload PDF to Supabase storage"""
        try:
            if not self.supabase:
                logger.error("Supabase client not available")
                return None
            
            # Upload to 'pdfs' bucket, fallback to 'images' bucket
            storage_bucket = "pdfs"
            try:
                response = self.supabase.storage.from_(storage_bucket).upload(
                    filename,
                    pdf_bytes,
                    {
                        'content-type': 'application/pdf',
                        'upsert': 'true'
                    }
                )
            except Exception as e:
                # Fallback to images bucket if pdfs bucket doesn't exist
                logger.warning(f"PDF bucket not found, using images bucket: {e}")
                storage_bucket = "images"
                response = self.supabase.storage.from_(storage_bucket).upload(
                    filename,
                    pdf_bytes,
                    {
                        'content-type': 'application/pdf',
                        'upsert': 'true'
                    }
                )
            
            if hasattr(response, 'full_path') and response.full_path:
                public_url = self.supabase.storage.from_(storage_bucket).get_public_url(filename)
                logger.info(f"✅ PDF uploaded successfully: {public_url}")
                return public_url
            else:
                logger.error(f"Unexpected Supabase response: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading PDF to storage: {e}")
            return None
    
    async def _update_story_with_pdf_url(self, job_id: int, pdf_url: str) -> bool:
        """Update story/book record with PDF URL"""
        try:
            if not self.supabase:
                logger.warning("Supabase client not available")
                return False
            
            # Get job to find book_id
            job_response = self.supabase.table("book_generation_jobs").select("book_id").eq("id", job_id).execute()
            
            if not job_response.data or len(job_response.data) == 0:
                logger.warning(f"Job {job_id} not found")
                return False
            
            book_id = job_response.data[0].get("book_id")
            
            if not book_id:
                logger.info(f"Job {job_id} has no book_id, skipping story update")
                return False
            
            # Update story with PDF URL
            update_response = self.supabase.table("stories").update({"pdf_url": pdf_url}).eq("id", book_id).execute()
            
            if update_response.data:
                logger.info(f"✅ Updated story {book_id} with PDF URL")
                return True
            else:
                logger.warning(f"Failed to update story {book_id} with PDF URL")
                return False
                
        except Exception as e:
            logger.error(f"Error updating story with PDF URL: {e}")
            return False
    
    async def _send_book_completion_email(self, job_id: int, job: Dict[str, Any], job_data: Dict[str, Any]):
        """Send book completion email after successful generation"""
        try:
            if not self.supabase:
                logger.warning("Supabase not available, skipping book completion email")
                return
            
            # Check if email service is enabled via API key
            
            # Get user and child profile information
            user_id = job.get("user_id")
            child_profile_id = job.get("child_profile_id")
            
            if not user_id:
                logger.warning(f"No user_id for job {job_id}, skipping book completion email")
                return
            
            # Get book/story details first
            story_result = self.supabase.table("stories").select("*").eq("job_id", job_id).execute()
            if not story_result.data or len(story_result.data) == 0:
                logger.warning(f"No story found for job {job_id}, skipping book completion email")
                return
            
            story = story_result.data[0]
            book_id = story.get("id")
            book_title = story.get("title", "Your Story")
            
            # Check if this book is linked to a gift order
            gift_result = self.supabase.table("gifts").select("*").eq("child_profile_id", str(child_profile_id)).execute()
            
            is_gift = False
            gift_data = None
            if gift_result.data and len(gift_result.data) > 0:
                # Find the most recent gift for this child profile that matches
                for gift in gift_result.data:
                    if gift.get("status") == "generating" or gift.get("status") == "completed":
                        is_gift = True
                        gift_data = gift
                        break
            
            # Generate preview and download links
            frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
            preview_link = f"{frontend_url}/story/{book_id}"
            download_link = f"{frontend_url}/api/books/{book_id}/download"
            
            if is_gift and gift_data:
                # This is a gift - send gift delivery email instead
                logger.info(f"Book {book_id} is a gift, sending gift delivery email")
                
                # Get giver details
                giver_id = gift_data.get("from_user_id") or gift_data.get("user_id")
                giver_result = self.supabase.table("users").select("first_name, last_name").eq("id", giver_id).execute()
                giver_name = "Someone special"
                if giver_result.data and len(giver_result.data) > 0:
                    giver = giver_result.data[0]
                    giver_name = f"{giver.get('first_name', '')} {giver.get('last_name', '')}".strip() or giver_name
                
                # Send gift delivery email via API
                if os.getenv("RESEND_API_KEY"):
                    try:
                        result = await call_email_api("/emails/gift-delivery", {
                            "to_email": gift_data.get("delivery_email"),
                            "recipient_name": gift_data.get("child_name", "there"),
                            "giver_name": giver_name,
                            "character_name": job_data.get("character_name", "Your Character"),
                            "character_type": job_data.get("character_type", "Character"),
                            "book_title": book_title,
                            "special_ability": job_data.get("special_ability", "special powers"),
                            "gift_message": gift_data.get("special_msg", "Enjoy your special story!"),
                            "story_link": preview_link,
                            "download_link": download_link,
                            "book_format": job.get("job_type", "story_adventure")
                        })
                        if result.get("success"):
                            logger.info(f"✅ Gift delivery email sent to {gift_data.get('delivery_email')} (Job {job_id})")
                            
                            # Update gift status to completed
                            self.supabase.table("gifts").update({"status": "completed"}).eq("id", gift_data.get("id")).execute()
                        else:
                            logger.error(f"❌ Failed to send gift delivery email: {result.get('error')}")
                    except Exception as email_error:
                        logger.error(f"❌ Failed to send gift delivery email: {email_error}")
                
                return
            
            # Not a gift - send regular book completion email
            # Get user details
            user_result = self.supabase.table("users").select("email, first_name, last_name").eq("id", user_id).execute()
            if not user_result.data or len(user_result.data) == 0:
                logger.warning(f"User {user_id} not found, skipping book completion email")
                return
            
            user = user_result.data[0]
            user_email = user.get("email")
            user_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "there"
            
            if not user_email:
                logger.warning(f"No email for user {user_id}, skipping book completion email")
                return
            
            # Get child profile details if available
            child_name = "your child"
            if child_profile_id:
                child_result = self.supabase.table("child_profiles").select("first_name").eq("id", child_profile_id).execute()
                if child_result.data and len(child_result.data) > 0:
                    child_name = child_result.data[0].get("first_name", child_name)
            
            # Send the email via API
            if os.getenv("RESEND_API_KEY"):
                try:
                    result = await call_email_api("/emails/book-completion", {
                        "to_email": user_email,
                        "parent_name": user_name,
                        "child_name": child_name,
                        "character_name": job_data.get("character_name", "Your Character"),
                        "character_type": job_data.get("character_type", "Character"),
                        "book_title": book_title,
                        "special_ability": job_data.get("special_ability", "special powers"),
                        "book_format": job.get("job_type", "story_adventure"),
                        "preview_link": preview_link,
                        "download_link": download_link,
                        "story_world": job_data.get("story_world"),
                        "adventure_type": job_data.get("adventure_type")
                    })
                    if result.get("success"):
                        logger.info(f"✅ Book completion email sent to {user_email} (Job {job_id})")
                    else:
                        logger.error(f"❌ Failed to send book completion email: {result.get('error')}")
                except Exception as email_error:
                    logger.error(f"❌ Failed to send book completion email: {email_error}")
                
        except Exception as e:
            logger.error(f"Error sending book completion email for job {job_id}: {e}")

