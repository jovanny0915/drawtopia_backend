from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import requests
import base64
import time
import uvicorn
import json
import re
from io import BytesIO
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import Header
import logging
import uuid
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image as PILImage
from google import genai
from google.genai import types
from google.genai.types import Image as GeminiImage
from apis import email_api
from story_lib import generate_story
from typing import List, Optional, Dict, Any
from queue_manager import QueueManager
from batch_processor import BatchProcessor
from validation_utils import ConsistencyValidationResult
from audio_generator import AudioGenerator
# Email service removed - all emails go through API endpoints now
import asyncio
from contextlib import asynccontextmanager
import httpx

# Import security utilities
from rate_limiter import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from security_utils import sanitize_input, sanitize_filename, validate_email, validate_phone, encrypt_data, decrypt_data
from virus_scanner import get_virus_scanner
import jwt
import stripe

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# MODEL = "gemini-2.5-flash"
MODEL = "gemini-3-pro-image-preview"
GEMINI_TEXT_MODEL = "gemini-2.5-flash"  # Model for text generation (scenes)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for storage operations
STORAGE_BUCKET = "images"

# Security Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-production")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", os.getenv("JWT_SECRET", "change-this-in-production"))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# CORS Configuration - use environment variables for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")

# Production mode check
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Stripe Configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID_MONTHLY = os.getenv("STRIPE_PRICE_ID_MONTHLY", "")
STRIPE_PRICE_ID_YEARLY = os.getenv("STRIPE_PRICE_ID_YEARLY", "")
STRIPE_PRICE_ID_SINGLE_STORY = os.getenv("STRIPE_PRICE_ID_SINGLE_STORY", "")
STRIPE_PRICE_ID_STORY_BUNDLE = os.getenv("STRIPE_PRICE_ID_STORY_BUNDLE", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Initialize Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe initialized successfully")
else:
    logger.warning("⚠️ STRIPE_SECRET_KEY not found. Stripe payments will be disabled.")

# Initialize Gemini client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY not found. Image generation will be disabled.")

supabase: Client = None
if SUPABASE_URL:
    # Try service key first (bypasses RLS), then anon key
    key_to_use = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
    key_type = "service" if SUPABASE_SERVICE_KEY else "anon"
    
    if key_to_use:
        try:
            supabase = create_client(SUPABASE_URL, key_to_use)
            logger.info(f"✅ Supabase client initialized successfully using {key_type} key")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
    else:
        logger.warning("⚠️ No Supabase key found (SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY)")
else:
    logger.warning("⚠️ Supabase URL not found. Storage upload will be disabled.")

# Initialize queue manager and batch processor
queue_manager = None
batch_processor = None
worker_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for background tasks"""
    global queue_manager, batch_processor, worker_task
    
    # Queue manager disabled - uncomment to re-enable
    # if supabase:
    #     queue_manager = QueueManager(supabase)
    #     
    #     # Initialize batch processor (without email queue manager)
    #     batch_processor = BatchProcessor(
    #         queue_manager=queue_manager,
    #         gemini_client=gemini_client,
    #         openai_api_key=OPENAI_API_KEY,
    #         supabase_client=supabase,
    #         gemini_text_model=GEMINI_TEXT_MODEL
    #     )
    #     logger.info("✅ Queue manager and batch processor initialized")
    #     
    #     # Start background worker
    #     worker_task = asyncio.create_task(background_worker())
    #     logger.info("✅ Background worker started")
    
    logger.info("✅ Server started (queue system disabled)")
    
    yield
    
    # Cleanup
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("✅ Background worker stopped")

# FastAPI app
app = FastAPI(
    title="AI Image Editor API",
    description="API for editing images using Google Gemini's image generation capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

# Add trusted host middleware (helps prevent invalid requests)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=ALLOWED_HOSTS
)

# Add CORS middleware with environment-based configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=[]
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# Global exception handler for better error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Handle validation errors
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format or data"}
    )

# Import and include API routers
# Note: Import AFTER app initialization to avoid circular imports
from apis.image import router as image_router
from apis.children import router as children_router
from apis.character import router as character_router
from apis.story import router as story_router

# Helper function to call email API endpoints internally
async def call_email_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to call email API endpoints internally.
    This allows all email flows to go through the API layer.
    """
    try:
        # Get the base URL for internal API calls
        # In production, this could be the actual server URL, but for internal calls we can use localhost
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
app.include_router(image_router)
app.include_router(children_router)
app.include_router(character_router)
app.include_router(story_router)

# Request model to receive input data
class ImageRequest(BaseModel):
    image_url: HttpUrl  # This validates the URL format
    prompt: str
    
    class Config:
        # Example values for API documentation
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "prompt": "Make this image more colorful and vibrant"
            }
        }

# Response model for image editing
class ImageResponse(BaseModel):
    success: bool
    message: str
    storage_info: dict = None
    quality_validation: Optional[Dict[str, Any]] = None

# Response model for quality validation
class QualityValidationResponse(BaseModel):
    success: bool
    validation: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "validation": {
                    "is_valid": True,
                    "quality_score": 0.85,
                    "is_appropriate": True,
                    "is_clear": True,
                    "has_sufficient_detail": True,
                    "issues": [],
                    "recommendations": ["Image quality is good"],
                    "details": {
                        "image_properties": {
                            "actual_resolution": "1024x768",
                            "format": "JPEG",
                            "clarity": "high"
                        }
                    }
                }
            }
        }

# Request model for story generation
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
                "scene_prompts": ["Scene prompt for page 1...", "Scene prompt for page 2...", ...],
                "reading_level": "developing_reader",
                "story_title": "The Great Adventure of Luna"
            }
        }

# Page model for story pages with text and scene image
class StoryPage(BaseModel):
    text: str
    scene: Optional[HttpUrl] = None  # URL to the generated scene image
    audio: Optional[HttpUrl] = None  # URL to the generated audio file
    consistency_validation: Optional[ConsistencyValidationResult] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Meet Luna, a brave dragon who loves adventures. Luna has a special power: Luna can fly through clouds.",
                "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page1_20240101_120000_abc123.jpg",
                "consistency_validation": {
                    "is_consistent": True,
                    "similarity_score": 0.85,
                    "validation_time_seconds": 3.2,
                    "flagged": False
                }
            }
        }

# Response model for story generation
class StoryResponse(BaseModel):
    success: bool
    pages: List[StoryPage]
    full_story: str
    word_count: int
    page_word_counts: List[int]
    consistency_summary: Optional[Dict[str, Any]] = None  # Overall validation summary
    audio_urls: Optional[List[Optional[str]]] = None  # List of audio URLs (one per page, None if failed)
    dedication_image_url: Optional[str] = None  # URL to the generated dedication image
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "pages": [
                    {
                        "text": "Meet Luna, a brave dragon who loves adventures. Luna has a special power: Luna can fly through clouds.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page1_20240101_120000_abc123.jpg"
                    },
                    {
                        "text": "While exploring, Luna discovered a magical entrance that led to the Enchanted Forest.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page2_20240101_120001_def456.jpg"
                    },
                    {
                        "text": "Suddenly, Luna realized that a treasure hunt was beginning, and Luna was right in the middle of it.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page3_20240101_120002_ghi789.jpg"
                    },
                    {
                        "text": "When the moment of truth arrived, Luna faced the challenge head-on, even though it seemed impossible at first.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page4_20240101_120003_jkl012.jpg"
                    },
                    {
                        "text": "The adventure came to a wonderful conclusion, and Luna felt proud of what had been accomplished.",
                        "scene": "https://your-project.supabase.co/storage/v1/object/public/images/story_scene_page5_20240101_120004_mno345.jpg"
                    }
                ],
                "full_story": "Meet Luna, a brave dragon who loves adventures...",
                "word_count": 250,
                "page_word_counts": [20, 25, 30, 28, 27]
            }
        }

def get_content_type_from_url(url):
    """Determine content type based on URL extension"""
    url_lower = url.lower()
    if url_lower.endswith(('.png', '.PNG')):
        return "image/png"
    elif url_lower.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        return "image/jpeg"
    elif url_lower.endswith(('.gif', '.GIF')):
        return "image/gif"
    elif url_lower.endswith(('.webp', '.WEBP')):
        return "image/webp"
    else:
        return "image/jpeg"  # default fallback

def detect_image_mime_type(image_data: bytes) -> str:
    """Detect MIME type from image bytes using PIL"""
    try:
        image = PILImage.open(BytesIO(image_data))
        format_to_mime = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'GIF': 'image/gif',
            'WEBP': 'image/webp',
            'BMP': 'image/bmp',
            'TIFF': 'image/tiff'
        }
        return format_to_mime.get(image.format, 'image/jpeg')
    except Exception as e:
        logger.warning(f"Could not detect image format, defaulting to image/jpeg: {e}")
        return "image/jpeg"

def download_image_from_url(url):
    """Download image from URL and return image data"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL {url}: {e}")

def optimize_image_to_jpg(image_data: bytes, quality: int = 85) -> bytes:
    """Convert and optimize image to JPG format with compression while preserving original resolution"""
    try:
        # Open image from bytes
        image = PILImage.open(BytesIO(image_data))
        original_size_info = f"{image.width}x{image.height}"
        
        # Convert to RGB if necessary (PNG with transparency, etc.)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = PILImage.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPG with compression (keeping original resolution)
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
        optimized_data = output_buffer.getvalue()
        
        # Log compression results
        original_size = len(image_data)
        optimized_size = len(optimized_data)
        compression_ratio = (1 - optimized_size / original_size) * 100
        logger.info(f"Image optimized ({original_size_info}): {original_size:,} bytes → {optimized_size:,} bytes ({compression_ratio:.1f}% reduction)")
        
        return optimized_data
        
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        # Return original data if optimization fails
        return image_data

def upload_to_supabase(image_data: bytes, filename: str, use_signed_url: bool = True) -> dict:
    """Upload image to Supabase storage and return signed or public URL"""
    if not supabase:
        logger.warning("Supabase client not available, skipping upload")
        return {"uploaded": False, "url": None, "message": "Supabase not configured"}

    try:
        # Sanitize filename
        filename = sanitize_filename(filename)
        logger.info(f"Uploading {filename} to Supabase storage bucket '{STORAGE_BUCKET}'")

        # Scan file for viruses
        scanner = get_virus_scanner()
        scan_result = scanner.scan_file(image_data, filename)
        if not scan_result["is_safe"]:
            logger.error(f"❌ File failed security scan: {scan_result['threats_found']}")
            return {
                "uploaded": False,
                "url": None,
                "message": f"File failed security scan: {', '.join(scan_result['threats_found'])}"
            }

        # Pass image_data directly as bytes to Supabase storage
        response = supabase.storage.from_(STORAGE_BUCKET).upload(filename, image_data, {
            'content-type' : 'image/jpeg',
            'upsert' : 'true'
        })

        # Check response type - response is an UploadResponse object
        if hasattr(response, 'full_path') and response.full_path:
            # Use signed URL with 30-day expiry for production
            if use_signed_url and IS_PRODUCTION:
                try:
                    signed_url_response = supabase.storage.from_(STORAGE_BUCKET).create_signed_url(
                        filename,
                        60 * 60 * 24 * 30  # 30 days in seconds
                    )
                    if signed_url_response and 'signedURL' in signed_url_response:
                        url = signed_url_response['signedURL']
                        logger.info(f"✅ Successfully uploaded with signed URL (30-day expiry)")
                    else:
                        # Fallback to public URL
                        url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(filename)
                        logger.warning("⚠️ Signed URL failed, using public URL")
                except Exception as e:
                    logger.warning(f"⚠️ Signed URL creation failed: {e}, using public URL")
                    url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(filename)
            else:
                url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(filename)
            
            logger.info(f"✅ Successfully uploaded to Supabase: {url[:100]}...")

            return {
                "uploaded": True,
                "url": url,
                "filename": filename,
                "bucket": STORAGE_BUCKET,
                "message": "Successfully uploaded to Supabase storage",
                "security_scan": scan_result
            }

        logger.error(f"❌ Unexpected Supabase response: {response}")
        return {"uploaded": False, "url": None, "message": f"Unexpected response: {response}"}

    except Exception as e:
        logger.error(f"❌ Error uploading to Supabase: {e}")
        return {"uploaded": False, "url": None, "message": f"Upload error: {e}"}

def edit_image(image_data, prompt, image_url=None):
    """Send image to Gemini API for editing/generation"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized. Please check GEMINI_API_KEY.")
    
    logger.info(f"Sending request to Gemini API with model: {MODEL}")
    
    try:
        start_time = time.time()
        
        # Detect MIME type from image data
        mime_type = detect_image_mime_type(image_data)
        logger.info(f"Detected image MIME type: {mime_type}")
        
        # Encode image to base64 for the dictionary format
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Generate content with Gemini API using the expected dictionary format
        # The API expects contents to be a list with role and parts
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Gemini API response received in {elapsed:.2f} seconds")
        
        # Extract image from response
        # Prioritize inline_data as it's the most direct source of image bytes
        edited_image_bytes = None
        for part in response.parts:
            if part.text is not None:
                logger.info(f"Gemini text response: {part.text}")
            
            # Check inline_data first - this is the most reliable source
            if hasattr(part, 'inline_data'):
                try:
                    inline_data = part.inline_data
                    logger.info(f"Found inline_data, type: {type(inline_data)}")
                    
                    # Try to get data from inline_data
                    if inline_data and hasattr(inline_data, 'data'):
                        data = inline_data.data
                        if isinstance(data, bytes):
                            edited_image_bytes = data
                            logger.info(f"✅ Image extracted from inline_data.data (bytes) ({len(edited_image_bytes)} bytes)")
                        elif isinstance(data, str):
                            # Try to decode base64
                            try:
                                edited_image_bytes = base64.b64decode(data)
                                logger.info(f"✅ Image extracted from inline_data.data (base64) ({len(edited_image_bytes)} bytes)")
                            except Exception as e:
                                logger.warning(f"Failed to decode base64 data: {e}")
                                # If it's not base64, try encoding as latin-1 (unlikely but possible)
                                edited_image_bytes = data.encode('latin-1')
                                logger.info(f"✅ Image extracted from inline_data.data (string) ({len(edited_image_bytes)} bytes)")
                    elif inline_data and hasattr(inline_data, 'bytes'):
                        edited_image_bytes = inline_data.bytes
                        logger.info(f"✅ Image extracted from inline_data.bytes ({len(edited_image_bytes)} bytes)")
                    
                    # Validate the extracted data
                    if edited_image_bytes and len(edited_image_bytes) > 1000:
                        logger.info(f"✅ Valid image extracted from inline_data ({len(edited_image_bytes)} bytes)")
                        break
                    elif edited_image_bytes:
                        logger.warning(f"Extracted data from inline_data is suspiciously small ({len(edited_image_bytes)} bytes), trying other methods...")
                        edited_image_bytes = None  # Reset to try other methods
                    else:
                        logger.warning(f"inline_data exists but no valid data found. inline_data attributes: {[a for a in dir(inline_data) if not a.startswith('_')]}")
                except Exception as e:
                    logger.warning(f"Error extracting from inline_data: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to as_image() if inline_data didn't work
            if not edited_image_bytes and hasattr(part, 'as_image'):
                try:
                    gemini_image = part.as_image()
                    logger.info(f"Got Gemini Image object: {type(gemini_image)}")
                    
                    # Check if it's already a PIL Image
                    if isinstance(gemini_image, PILImage.Image):
                        img_buffer = BytesIO()
                        gemini_image.save(img_buffer, format='PNG')
                        edited_image_bytes = img_buffer.getvalue()
                        logger.info(f"✅ Image extracted from PIL Image ({len(edited_image_bytes)} bytes)")
                        break
                    # Try to get bytes from Gemini Image object
                    elif hasattr(gemini_image, 'to_bytes'):
                        edited_image_bytes = gemini_image.to_bytes()
                    elif hasattr(gemini_image, 'bytes'):
                        edited_image_bytes = gemini_image.bytes
                    elif hasattr(gemini_image, 'data'):
                        data = gemini_image.data
                        if isinstance(data, bytes):
                            edited_image_bytes = data
                        elif isinstance(data, str):
                            edited_image_bytes = base64.b64decode(data)
                    else:
                        # Log available attributes for debugging
                        attrs = [a for a in dir(gemini_image) if not a.startswith('_')]
                        logger.warning(f"Gemini Image object attributes: {attrs}")
                        # Try accessing mime_type and data if they exist
                        if hasattr(gemini_image, 'mime_type') and hasattr(gemini_image, 'data'):
                            if isinstance(gemini_image.data, bytes):
                                edited_image_bytes = gemini_image.data
                            elif isinstance(gemini_image.data, str):
                                edited_image_bytes = base64.b64decode(gemini_image.data)
                    
                    # Validate size before accepting
                    if edited_image_bytes and len(edited_image_bytes) > 1000:
                        logger.info(f"✅ Image extracted from as_image() ({len(edited_image_bytes)} bytes)")
                        break
                    elif edited_image_bytes:
                        logger.warning(f"Extracted data from as_image() too small ({len(edited_image_bytes)} bytes), trying other methods...")
                        edited_image_bytes = None  # Reset to try other methods
                except Exception as e:
                    logger.warning(f"Error extracting from as_image(): {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if not edited_image_bytes:
            # Log more details for debugging
            logger.error(f"No valid image found in response. Response has {len(response.parts)} parts")
            for i, part in enumerate(response.parts):
                part_type = type(part).__name__
                attrs = [a for a in dir(part) if not a.startswith('_')]
                logger.error(f"Part {i}: type={part_type}, attributes={attrs}")
                # Try to log more details about each part
                if hasattr(part, 'inline_data'):
                    logger.error(f"  Part {i} inline_data: {part.inline_data}")
                if hasattr(part, 'text'):
                    logger.error(f"  Part {i} text: {part.text}")
            raise HTTPException(status_code=500, detail="No valid image was generated in the response from Gemini API")
        
        # Validate that we have a valid image before returning
        try:
            test_image = PILImage.open(BytesIO(edited_image_bytes))
            logger.info(f"✅ Validated image: {test_image.size[0]}x{test_image.size[1]}, format: {test_image.format}")
        except Exception as e:
            logger.error(f"Extracted data is not a valid image: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid image data extracted from Gemini API response: {str(e)}")
        
        return edited_image_bytes
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error from Gemini API: {str(e)}")

def validate_character_consistency(scene_image_data: bytes, reference_image_data: bytes, page_number: int, timeout_seconds: int = 15) -> ConsistencyValidationResult:
    """Wrapper for validation_utils.validate_character_consistency"""
    from validation_utils import validate_character_consistency as _validate_character_consistency
    return _validate_character_consistency(
        scene_image_data=scene_image_data,
        reference_image_data=reference_image_data,
        page_number=page_number,
        gemini_client=gemini_client,
        gemini_text_model=GEMINI_TEXT_MODEL,
        timeout_seconds=timeout_seconds
    )


def validate_image_quality(image_data: bytes, image_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate image quality using Gemini Vision API.
    Checks for: image quality, appropriateness, clarity, and basic properties.
    
    Returns a dictionary with validation results including:
    - is_valid: bool
    - quality_score: float (0-1)
    - issues: List[str]
    - recommendations: List[str]
    - details: Dict with specific checks
    """
    if not gemini_client:
        logger.warning("Gemini client not available for quality validation")
        return {
            "is_valid": True,  # Default to valid if validation unavailable
            "quality_score": 0.5,
            "issues": [],
            "recommendations": ["Quality validation unavailable - Gemini client not initialized"],
            "details": {"validation_available": False}
        }
    
    try:
        logger.info("Starting image quality validation with Gemini Vision API")
        
        # Detect MIME type
        mime_type = detect_image_mime_type(image_data)
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create validation prompt
        validation_prompt = """Analyze this image and provide a quality assessment in the following JSON format:
{
  "quality_score": <float 0.0-1.0>,
  "is_appropriate": <boolean>,
  "is_clear": <boolean>,
  "has_sufficient_detail": <boolean>,
  "issues": [<array of issue strings>],
  "recommendations": [<array of recommendation strings>],
  "image_properties": {
    "estimated_resolution": "<width>x<height>",
    "clarity": "<low/medium/high>",
    "brightness": "<too_dark/normal/too_bright>",
    "composition": "<poor/fair/good/excellent>"
  }
}

Focus on:
1. Image clarity and sharpness
2. Appropriate content for children (no violence, adult content, etc.)
3. Sufficient detail and resolution
4. Overall visual quality
5. Any technical issues (blur, distortion, artifacts)

Be strict but fair. Return ONLY valid JSON, no additional text."""
        
        # Call Gemini API for validation
        response = gemini_client.models.generate_content(
            model=GEMINI_TEXT_MODEL,  # Use text model for analysis
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": validation_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT'],
                temperature=0.1  # Lower temperature for more consistent validation
            )
        )
        
        # Extract text response
        validation_text = ""
        for part in response.parts:
            if part.text:
                validation_text += part.text
        
        # Parse JSON response
        # Try to extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
        if json_match:
            validation_json = json.loads(json_match.group())
        else:
            # Try parsing the whole response
            validation_json = json.loads(validation_text)
        
        # Extract validation results
        quality_score = validation_json.get("quality_score", 0.5)
        is_appropriate = validation_json.get("is_appropriate", True)
        is_clear = validation_json.get("is_clear", True)
        has_sufficient_detail = validation_json.get("has_sufficient_detail", True)
        issues = validation_json.get("issues", [])
        recommendations = validation_json.get("recommendations", [])
        image_properties = validation_json.get("image_properties", {})
        
        # Determine overall validity
        # Image is valid if: appropriate, clear, and quality score > 0.5
        is_valid = (
            is_appropriate and 
            is_clear and 
            quality_score >= 0.5 and
            has_sufficient_detail
        )
        
        # Add basic image properties from PIL
        try:
            pil_image = PILImage.open(BytesIO(image_data))
            image_properties["actual_resolution"] = f"{pil_image.width}x{pil_image.height}"
            image_properties["format"] = pil_image.format or "unknown"
            image_properties["mode"] = pil_image.mode
            image_properties["file_size_bytes"] = len(image_data)
        except Exception as e:
            logger.warning(f"Could not extract PIL image properties: {e}")
        
        result = {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "is_appropriate": is_appropriate,
            "is_clear": is_clear,
            "has_sufficient_detail": has_sufficient_detail,
            "issues": issues,
            "recommendations": recommendations,
            "details": {
                "image_properties": image_properties,
                "validation_available": True,
                "model_used": GEMINI_TEXT_MODEL
            }
        }
        
        logger.info(f"Quality validation completed: valid={is_valid}, score={quality_score:.2f}")
        if issues:
            logger.info(f"Validation issues found: {', '.join(issues)}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse validation JSON response: {e}")
        logger.error(f"Response text: {validation_text[:500] if 'validation_text' in locals() else 'N/A'}")
        return {
            "is_valid": True,  # Default to valid on parse error
            "quality_score": 0.5,
            "issues": ["Could not parse validation response"],
            "recommendations": ["Validation service error - proceeding with caution"],
            "details": {"validation_available": False, "error": "JSON parse error"}
        }
    except Exception as e:
        logger.error(f"Error during quality validation: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {
            "is_valid": True,  # Default to valid on error
            "quality_score": 0.5,
            "issues": [f"Validation error: {str(e)}"],
            "recommendations": ["Validation service error - proceeding with caution"],
            "details": {"validation_available": False, "error": str(e)}
        }

def create_blank_base_image(width: int = 768, height: int = 512) -> bytes:
    """Create a blank white image in 768x512 dimensions to use as base for image generation"""
    try:
        # Create a white image in 768x512 dimensions (default)
        blank_image = PILImage.new('RGB', (width, height), color=(255, 255, 255))
        img_buffer = BytesIO()
        blank_image.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error creating blank base image: {e}")
        raise

def get_environment_details(story_world: str) -> str:
    """Get environment-specific details based on story world."""
    world_lower = story_world.lower()
    if 'enchanted forest' in world_lower or world_lower == 'forest':
        return "ENVIRONMENT DETAILS: Include magical trees with glowing elements, mystical flora, enchanted atmosphere with soft magical light, fairy-tale forest setting with whimsical details."
    elif 'outer space' in world_lower or world_lower == 'space':
        return "ENVIRONMENT DETAILS: Include planets, stars, alien landscapes, cosmic scenery, space nebulas, celestial bodies, and otherworldly terrain."
    elif 'underwater kingdom' in world_lower or world_lower == 'underwater':
        return "ENVIRONMENT DETAILS: Include coral reefs, sea creatures, underwater flora, aquatic plants, marine life, and oceanic elements."
    else:
        return "ENVIRONMENT DETAILS: Match the setting and atmosphere of the story world."

def generate_story_scene_image(story_page_text: str, page_number: int, character_name: str, character_type: str, story_world: str, reference_image_url: Optional[str] = None, scene_prompt: Optional[str] = None) -> str:
    """Generate a scene image for a story page using edit_image function and return the image URL.
    
    If scene_prompt is provided, use it; otherwise generate prompt from parameters.
    """
    if not gemini_client:
        logger.warning("Gemini client not available, returning empty scene URL")
        return ""
    
    logger.info(f"Generating scene image for page {page_number} using edit_image function")
    if reference_image_url:
        logger.info(f"Using reference character image from: {reference_image_url}")
    
    try:
        # Get base image - use reference image if provided, otherwise create a blank image
        base_image_data = None
        if reference_image_url:
            try:
                logger.info(f"Downloading reference image from: {reference_image_url}")
                base_image_data = download_image_from_url(reference_image_url)
                logger.info(f"✅ Reference image downloaded successfully ({len(base_image_data)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to download reference image, creating blank base image: {e}")
                base_image_data = None
        
        # If no reference image, create a blank white image in 768x512 dimensions
        if not base_image_data:
            logger.info("Creating blank base image for scene generation")
            base_image_data = create_blank_base_image()
            logger.info(f"✅ Blank base image created ({len(base_image_data)} bytes)")
        
        # Use provided prompt if available, otherwise generate one (for backward compatibility)
        if scene_prompt:
            # Enhance custom prompt with explicit text requirement
            text_requirement = f"""

CRITICAL TEXT REQUIREMENT:
- The EXACT story page text below MUST be visibly embedded and displayed within the image:
"{story_page_text}"
- The text should be clearly readable, using a child-friendly font style
- Place the text in a prominent location (typically at the bottom or top of the image, or integrated naturally into the scene)
- Use contrasting colors for text readability (e.g., white text on dark backgrounds, dark text on light backgrounds)
- The text should be large enough to be easily read by children
- Integrate the text naturally into the illustration style, making it look like part of a storybook page
- DO NOT omit or modify any part of the story text - include it exactly as provided above
- TEXT VISIBILITY: Ensure the story text is clearly visible and readable, with proper contrast against the background
"""
            prompt = scene_prompt + text_requirement
            logger.info(f"Using scene prompt from frontend for page {page_number} (enhanced with text requirement)")
        else:
            # Fallback: generate prompt from parameters (for backward compatibility)
            character_reference_note = ""
            character_consistency_enforcement = ""
            negative_prompts = ""
            
            if reference_image_url and base_image_data:
                character_reference_note = f"""
CHARACTER REFERENCE:
- A reference image of {character_name} is provided below
- Use this reference image to maintain consistent character appearance across all scenes
- The character in the scene must match the appearance, style, and features shown in the reference image
- Keep the character's visual identity consistent with the reference image
"""
                character_consistency_enforcement = f"""
=== MANDATORY CHARACTER STYLE CONSISTENCY REQUIREMENTS ===
CRITICAL: The character from the provided reference image MUST be embedded with EXACT visual fidelity.

REQUIRED CHARACTER FEATURES (DO NOT CHANGE):
* Face: Exact same facial features, eye shape, nose, mouth, and expression style as reference
* Limbs: Exact same proportions, length, and structure as reference
* Body proportions: Exact same height-to-width ratio and body shape as reference
* Hair: Exact same hair style, color, texture, and length as reference
* Skin tone: Exact same skin color and tone as reference
* Clothing: Exact same clothing design, colors, patterns, and details as reference
* Overall design: Exact same character design language, style, and visual identity as reference
* Anatomy: Exact same anatomical structure - no changes to bone structure, muscle definition, or body type
* Style: The character's artistic style must remain consistent with the reference image

STRICT PROHIBITIONS:
* DO NOT alter the character's facial features
* DO NOT change the character's body proportions or anatomy
* DO NOT modify the character's hair style, color, or texture
* DO NOT change the character's skin tone or color
* DO NOT alter the character's clothing design, colors, or patterns
* DO NOT modify the character's overall design or visual identity
* DO NOT apply different artistic styles to the character than what appears in the reference
* DO NOT distort, stretch, or resize the character in ways that change their appearance
* DO NOT add features not present in the reference image
* DO NOT remove features present in the reference image

ENFORCEMENT:
The character must be reproduced with pixel-perfect fidelity to the reference image. Any deviation from the reference character's appearance is strictly prohibited. The scene style may vary, but the character's appearance must remain identical to the reference image in all aspects.
"""
                negative_prompts = """
=== NEGATIVE PROMPTS (STRICTLY AVOID) ===
DO NOT:
* Alter the character's facial features, proportions, or anatomy
* Change the character's hair style, color, or texture
* Modify the character's skin tone or color
* Alter the character's clothing design, colors, or patterns
* Change the character's body proportions or structure
* Apply different artistic styles to the character than the reference
* Distort, stretch, or resize the character in ways that change appearance
* Add features not present in the reference image
* Remove features present in the reference image
* Create variations of the character - use the exact reference character only
"""
            
            environment_details = get_environment_details(story_world)
            
            prompt = f"""Create a beautiful, colorful children's storybook illustration for this story page.

STORY PAGE TEXT (Page {page_number}) - MUST BE INSERTED INTO IMAGE:
{story_page_text}

CRITICAL TEXT REQUIREMENT:
- The EXACT story page text above MUST be visibly embedded and displayed within the image
- The text should be clearly readable, using a child-friendly font style
- Place the text in a prominent location (typically at the bottom or top of the image, or integrated naturally into the scene)
- Use contrasting colors for text readability (e.g., white text on dark backgrounds, dark text on light backgrounds)
- The text should be large enough to be easily read by children
- Integrate the text naturally into the illustration style, making it look like part of a storybook page
- DO NOT omit or modify any part of the story text - include it exactly as provided above

CHARACTER INFORMATION:
- Character Name: {character_name}
- Character Type: {character_type}
- Story World: {story_world}
{environment_details}
{character_reference_note}
{character_consistency_enforcement}
ILLUSTRATION REQUIREMENTS:
1. Create a vibrant, age-appropriate children's book illustration
2. Include the main character ({character_name}) as a {character_type} - {character_name} is the clear hero of this story
3. CHARACTER PROMINENCE: The character ({character_name}) must occupy 60-70% of the composition. The character should be the dominant visual element, clearly visible and prominent in the scene
4. Match the mood, setting, and events from the story text
5. Use bright, cheerful colors suitable for children
6. Make it visually appealing and engaging
7. Ensure the scene is positive and appropriate for children
8. Include relevant details about the setting and characters
9. Style should be like a professional children's book illustration
10. IMPORTANT: The image must be in 768x512 dimensions
{"11. CRITICAL: The character must match the appearance shown in the reference image provided" if reference_image_url and base_image_data else ""}
12. TEXT VISIBILITY: Ensure the story text is clearly visible and readable, with proper contrast against the background
{negative_prompts}

Generate a high-quality illustration that perfectly captures this story moment in 768x512 dimensions, with the story text clearly visible and integrated into the image."""

        # Use edit_image function to generate the scene
        logger.info(f"Calling edit_image function with prompt for page {page_number}")
        scene_image_bytes = edit_image(base_image_data, prompt, reference_image_url)
        
        # Optimize image to JPG format
        logger.info("Optimizing scene image to JPG format...")
        optimized_image = optimize_image_to_jpg(scene_image_bytes)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"story_scene_page{page_number}_{timestamp}_{unique_id}.jpg"
        
        # Upload to Supabase and get URL
        storage_result = upload_to_supabase(optimized_image, filename)
        
        if storage_result.get("uploaded") and storage_result.get("url"):
            logger.info(f"✅ Scene image generated and uploaded for page {page_number}: {storage_result['url']}")
            return storage_result['url']
        else:
            logger.warning(f"Failed to upload scene image for page {page_number}")
            return ""
        
    except HTTPException as e:
        logger.error(f"HTTP error generating scene image for page {page_number}: {e.detail}")
        return ""
    except Exception as e:
        logger.error(f"Error generating scene image for page {page_number}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return ""

def create_jwt_token(user_id: str, additional_claims: Optional[Dict] = None) -> str:
    """
    Create JWT token with expiration
    
    Args:
        user_id: User ID to encode in token
        additional_claims: Additional claims to include
        
    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    
    if additional_claims:
        payload.update(additional_claims)
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict]:
    """
    Verify and decode JWT token. Tries Supabase JWT secret first, then custom JWT secret.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded payload or None if invalid
    """
    # Try with Supabase JWT secret first (for tokens from frontend)
    secrets_to_try = [SUPABASE_JWT_SECRET, JWT_SECRET]
    
    for secret in secrets_to_try:
        try:
            payload = jwt.decode(
                token, 
                secret, 
                algorithms=[JWT_ALGORITHM],
                options={"verify_aud": False}  # Supabase tokens may have audience claim
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            # Try next secret
            continue
    
    # If all secrets failed, try to decode without verification to get user info
    # This is a fallback for development/debugging - in production, ensure SUPABASE_JWT_SECRET is set
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        logger.warning("JWT signature verification failed, but decoded without verification for user extraction")
        return payload
    except Exception as e:
        logger.warning(f"Failed to decode JWT token: {e}")
        return None


def extract_user_from_token(authorization: Optional[str]) -> Optional[str]:
    """
    Extract user ID from Authorization header
    
    Args:
        authorization: Authorization header value
        
    Returns:
        User ID or None
    """
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            payload = verify_jwt_token(token)
            if payload:
                return payload.get("sub")
    except Exception as e:
        logger.warning(f"Error extracting user from token: {e}")
    
    return None


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """Root endpoint with API information"""
    return {
        "message": "AI Image Editor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "security": {
            "rate_limiting": "enabled",
            "jwt_expiration": f"{JWT_EXPIRATION_HOURS} hours"
        },
        "email_service": {
            "enabled": bool(os.getenv("RESEND_API_KEY")),
            "provider": "Resend API"
        }
    }


@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    scanner = get_virus_scanner()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_client_initialized": bool(gemini_client is not None),
        "openai_api_key_configured": bool(OPENAI_API_KEY),
        "model": MODEL,
        "supabase_configured": bool(supabase is not None),
        "storage_bucket": STORAGE_BUCKET if supabase else None,
        "quality_validation_enabled": bool(gemini_client is not None),
        "virus_scanner_available": scanner.is_available(),
        "security": {
            "rate_limiting": "enabled",
            "virus_scanning": "enabled" if scanner.is_available() else "basic_checks_only"
        }
    }

# Request model for batch job creation
class BatchJobRequest(BaseModel):
    job_type: str  # 'interactive_search' or 'story_adventure'
    character_name: str
    character_type: str
    special_ability: str
    age_group: str
    story_world: str
    adventure_type: str
    occasion_theme: Optional[str] = None
    character_image_url: Optional[HttpUrl] = None
    priority: int = 5  # 1-10, 1 is highest
    user_id: Optional[str] = None
    child_profile_id: Optional[int] = None

# Response model for job creation
class JobResponse(BaseModel):
    success: bool
    job_id: int
    message: str

# Response model for job status
class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    overall_progress: int
    stages: List[Dict[str, Any]]
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None

# Response model for PDF generation
class PDFGenerationResponse(BaseModel):
    success: bool
    pdf_url: Optional[str] = None
    message: str

async def background_worker():
    """Background worker that processes jobs from the queue (DISABLED)"""
    logger.info("Background worker disabled - queue system not in use")
    return
    
    # Original code - uncomment to re-enable queue processing
    # logger.info("Background worker started")
    # while True:
    #     try:
    #         if not queue_manager:
    #             await asyncio.sleep(5)
    #             continue
    #         
    #         # Get next job
    #         job = queue_manager.get_next_job()
    #         
    #         if job:
    #             job_id = job["id"]
    #             logger.info(f"Processing job {job_id}")
    #             await batch_processor.process_job(job_id)
    #         else:
    #             # No jobs available, wait before checking again
    #             await asyncio.sleep(2)
    #             
    #     except asyncio.CancelledError:
    #         logger.info("Background worker cancelled")
    #         break
    #     except Exception as e:
    #         logger.error(f"Error in background worker: {e}")
    #         await asyncio.sleep(5)


@app.get("/api/dashboard/user-statistics")
@limiter.limit("30/minute")
async def get_user_statistics(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get user statistics for dashboard
    
    Args:
        start_date: ISO format date string (optional) - filter users created after this date
        end_date: ISO format date string (optional) - filter users created before this date
    
    Returns:
        Dictionary with user statistics including:
        - Total registered users
        - New users (daily/weekly/monthly)
        - Active users (users who created stories/books)
        - User role distribution
        - Users by subscription status
    """
    try:
        if not supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        logger.info(f"Fetching user statistics (start_date={start_date}, end_date={end_date})")
        
        # === TOTAL REGISTERED USERS ===
        users_query = supabase.table("users").select("id, created_at, role, subscription_status")
        
        # Apply date filters if provided
        if start_date:
            users_query = users_query.gte("created_at", start_date)
        if end_date:
            users_query = users_query.lte("created_at", end_date)
        
        users_response = users_query.execute()
        all_users = users_response.data if users_response.data else []
        total_users = len(all_users)
        
        logger.info(f"Total users found: {total_users}")
        
        # === USER ROLE DISTRIBUTION ===
        role_distribution = {}
        subscription_distribution = {}
        
        for user in all_users:
            role = user.get('role', 'unknown')
            role_distribution[role] = role_distribution.get(role, 0) + 1
            
            sub_status = user.get('subscription_status') or 'free'
            subscription_distribution[sub_status] = subscription_distribution.get(sub_status, 0) + 1
        
        # === NEW USERS (DAILY/WEEKLY/MONTHLY) ===
        from datetime import datetime, timedelta
        
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).isoformat()
        last_week = (now - timedelta(days=7)).isoformat()
        last_month = (now - timedelta(days=30)).isoformat()
        
        # New users in last 24 hours
        new_users_daily_response = supabase.table("users").select("id", count="exact").gte("created_at", yesterday).execute()
        new_users_daily = len(new_users_daily_response.data) if new_users_daily_response.data else 0
        
        # New users in last 7 days
        new_users_weekly_response = supabase.table("users").select("id", count="exact").gte("created_at", last_week).execute()
        new_users_weekly = len(new_users_weekly_response.data) if new_users_weekly_response.data else 0
        
        # New users in last 30 days
        new_users_monthly_response = supabase.table("users").select("id", count="exact").gte("created_at", last_month).execute()
        new_users_monthly = len(new_users_monthly_response.data) if new_users_monthly_response.data else 0
        
        # === ACTIVE USERS (users who created stories) ===
        # Get all child profiles with their parent_id and id
        child_profiles_response = supabase.table("child_profiles").select("id, parent_id").execute()
        child_profiles = child_profiles_response.data if child_profiles_response.data else []
        
        # Create a mapping from child_profile_id to parent_id
        child_to_parent = {profile['id']: profile['parent_id'] for profile in child_profiles}
        
        # Get all stories with their child_profile_id
        stories_response = supabase.table("stories").select("child_profile_id").execute()
        stories = stories_response.data if stories_response.data else []
        
        # Find unique parent users who have created stories
        active_user_ids = set()
        for story in stories:
            child_profile_id = story.get('child_profile_id')
            if child_profile_id and child_profile_id in child_to_parent:
                parent_id = child_to_parent[child_profile_id]
                active_user_ids.add(parent_id)
        
        active_users_count = len(active_user_ids)
        
        # === BUILD RESPONSE ===
        statistics = {
            "total_users": total_users,
            "new_users": {
                "daily": new_users_daily,
                "weekly": new_users_weekly,
                "monthly": new_users_monthly
            },
            "active_users": {
                "count": active_users_count,
                "percentage": round((active_users_count / total_users * 100), 2) if total_users > 0 else 0
            },
            "by_role": role_distribution,
            "by_subscription_status": subscription_distribution,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            }
        }
        
        logger.info(f"User statistics generated successfully: {statistics}")
        return statistics
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating user statistics: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating user statistics: {str(e)}")

@app.get("/api/dashboard/user-statistics/summary")
@limiter.limit("30/minute")
async def get_user_statistics_summary(request: Request):
    """
    Get quick summary of user statistics (optimized for dashboard widgets)
    
    Returns:
        Dictionary with quick user statistics summary
    """
    try:
        if not supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        from datetime import datetime, timedelta
        
        # Quick counts using count queries
        users_response = supabase.table("users").select("id", count="exact").execute()
        total_users = len(users_response.data) if users_response.data else 0
        
        # Recent activity (last 24 hours)
        last_24h = (datetime.now() - timedelta(hours=24)).isoformat()
        new_users_24h_response = supabase.table("users").select("id", count="exact").gte("created_at", last_24h).execute()
        new_users_24h = len(new_users_24h_response.data) if new_users_24h_response.data else 0
        
        # Get child profiles and stories for active users count
        child_profiles_response = supabase.table("child_profiles").select("id, parent_id").execute()
        child_profiles = child_profiles_response.data if child_profiles_response.data else []
        child_to_parent = {profile['id']: profile['parent_id'] for profile in child_profiles}
        
        stories_response = supabase.table("stories").select("child_profile_id").execute()
        stories = stories_response.data if stories_response.data else []
        
        active_user_ids = set()
        for story in stories:
            child_profile_id = story.get('child_profile_id')
            if child_profile_id and child_profile_id in child_to_parent:
                active_user_ids.add(child_to_parent[child_profile_id])
        
        return {
            "summary": {
                "total_users": total_users,
                "active_users": len(active_user_ids),
                "new_users_24h": new_users_24h
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating summary statistics: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating summary statistics: {str(e)}")


def verify_purchase(story_id: int, user_id: Optional[str] = None) -> bool:
    """
    Verify if user has purchased the book/story
    
    Args:
        story_id: Story/Book ID
        user_id: User ID (optional, for direct verification)
    
    Returns:
        True if purchase verified, False otherwise
    """
    try:
        if not supabase:
            logger.warning("Supabase not available for purchase verification")
            return False
        
        # Check if purchase exists
        query = supabase.table("book_purchases").select("*").eq("story_id", story_id)
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        response = query.eq("purchase_status", "completed").execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"✅ Purchase verified for story {story_id}, user {user_id}")
            return True
        
        # In production mode, enforce purchase verification
        if IS_PRODUCTION:
            logger.warning(f"❌ No purchase found for story {story_id}, user {user_id} - access denied")
            return False
        
        # Development mode: allow free access
        logger.warning(f"⚠️ No purchase found for story {story_id}, user {user_id} - allowing access (development mode)")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying purchase: {e}")
        # In production, fail closed (deny access on error)
        return not IS_PRODUCTION


# ============================================================================
# STRIPE SUBSCRIPTION ENDPOINTS
# ============================================================================

class CreateSubscriptionRequest(BaseModel):
    """Request model for creating a subscription checkout session"""
    price_type: str = "monthly"  # "monthly" or "yearly"
    user_email: Optional[str] = None
    user_id: Optional[str] = None
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class CreateOnetimeCheckoutRequest(BaseModel):
    """Request model for creating a one-time purchase checkout session"""
    purchase_type: str  # "single_story" or "story_bundle"
    story_id: Optional[str] = None  # Story ID to mark as purchased after payment
    user_email: Optional[str] = None
    user_id: Optional[str] = None
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class SubscriptionResponse(BaseModel):
    """Response model for subscription operations"""
    success: bool
    checkout_url: Optional[str] = None
    session_id: Optional[str] = None
    message: Optional[str] = None

class SubscriptionStatusResponse(BaseModel):
    """Response model for subscription status"""
    success: bool
    is_active: bool = False
    subscription_id: Optional[str] = None
    status: Optional[str] = None
    current_period_end: Optional[str] = None
    plan_type: Optional[str] = None
    message: Optional[str] = None

class CustomerPortalResponse(BaseModel):
    """Response model for customer portal"""
    success: bool
    portal_url: Optional[str] = None
    message: Optional[str] = None

class CancelSubscriptionRequest(BaseModel):
    """Request model for cancelling a subscription"""
    stripe_subscription_id: str

class CancelSubscriptionResponse(BaseModel):
    """Response model for subscription cancellation"""
    success: bool
    message: Optional[str] = None
    access_until: Optional[str] = None


@app.post("/api/stripe/create-onetime-checkout", response_model=SubscriptionResponse)
async def create_onetime_checkout(request: CreateOnetimeCheckoutRequest):
    """
    Create a Stripe checkout session for one-time purchases (single story or story bundle).
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    try:
        # Determine price ID based on purchase type
        if request.purchase_type == "single_story":
            price_id = STRIPE_PRICE_ID_SINGLE_STORY
        elif request.purchase_type == "story_bundle":
            price_id = STRIPE_PRICE_ID_STORY_BUNDLE
        else:
            raise HTTPException(status_code=400, detail=f"Invalid purchase_type: {request.purchase_type}")
        
        if not price_id:
            raise HTTPException(status_code=503, detail=f"Price ID not configured for {request.purchase_type}")
        
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": price_id,
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=request.success_url or f"{FRONTEND_URL}/purchase/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=request.cancel_url or f"{FRONTEND_URL}/pricing",
            customer_email=request.user_email,
            metadata={
                "user_id": request.user_id or "unknown",
                "purchase_type": request.purchase_type,
                "story_id": request.story_id or "none"
            }
        )
        
        logger.info(f"Created one-time checkout session {checkout_session.id} for {request.purchase_type}")
        
        return SubscriptionResponse(
            success=True,
            checkout_url=checkout_session.url,
            session_id=checkout_session.id
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating one-time checkout: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating one-time checkout: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {str(e)}")


@app.post("/api/stripe/create-subscription-checkout", response_model=SubscriptionResponse)
async def create_subscription_checkout(request: CreateSubscriptionRequest):
    """
    Create a Stripe checkout session for subscription plans (monthly or yearly).
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    try:
        # Determine price ID based on subscription type
        if request.price_type == "monthly":
            price_id = STRIPE_PRICE_ID_MONTHLY
        elif request.price_type == "yearly":
            price_id = STRIPE_PRICE_ID_YEARLY
        else:
            raise HTTPException(status_code=400, detail=f"Invalid price_type: {request.price_type}")
        
        if not price_id:
            raise HTTPException(status_code=503, detail=f"Price ID not configured for {request.price_type} subscription")
        
        # Create checkout session for subscription
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": price_id,
                    "quantity": 1,
                }
            ],
            mode="subscription",
            success_url=request.success_url or f"{FRONTEND_URL}/purchase/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=request.cancel_url or f"{FRONTEND_URL}/pricing",
            customer_email=request.user_email,
            metadata={
                "user_id": request.user_id or "unknown",
                "price_type": request.price_type
            }
        )
        
        logger.info(f"Created subscription checkout session {checkout_session.id} for {request.price_type} plan")
        
        return SubscriptionResponse(
            success=True,
            checkout_url=checkout_session.url,
            session_id=checkout_session.id
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating subscription checkout: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating subscription checkout: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {str(e)}")


@app.post("/api/stripe/create-customer-portal", response_model=CustomerPortalResponse)
async def create_customer_portal(user_id: str, return_url: Optional[str] = None):
    """
    Create a Stripe Customer Portal session for managing subscriptions.
    Allows users to update payment method, cancel subscription, etc.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Database is not configured")
    
    try:
        # Get the customer ID from subscriptions table
        response = supabase.table("subscriptions").select("stripe_customer_id").eq("user_id", user_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="No subscription found for this user")
        
        customer_id = response.data[0].get("stripe_customer_id")
        if not customer_id:
            raise HTTPException(status_code=404, detail="Customer ID not found")
        
        # Create the portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url or f"{FRONTEND_URL}/dashboard"
        )
        
        return CustomerPortalResponse(
            success=True,
            portal_url=portal_session.url
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating portal session: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating customer portal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create customer portal: {str(e)}")


@app.post("/api/subscriptions/cancel", response_model=CancelSubscriptionResponse)
@limiter.limit("10/minute")
async def cancel_subscription(request: Request, cancel_request: CancelSubscriptionRequest):
    """
    Cancel a subscription via Stripe API.
    Requires authentication via Bearer token.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Database is not configured")
    
    try:
        # Extract user ID from authorization token
        authorization = request.headers.get("Authorization")
        user_id = extract_user_from_token(authorization)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please provide a valid Bearer token."
            )
        
        stripe_subscription_id = cancel_request.stripe_subscription_id
        
        # Verify the subscription belongs to this user
        subscription_result = supabase.table("subscriptions").select(
            "id, user_id, stripe_customer_id, status, customer_email"
        ).eq("stripe_subscription_id", stripe_subscription_id).execute()
        
        if not subscription_result.data or len(subscription_result.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="Subscription not found"
            )
        
        subscription_data = subscription_result.data[0]
        
        # Verify ownership
        if subscription_data.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to cancel this subscription"
            )
        
        # Check if already cancelled
        if subscription_data.get("status") in ["cancelled", "canceled"]:
            raise HTTPException(
                status_code=400,
                detail="Subscription is already cancelled"
            )
        
        # Cancel the subscription in Stripe (at period end)
        try:
            stripe_subscription = stripe.Subscription.modify(
                stripe_subscription_id,
                cancel_at_period_end=True
            )
            
            logger.info(f"Cancelled subscription {stripe_subscription_id} for user {user_id}")
            
            # Get plan type and period end
            plan_type = "monthly"
            access_until = None
            current_period_end = None
            try:
                current_period_end = stripe_subscription.get("current_period_end")
                if current_period_end:
                    access_until = datetime.fromtimestamp(current_period_end).strftime("%B %d, %Y")
                
                if stripe_subscription.get("items", {}).get("data"):
                    price = stripe_subscription["items"]["data"][0].get("price", {})
                    interval = price.get("recurring", {}).get("interval", "month")
                    plan_type = "yearly" if interval == "year" else "monthly"
            except Exception:
                pass
            
            # Update database
            supabase.table("subscriptions").update({
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("stripe_subscription_id", stripe_subscription_id).execute()
            
            # Update users table
            user_update_data = {
                "subscription_status": "cancelled",
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Keep subscription_expires until period end
            if current_period_end:
                user_update_data["subscription_expires"] = datetime.fromtimestamp(current_period_end).isoformat() + "Z"
            
            supabase.table("users").update(user_update_data).eq("id", user_id).execute()
            logger.info(f"Updated user {user_id} with cancelled subscription status")
            
            # Get customer email and name for email
            customer_email = subscription_data.get("customer_email")
            customer_name = None
            
            if not customer_email:
                user_result = supabase.table("users").select("email, first_name, last_name").eq("id", user_id).execute()
                if user_result.data and len(user_result.data) > 0:
                    user_data = user_result.data[0]
                    customer_email = user_data.get("email")
                    first_name = user_data.get("first_name", "")
                    last_name = user_data.get("last_name", "")
                    customer_name = f"{first_name} {last_name}".strip() or None
            
            # Send subscription cancelled email directly (avoid HTTP connection issues)
            if customer_email:
                try:
                    # Call the email helper function directly instead of making HTTP call
                    result = await email_api.send_subscription_cancelled_email_direct(
                        to_email=customer_email,
                        customer_name=customer_name,
                        plan_type=plan_type,
                        access_until=access_until
                    )
                    if result.get("success"):
                        logger.info(f"✅ Subscription cancelled email sent to {customer_email}")
                    else:
                        error_msg = result.get("error", "Unknown error")
                        logger.error(f"❌ Failed to send subscription cancelled email: {error_msg}")
                except Exception as email_error:
                    logger.error(f"Error sending cancellation email: {email_error}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            return CancelSubscriptionResponse(
                success=True,
                message="Subscription cancelled successfully. You'll retain access until the end of your current billing period.",
                access_until=access_until
            )
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error cancelling subscription: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to cancel subscription: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel subscription: {str(e)}"
        )


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    This endpoint receives events from Stripe about subscription changes.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    # Get the raw body and signature
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        # Verify the webhook signature if secret is configured
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        else:
            # For development, parse without verification
            event = stripe.Event.construct_from(
                json.loads(payload), stripe.api_key
            )
        
        event_type = event["type"]
        event_data = event["data"]["object"]
        
        logger.info(f"Received Stripe webhook: {event_type}")
        
        # Handle different event types
        if event_type == "checkout.session.completed":
            await handle_checkout_completed(event_data)
        elif event_type == "customer.subscription.created":
            await handle_subscription_created(event_data)
        elif event_type == "customer.subscription.updated":
            await handle_subscription_updated(event_data)
        elif event_type == "customer.subscription.deleted":
            await handle_subscription_deleted(event_data)
        elif event_type == "invoice.payment_succeeded":
            await handle_payment_succeeded(event_data)
        elif event_type == "invoice.payment_failed":
            await handle_payment_failed(event_data)
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")
        
        return {"status": "success"}
        
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing error: {str(e)}")


def get_product_metadata_from_subscription(subscription):
    """
    Extract product metadata (credit and amount) from a Stripe subscription.
    Returns dict with 'credit' and 'amount' if found, None otherwise.
    """
    try:
        # Get the first subscription item
        items = subscription.get("items", {}).get("data", [])
        if not items:
            logger.warning("No items found in subscription")
            return None
        
        # Get the price from the first item
        price = items[0].get("price", {})
        if not price:
            logger.warning("No price found in subscription item")
            return None
        
        # Get the product ID from the price
        product_id = price.get("product")
        if not product_id:
            logger.warning("No product ID found in price")
            return None
        
        # Retrieve the product to get metadata
        product = stripe.Product.retrieve(product_id)
        product_metadata = product.get("metadata", {})
        
        # Extract credit and amount from metadata
        credit = product_metadata.get("credit")
        amount = product_metadata.get("amount")
        
        if credit or amount:
            result = {}
            if credit:
                result["credit"] = credit
            if amount:
                result["amount"] = amount
            logger.info(f"Retrieved product metadata: credit={credit}, amount={amount} from product {product_id}")
            return result
        else:
            logger.info(f"No credit/amount metadata found in product {product_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving product metadata from subscription: {e}")
        return None


async def handle_checkout_completed(session):
    """Handle successful checkout session completion"""
    try:
        mode = session.get("mode")
        metadata = session.get("metadata", {})
        logger.info(f"Checkout completed: {session}")
        
        # Handle one-time payment (story purchase)
        if mode == "payment":
            story_id = metadata.get("story_id")
            user_id = metadata.get("user_id")
            purchase_type = metadata.get("purchase_type")
            payment_status = session.get("payment_status")
            
            logger.info(f"Checkout completed for one-time payment: story_id={story_id}, user_id={user_id}")
            
            # Mark story as purchased if story_id is provided and payment is successful
            if story_id and payment_status == "paid" and supabase:
                try:
                    # Update the story's purchased field to true
                    logger.info(f"Updating story {story_id} as purchased")
                    
                    update_result = supabase.table("stories").update({
                        "purchased": True
                    }).eq("uid", story_id).execute()
                    
                    if update_result.data and len(update_result.data) > 0:
                        logger.info(f"Successfully marked story {story_id} as purchased")
                    else:
                        logger.warning(f"No story found with id {story_id} to mark as purchased")
                        
                except Exception as e:
                    logger.error(f"Error marking story {story_id} as purchased: {e}")
            
            return
        
        # Handle subscription payment
        if mode != "subscription":
            return
        
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")
        customer_email = session.get("customer_email") or session.get("customer_details", {}).get("email")
        user_id = metadata.get("user_id")
        price_type = metadata.get("price_type", "monthly")
        
        logger.info(f"Checkout completed for subscription {subscription_id}")
        
        # Get subscription details from Stripe
        subscription = stripe.Subscription.retrieve(subscription_id)

        print('[handle_checkout_completed] subscription:', user_id, customer_id, subscription_id, customer_email, price_type);
        
        # Get product metadata (credit and amount) from subscription
        product_metadata = get_product_metadata_from_subscription(subscription)
        credit = None
        amount = None
        if product_metadata:
            credit = product_metadata.get("credit")
            amount = product_metadata.get("amount")
            logger.info(f"Subscription purchase - Product metadata: credit={credit}, amount={amount}")
        
        # Get subscription expiration from Stripe
        subscription_expires = None
        current_period_end = subscription.get("current_period_end")
        if current_period_end:
            subscription_expires = datetime.fromtimestamp(current_period_end).isoformat() + "Z"
        
        # Save to database
        if supabase:
            # Use actual dates from Stripe subscription
            current_period_start_iso = None
            current_period_end_iso = None
            if subscription.get("current_period_start"):
                current_period_start_iso = datetime.fromtimestamp(subscription.get("current_period_start")).isoformat() + "Z"
            if subscription.get("current_period_end"):
                current_period_end_iso = datetime.fromtimestamp(subscription.get("current_period_end")).isoformat() + "Z"
            
            subscription_data = {
                "user_id": user_id if user_id else None,
                "stripe_customer_id": customer_id,
                "stripe_subscription_id": subscription_id,
                "customer_email": customer_email,
                "status": subscription.status,
                "plan_type": "monthly", # price_type,
                "current_period_start": current_period_start_iso or datetime.utcnow().isoformat() + "Z",
                "current_period_end": current_period_end_iso or datetime.utcnow().replace(month=(datetime.utcnow().month + 1) % 12 if datetime.utcnow().month == 12 else datetime.utcnow().month + 1).isoformat() + "Z",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Add credit and amount to subscription data if available
            if credit is not None:
                subscription_data["credit"] = credit
            if amount is not None:
                subscription_data["amount"] = amount
            
            # Upsert subscription record
            supabase.table("subscriptions").upsert(
                subscription_data,
                on_conflict="stripe_subscription_id"
            ).execute()
            
            logger.info(f"Saved subscription {subscription_id} to database")
            
            # Always update stripe_customer_id in users table
            # Find user by user_id if available, otherwise by email
            target_user_id = None
            if user_id:
                target_user_id = user_id
            elif customer_email:
                try:
                    user_result = supabase.table("users").select("id").eq("email", customer_email).execute()
                    if user_result.data and len(user_result.data) > 0:
                        target_user_id = user_result.data[0].get("id")
                except Exception as e:
                    logger.warning(f"Could not find user by email {customer_email}: {e}")
            
            if target_user_id and customer_id:
                try:
                    # Determine subscription status
                    subscription_status = "premium" if subscription.status in ["active", "trialing"] else subscription.status
                    
                    user_update_data = {
                        "stripe_customer_id": customer_id
                    }
                    
                    # Update subscription_status and subscription_expires if subscription is active or trialing
                    if subscription.status in ["active", "trialing"]:
                        user_update_data["subscription_status"] = subscription_status
                        if subscription_expires:
                            user_update_data["subscription_expires"] = subscription_expires
                    
                    # Update credit column in users table if available from product metadata
                    if credit is not None:
                        user_update_data["credit"] = credit
                    
                    supabase.table("users").update(user_update_data).eq("id", target_user_id).execute()
                    logger.info(f"Updated user {target_user_id} with stripe_customer_id={customer_id}, subscription_status={subscription_status}, and credit={credit} from checkout completed")
                except Exception as e:
                    logger.error(f"Error updating user stripe_customer_id in checkout completed: {e}")
            elif customer_id:
                logger.warning(f"Could not update stripe_customer_id: user_id={user_id}, customer_email={customer_email}, customer_id={customer_id}")
            
    except Exception as e:
        logger.error(f"Error handling checkout completed: {e}")


async def handle_subscription_created(subscription):
    """Handle subscription created event"""
    try:
        subscription_id = subscription.get("id")
        customer_id = subscription.get("customer")
        status = subscription.get("status")
        
        logger.info(f"Subscription created: {subscription_id} with status {status}")
        
        # Get product metadata (credit and amount) from subscription
        product_metadata = get_product_metadata_from_subscription(subscription)
        credit = None
        amount = None
        if product_metadata:
            credit = product_metadata.get("credit")
            amount = product_metadata.get("amount")
            logger.info(f"Subscription created - Product metadata: credit={credit}, amount={amount}")
        
        if supabase:
            # Check if subscription already exists
            existing = supabase.table("subscriptions").select("id").eq("stripe_subscription_id", subscription_id).execute()
            
            if not existing.data or len(existing.data) == 0:
                # Create new subscription record
                subscription_data = {
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": subscription_id,
                    "status": status,
                    "current_period_start": datetime.utcnow().isoformat() + "Z",
                    "current_period_end": datetime.utcnow().replace(month=(datetime.utcnow().month + 1) % 12 if datetime.utcnow().month == 12 else datetime.utcnow().month + 1).isoformat() + "Z",
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # Add credit and amount to subscription data if available
                if credit is not None:
                    subscription_data["credit"] = credit
                if amount is not None:
                    subscription_data["amount"] = amount
                
                supabase.table("subscriptions").insert(subscription_data).execute()
            
            # Update users table - find user by stripe_customer_id
            user_result = supabase.table("users").select("id").eq("stripe_customer_id", customer_id).execute()
            
            if user_result.data and len(user_result.data) > 0:
                user_id = user_result.data[0].get("id")
                subscription_expires = datetime.utcnow().replace(month=(datetime.utcnow().month + 1) % 12 if datetime.utcnow().month == 12 else datetime.utcnow().month + 1).isoformat() + "Z"
                
                # Set subscription_status to "premium" if subscription is active or trialing
                user_subscription_status = "premium" if status in ["active", "trialing"] else status
                
                user_update_data = {
                    "subscription_status": user_subscription_status,
                    "stripe_customer_id": customer_id,
                    "subscription_expires": subscription_expires
                }
                
                # Update credit column in users table if available from product metadata
                if credit is not None:
                    user_update_data["credit"] = credit
                
                supabase.table("users").update(user_update_data).eq("id", user_id).execute()
                logger.info(f"Updated user {user_id} with subscription info from subscription created event (status: {user_subscription_status}, credit: {credit})")
                
    except Exception as e:
        logger.error(f"Error handling subscription created: {e}")


async def handle_subscription_updated(subscription):
    """Handle subscription updated event"""
    try:
        subscription_id = subscription.get("id")
        customer_id = subscription.get("customer")
        status = subscription.get("status")
        current_period_end = subscription.get("current_period_end")
        current_period_start = subscription.get("current_period_start")
        
        logger.info(f"Subscription updated: {subscription_id} to status {status}")
        
        # Normalize "canceled" to "cancelled" for consistency
        normalized_status = "cancelled" if status in ["canceled", "cancelled"] else status
        
        if supabase:
            update_data = {
                "status": normalized_status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Set cancelled_at if status is cancelled (preserve existing cancelled_at if already set)
            if normalized_status == "cancelled":
                existing_sub = supabase.table("subscriptions").select("cancelled_at").eq("stripe_subscription_id", subscription_id).execute()
                if not existing_sub.data or not existing_sub.data[0].get("cancelled_at"):
                    update_data["cancelled_at"] = datetime.utcnow().isoformat()
            
            # Use actual period dates from Stripe subscription if available
            if current_period_start:
                update_data["current_period_start"] = datetime.fromtimestamp(current_period_start).isoformat() + "Z"
            if current_period_end:
                update_data["current_period_end"] = datetime.fromtimestamp(current_period_end).isoformat() + "Z"
            
            supabase.table("subscriptions").update(update_data).eq("stripe_subscription_id", subscription_id).execute()
            
            # Update users table - find user by stripe_customer_id
            user_result = supabase.table("users").select("id").eq("stripe_customer_id", customer_id).execute()
            
            if user_result.data and len(user_result.data) > 0:
                user_id = user_result.data[0].get("id")
                
                # Set subscription_status to "premium" if subscription is active or trialing
                if normalized_status in ["active", "trialing"]:
                    user_subscription_status = "premium"
                else:
                    user_subscription_status = normalized_status
                
                user_update_data = {
                    "subscription_status": user_subscription_status
                }
                
                # Keep subscription_expires until period end (even when cancelled)
                if current_period_end:
                    user_update_data["subscription_expires"] = datetime.fromtimestamp(current_period_end).isoformat() + "Z"
                
                supabase.table("users").update(user_update_data).eq("id", user_id).execute()
                logger.info(f"Updated user {user_id} with subscription info from subscription updated event (status: {user_subscription_status})")
            
    except Exception as e:
        logger.error(f"Error handling subscription updated: {e}")


async def handle_subscription_deleted(subscription):
    """Handle subscription cancelled/deleted event"""
    try:
        subscription_id = subscription.get("id")
        customer_id = subscription.get("customer")
        
        logger.info(f"Subscription deleted: {subscription_id}")
        
        # Get plan type and period end
        plan_type = "monthly"
        access_until = None
        try:
            current_period_end = subscription.get("current_period_end")
            if current_period_end:
                access_until = datetime.fromtimestamp(current_period_end).strftime("%B %d, %Y")
            
            if subscription.get("items", {}).get("data"):
                price = subscription["items"]["data"][0].get("price", {})
                interval = price.get("recurring", {}).get("interval", "month")
                plan_type = "yearly" if interval == "year" else "monthly"
        except Exception:
            pass
        
        customer_email = None
        customer_name = None
        
        if supabase:
            # Get email from subscription record
            try:
                sub_result = supabase.table("subscriptions").select("customer_email").eq("stripe_subscription_id", subscription_id).execute()
                if sub_result.data:
                    customer_email = sub_result.data[0].get("customer_email")
            except Exception:
                pass
            
            supabase.table("subscriptions").update({
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("stripe_subscription_id", subscription_id).execute()
            
            # Update users table - find user by stripe_customer_id
            user_result = supabase.table("users").select("id, email").eq("stripe_customer_id", customer_id).execute()
            
            if user_result.data and len(user_result.data) > 0:
                user_data = user_result.data[0]
                user_id = user_data.get("id")
                
                # Get email from user if not found in subscription
                if not customer_email:
                    customer_email = user_data.get("email")
                
                user_update_data = {
                    "subscription_status": "cancelled",
                    "subscription_expires": None
                }
                
                supabase.table("users").update(user_update_data).eq("id", user_id).execute()
                logger.info(f"Updated user {user_id} with cancelled subscription status")
        
        # Send subscription cancelled email directly (avoid HTTP connection issues)
        if customer_email:
            try:
                # Call the email helper function directly instead of making HTTP call
                result = await email_api.send_subscription_cancelled_email_direct(
                    to_email=customer_email,
                    customer_name=customer_name,
                    plan_type=plan_type,
                    access_until=access_until
                )
                if result.get("success"):
                    logger.info(f"✅ Subscription cancelled email sent to {customer_email}")
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"❌ Failed to send subscription cancelled email: {error_msg}")
            except Exception as email_error:
                logger.error(f"Error sending cancellation email: {email_error}")
                import traceback
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Error handling subscription deleted: {e}")


async def handle_payment_succeeded(invoice):
    """Handle successful payment"""
    try:
        # Try to get subscription ID from multiple locations
        subscription_id = invoice.get("subscription")
        
        # If not at top level, try nested in parent.subscription_details
        if not subscription_id:
            subscription_id = invoice.get("parent", {}).get("subscription_details", {}).get("subscription")
        
        # If still not found, try from line items
        if not subscription_id:
            lines_data = invoice.get("lines", {}).get("data", [])
            if lines_data:
                subscription_id = lines_data[0].get("parent", {}).get("subscription_item_details", {}).get("subscription")
        
        customer_id = invoice.get("customer")
        customer_email = invoice.get("customer_email")
        customer_name = invoice.get("customer_name")
        amount_paid = invoice.get("amount_paid", 0)
        
        if subscription_id:
            logger.info(f"Payment succeeded for subscription: {subscription_id}")
            
            # Get subscription details from Stripe
            plan_type = "monthly"
            next_billing_date = None
            subscription_status = None
            subscription_expires = None
            try:
                stripe_subscription = stripe.Subscription.retrieve(subscription_id)
                
                # Get subscription status from Stripe
                subscription_status = stripe_subscription.get("status")
                
                # Get subscription expiration from current_period_end
                current_period_end = stripe_subscription.get("current_period_end")
                if current_period_end:
                    subscription_expires = datetime.fromtimestamp(current_period_end).isoformat() + "Z"
                    next_billing_date = datetime.fromtimestamp(current_period_end).strftime("%B %d, %Y")
                
                # Determine plan type from price interval
                if stripe_subscription.get("items", {}).get("data"):
                    price = stripe_subscription["items"]["data"][0].get("price", {})
                    interval = price.get("recurring", {}).get("interval", "month")
                    plan_type = "yearly" if interval == "year" else "monthly"
            except Exception as e:
                logger.warning(f"Could not retrieve subscription details: {e}")
                subscription_expires = None
                subscription_status = None
            
            if supabase:
                # Get customer email from subscription if not in invoice
                if not customer_email:
                    try:
                        sub_result = supabase.table("subscriptions").select("customer_email").eq("stripe_subscription_id", subscription_id).execute()
                        if sub_result.data:
                            customer_email = sub_result.data[0].get("customer_email")
                    except Exception:
                        pass
                
                # Update subscription status in subscriptions table
                subscription_update_data = {
                    "last_payment_date": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                if subscription_status:
                    subscription_update_data["status"] = subscription_status
                
                supabase.table("subscriptions").update(subscription_update_data).eq("stripe_subscription_id", subscription_id).execute()
                
                # Update users table with subscription status and expiration from Stripe response
                if customer_id:
                    user_result = supabase.table("users").select("id, email").eq("stripe_customer_id", customer_id).execute()
                    
                    if user_result.data and len(user_result.data) > 0:
                        user_data = user_result.data[0]
                        user_id = user_data.get("id")
                        
                        # Use user email if invoice email not available
                        if not customer_email:
                            customer_email = user_data.get("email")
                        
                        # Update user table with subscription status and expiration from Stripe
                        user_update_data = {}
                        
                        # Update stripe_customer_id if not already set
                        user_update_data["stripe_customer_id"] = customer_id
                        
                        # Update subscription_status based on Stripe subscription status
                        if subscription_status:
                            # Map Stripe status to our status
                            if subscription_status in ["active", "trialing"]:
                                user_update_data["subscription_status"] = "premium"
                            elif subscription_status in ["past_due", "unpaid", "canceled", "incomplete", "incomplete_expired"]:
                                user_update_data["subscription_status"] = subscription_status
                            else:
                                user_update_data["subscription_status"] = subscription_status
                        
                        # Update subscription_expires from Stripe response
                        if subscription_expires:
                            user_update_data["subscription_expires"] = subscription_expires
                        elif subscription_status in ["canceled", "incomplete_expired"]:
                            # Clear expiration if subscription is canceled
                            user_update_data["subscription_expires"] = None
                        
                        if user_update_data:
                            supabase.table("users").update(user_update_data).eq("id", user_id).execute()
                            logger.info(f"Updated user {user_id} with subscription_status={user_update_data.get('subscription_status')} and subscription_expires={user_update_data.get('subscription_expires')} from payment succeeded")
                    else:
                        # Try to find user by email if not found by stripe_customer_id
                        if customer_email:
                            try:
                                user_result = supabase.table("users").select("id, email").eq("email", customer_email).execute()
                                if user_result.data and len(user_result.data) > 0:
                                    user_data = user_result.data[0]
                                    user_id = user_data.get("id")
                                    
                                    user_update_data = {
                                        "stripe_customer_id": customer_id
                                    }
                                    
                                    if subscription_status:
                                        if subscription_status in ["active", "trialing"]:
                                            user_update_data["subscription_status"] = "premium"
                                        else:
                                            user_update_data["subscription_status"] = subscription_status
                                    
                                    if subscription_expires:
                                        user_update_data["subscription_expires"] = subscription_expires
                                    
                                    supabase.table("users").update(user_update_data).eq("id", user_id).execute()
                                    logger.info(f"Updated user {user_id} (found by email) with stripe_customer_id and subscription info from payment succeeded")
                            except Exception as e:
                                logger.warning(f"Could not find user by email to update stripe_customer_id: {e}")
            
            # Note: Payment confirmation emails are now sent from the frontend after successful payment
            # The frontend will call /api/stripe/session/{session_id} to get payment details
            # and then send emails via /api/emails/payment-success and /api/emails/receipt
            logger.info(f"Payment succeeded - Email: {customer_email}, Amount: ${amount_paid / 100:.2f if amount_paid else 0:.2f}")
                
    except Exception as e:
        logger.error(f"Error handling payment succeeded: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def handle_payment_failed(invoice):
    """Handle failed payment"""
    try:
        logger.info(f"Processing payment failed event")
        
        # Try to get subscription ID from multiple locations
        subscription_id = invoice.get("subscription")
        
        # If not at top level, try nested in parent.subscription_details
        if not subscription_id:
            subscription_id = invoice.get("parent", {}).get("subscription_details", {}).get("subscription")
        
        # If still not found, try from line items
        if not subscription_id:
            lines_data = invoice.get("lines", {}).get("data", [])
            if lines_data:
                subscription_id = lines_data[0].get("parent", {}).get("subscription_item_details", {}).get("subscription")
        
        customer_id = invoice.get("customer")
        customer_email = invoice.get("customer_email")
        customer_name = invoice.get("customer_name")
        amount_due = invoice.get("amount_due", 0)
        
        if subscription_id:
            logger.info(f"Payment failed for subscription: {subscription_id}")
            
            # Get plan type
            plan_type = "monthly"
            try:
                stripe_subscription = stripe.Subscription.retrieve(subscription_id)
                if stripe_subscription.get("items", {}).get("data"):
                    price = stripe_subscription["items"]["data"][0].get("price", {})
                    interval = price.get("recurring", {}).get("interval", "month")
                    plan_type = "yearly" if interval == "year" else "monthly"
            except Exception:
                pass
            
            if supabase:
                # Get customer email from subscription or user if not in invoice
                if not customer_email:
                    try:
                        sub_result = supabase.table("subscriptions").select("customer_email").eq("stripe_subscription_id", subscription_id).execute()
                        if sub_result.data:
                            customer_email = sub_result.data[0].get("customer_email")
                    except Exception:
                        pass
                
                if not customer_email and customer_id:
                    try:
                        user_result = supabase.table("users").select("email").eq("stripe_customer_id", customer_id).execute()
                        if user_result.data:
                            customer_email = user_result.data[0].get("email")
                    except Exception:
                        pass
                
                supabase.table("subscriptions").update({
                    "status": "past_due",
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("stripe_subscription_id", subscription_id).execute()
            
            # Send payment failed email via API
            logger.info(f"Attempting to send payment failed email - Email: {customer_email}, Service enabled: {bool(os.getenv('RESEND_API_KEY'))}")
            
            if not customer_email:
                logger.warning("Cannot send payment failed email: customer_email is missing")
            elif not os.getenv("RESEND_API_KEY"):
                logger.warning("Cannot send payment failed email: email service not enabled")
            else:
                try:
                    amount_display = f"${amount_due / 100:.2f}" if amount_due else None
                    result = await call_email_api("/emails/payment-failed", {
                        "to_email": customer_email,
                        "customer_name": customer_name,
                        "plan_type": plan_type,
                        "amount": amount_display,
                        "retry_url": f"{FRONTEND_URL}/account"
                    })
                    if result.get("success"):
                        logger.info(f"✅ Payment failed email sent to {customer_email}")
                    else:
                        logger.error(f"❌ Failed to send payment failed email: {result.get('error')}")
                except Exception as email_error:
                    logger.error(f"❌ Exception sending payment failed email: {email_error}")
                
    except Exception as e:
        logger.error(f"Error handling payment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


@app.get("/api/stripe/config")
async def get_stripe_config():
    """
    Get Stripe publishable key for frontend.
    """
    if not STRIPE_PUBLISHABLE_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    return {
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
        "monthly_price_id": STRIPE_PRICE_ID_MONTHLY,
        "yearly_price_id": STRIPE_PRICE_ID_YEARLY
    }


@app.get("/api/stripe/session/{session_id}")
async def get_checkout_session(session_id: str):
    """
    Get Stripe checkout session details for frontend to retrieve payment information.
    This allows the frontend to get payment details and send confirmation emails.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    try:
        # Retrieve the checkout session from Stripe
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Extract relevant information
        mode = session.get("mode")  # "payment" or "subscription"
        customer_email = session.get("customer_email") or session.get("customer_details", {}).get("email")
        customer_name = session.get("customer_details", {}).get("name")
        payment_status = session.get("payment_status")
        
        # Get amount and currency
        amount_total = session.get("amount_total", 0)
        currency = session.get("currency", "usd")
        amount_display = f"${amount_total / 100:.2f}" if amount_total else None
        
        # For subscriptions, get plan details
        plan_type = None
        next_billing_date = None
        subscription_id = session.get("subscription")
        
        if mode == "subscription" and subscription_id:
            try:
                subscription = stripe.Subscription.retrieve(subscription_id)
                # Get plan type from price interval
                if subscription.get("items", {}).get("data"):
                    price = subscription["items"]["data"][0].get("price", {})
                    interval = price.get("recurring", {}).get("interval", "month")
                    plan_type = "yearly" if interval == "year" else "monthly"
                
                # Get next billing date
                current_period_end = subscription.get("current_period_end")
                if current_period_end:
                    next_billing_date = datetime.fromtimestamp(current_period_end).strftime("%B %d, %Y")
            except Exception as e:
                logger.warning(f"Could not retrieve subscription details: {e}")
        
        # For one-time payments, determine purchase type from metadata
        purchase_type = None
        if mode == "payment":
            metadata = session.get("metadata", {})
            purchase_type = metadata.get("purchase_type", "single_story")
        
        # Get invoice ID if available
        invoice_id = session.get("invoice")
        
        return {
            "success": True,
            "mode": mode,
            "customer_email": customer_email,
            "customer_name": customer_name,
            "payment_status": payment_status,
            "amount": amount_display,
            "currency": currency,
            "plan_type": plan_type,
            "next_billing_date": next_billing_date,
            "purchase_type": purchase_type,
            "subscription_id": subscription_id,
            "invoice_id": invoice_id,
            "session_id": session_id
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error retrieving session: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving checkout session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")


# ==================== USER AUTH SYNC ====================

class AuthSyncRequest(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None


@app.post("/api/gift/deliver")
async def deliver_gift_endpoint(request: Request):
    """
    Deliver a scheduled gift via web push notification.
    This endpoint is called by the edge function cron job.
    No rate limiting applied as this is an internal service endpoint.
    """
    try:
        body = await request.json()
        gift_id = body.get("gift_id")
        
        if not gift_id:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: gift_id"
            )
        
        if not supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        logger.info(f"🎁 Processing gift delivery for gift_id: {gift_id}")
        
        # Get the gift details from database
        gift_response = supabase.table("gifts").select("*").eq("id", gift_id).single().execute()
        
        if not gift_response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Gift not found: {gift_id}"
            )
        
        gift = gift_response.data
        
        # Validate gift can be delivered
        if gift.get("notification_sent") == True:
            logger.warning(f"Gift {gift_id} already delivered")
            return {
                "success": True,
                "message": "Gift already delivered",
                "already_sent": True
            }
        
        if gift.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Gift status is '{gift.get('status')}', must be 'completed' to deliver"
            )
        
        if not gift.get("to_user_id"):
            raise HTTPException(
                status_code=400,
                detail="Gift has no recipient user ID (to_user_id)"
            )
        
        # Call the send-gift-notification edge function to send web push
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        
        edge_function_url = f"{supabase_url}/functions/v1/send-gift-notification"
        
        logger.info(f"📤 Calling edge function to send push notification for gift {gift_id}")
        
        edge_response = requests.post(
            edge_function_url,
            json={
                "giftId": gift_id,
                "mode": "single"
            },
            headers={
                "Authorization": f"Bearer {supabase_anon_key}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if edge_response.status_code != 200:
            logger.error(f"Edge function call failed: {edge_response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send push notification: {edge_response.text}"
            )
        
        edge_result = edge_response.json()
        
        if not edge_result.get("success"):
            logger.error(f"Edge function returned error: {edge_result}")
            raise HTTPException(
                status_code=500,
                detail=f"Push notification failed: {edge_result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"✅ Gift {gift_id} delivered successfully via web push notification")
        
        # Also send delivery email if enabled
        if os.getenv("RESEND_API_KEY"):
            try:
                # Get sender information
                sender_id = gift.get("from_user_id")
                sender_name = "Someone special"
                
                if sender_id:
                    try:
                        # Try to get sender's name from Supabase auth
                        auth_response = supabase.auth.admin.get_user_by_id(sender_id)
                        if auth_response and auth_response.user:
                            sender_name = (
                                auth_response.user.user_metadata.get("name") or
                                auth_response.user.user_metadata.get("full_name") or
                                auth_response.user.email.split('@')[0] if auth_response.user.email else sender_name
                            )
                    except Exception as e:
                        logger.warning(f"Could not fetch sender name: {e}")
                
                # Get recipient email
                recipient_email = gift.get("delivery_email")
                
                if recipient_email:
                    # Note: This is a gift notification (story is being created)
                    # Full gift delivery email with story details is sent from batch_processor when story is completed
                    result = await call_email_api("/emails/gift-notification", {
                        "recipient_email": recipient_email,
                        "recipient_name": gift.get("child_name", "there"),
                        "giver_name": sender_name,
                        "occasion": gift.get("occasion", "special occasion"),
                        "gift_message": gift.get("special_msg", "Enjoy your special story!")
                    })
                    if result.get("success"):
                        logger.info(f"✅ Gift notification email sent to {recipient_email}")
                    else:
                        logger.warning(f"Failed to send gift notification email: {result.get('error')}")
            except Exception as email_error:
                logger.warning(f"Failed to send delivery email (not critical): {email_error}")
        
        return {
            "success": True,
            "message": "Gift delivered successfully",
            "gift_id": gift_id,
            "push_notification_sent": True,
            "results": edge_result.get("results", [])
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error delivering gift: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/sync")
@limiter.limit("10/minute")
async def sync_user_after_auth(request: Request, body: AuthSyncRequest):
    """
    Sync user data after successful OTP/Magic Link verification.
    Sends welcome email only for new users (first registration).
    
    This endpoint should be called by the frontend after successful
    Supabase authentication (OTP code entered correctly or magic link clicked).
    """
    try:
        if not supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )
        
        user_id = body.user_id
        email = body.email
        name = body.name
        
        logger.info(f"Auth sync requested for user: {user_id} ({email})")
        
        # Check if user exists in our users table
        user_response = supabase.table("users").select("id").eq("id", user_id).execute()
        
        is_new_user = not user_response.data or len(user_response.data) == 0
        welcome_email_sent = False
        
        if is_new_user:
            # New user - send welcome email
            logger.info(f"New user detected: {user_id}, sending welcome email")
            
            # Get user's name for the email
            customer_name = name if name else None
            
            if os.getenv("RESEND_API_KEY"):
                try:
                    result = await call_email_api("/emails/welcome", {
                        "to_email": email,
                        "customer_name": customer_name
                    })
                    # Check if email was sent successfully
                    if result.get("success", False):
                        logger.info(f"✅ Welcome email sent to {email} (ID: {result.get('email_id', 'N/A')})")
                        welcome_email_sent = True
                    else:
                        error_msg = result.get("error", "Unknown error")
                        logger.error(f"❌ Failed to send welcome email to {email}: {error_msg}")
                except Exception as email_error:
                    logger.error(f"❌ Exception sending welcome email: {email_error}")
            else:
                logger.warning("Email service not enabled, skipping welcome email")
        else:
            logger.info(f"Existing user {user_id}, skipping welcome email")
        
        return {
            "success": True,
            "is_new_user": is_new_user,
            "welcome_email_sent": welcome_email_sent,
            "message": "User synced successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in auth sync: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error syncing user: {str(e)}")


app.include_router(email_api.router, prefix="/api")

if __name__ == "__main__":
    print("🚀 Starting AI Image Editor Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⚡ Server running on: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )
