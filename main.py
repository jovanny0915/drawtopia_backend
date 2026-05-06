from fastapi import FastAPI, HTTPException, Request, Depends, File, UploadFile, Query, WebSocket, WebSocketDisconnect
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
from PIL import Image as PILImage, ImageDraw, ImageFont
from google import genai
from google.genai import types
from google.genai.types import Image as GeminiImage
from apis import email_api
from story_lib import generate_story
from typing import List, Optional, Dict, Any, Tuple
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

# Twilio Verify (passwordless SMS)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID", "")

# Stripe Configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID_MONTHLY = os.getenv("STRIPE_PRICE_ID_MONTHLY", "")
STRIPE_PRICE_ID_YEARLY = os.getenv("STRIPE_PRICE_ID_YEARLY", "")
STRIPE_PRICE_ID_SINGLE_STORY = os.getenv("STRIPE_PRICE_ID_SINGLE_STORY", "")
STRIPE_PRICE_ID_STORY_BUNDLE = os.getenv("STRIPE_PRICE_ID_STORY_BUNDLE", "")
STRIPE_PRICE_ID_GIFT = os.getenv("STRIPE_PRICE_ID_GIFT", "")
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

# Twilio Verify client (passwordless SMS)
twilio_verify_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_VERIFY_SERVICE_SID:
    try:
        from twilio.rest import Client as TwilioClient
        twilio_verify_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("✅ Twilio Verify client initialized")
    except Exception as e:
        logger.warning(f"⚠️ Twilio Verify not available: {e}")
else:
    logger.warning("⚠️ Twilio Verify not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SERVICE_SID)")

# Google Cloud Vision API (uses existing Google Cloud credentials via GOOGLE_APPLICATION_CREDENTIALS)
from services.vision_character_features import get_vision_client, extract_character_features, VisionNotConfiguredError, VisionAPIError
vision_client = get_vision_client()

# Initialize queue manager and batch processor
queue_manager = None
batch_processor = None
worker_task = None


class StoryProgressConnectionManager:
    """Maps session_id to WebSocket for story generation progress updates (percentage only)."""
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}

    def register(self, websocket: WebSocket) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        self._connections[session_id] = websocket
        return session_id

    def unregister(self, session_id: str) -> None:
        self._connections.pop(session_id, None)

    async def send_progress(self, session_id: str, percentage: int) -> bool:
        ws = self._connections.get(session_id)
        if not ws:
            return False
        try:
            await ws.send_json({"percentage": min(100, max(0, percentage))})
            return True
        except Exception:
            return False


story_progress_manager = StoryProgressConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for background tasks"""
    global queue_manager, batch_processor, worker_task
    
    # Queue manager disabled - uncomment to re-enable
    # if supabase:
    #     queue_manager = QueueManager(supabase)
    #     
    #     # Initialize batch processor (with vision_client for mid-generation validation)
    #     batch_processor = BatchProcessor(
    #         queue_manager=queue_manager,
    #         gemini_client=gemini_client,
    #         openai_api_key=OPENAI_API_KEY,
    #         supabase_client=supabase,
    #         gemini_text_model=GEMINI_TEXT_MODEL,
    #         vision_client=vision_client,
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
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
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
from apis.monitoring import router as monitoring_router
from apis.admin import router as admin_router

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
app.include_router(monitoring_router)
app.include_router(admin_router, prefix="/api")


@app.websocket("/ws/story-progress")
async def websocket_story_progress(websocket: WebSocket):
    """WebSocket endpoint for story generation progress (percentage only). Client receives { percentage: 0-100 }."""
    await websocket.accept()
    session_id = story_progress_manager.register(websocket)
    try:
        await websocket.send_json({"session_id": session_id})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        story_progress_manager.unregister(session_id)


# Request model to receive input data
class ImageRequest(BaseModel):
    image_url: HttpUrl  # This validates the URL format
    prompt: str
    
    class Config:
        # Example values for API documentation
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "prompt": "<frontend-supplied image prompt>"
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
                "story_text_prompt": "<frontend-supplied story prompt>",
                "scene_prompts": ["<frontend-supplied scene prompt 1>", "<frontend-supplied scene prompt 2>", ...],
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


def _get_font_for_size(font_size: int):
    """Load a scalable TrueType font at the given size."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in font_paths:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    # Pillow commonly ships DejaVu; also try common logical names.
    for bundled_name in (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Bold.ttf",
        "NotoSans-Regular.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(bundled_name, font_size)
        except Exception:
            continue
    logger.warning("No scalable TrueType font found; using tiny Pillow default font fallback.")
    return ImageFont.load_default()


def overlay_cover_title_with_reference_style(
    image_data: bytes,
    title_text: str,
    y_position: float = 0.14,
) -> bytes:
    """Draw plain title text on the cover image."""
    image = PILImage.open(BytesIO(image_data)).convert("RGB")
    width, height = image.size
    title = (title_text or "").strip()
    if not title:
        return image_data

    draw = ImageDraw.Draw(image)
    lines = [ln.strip() for ln in title.split("\n") if ln.strip()]
    if not lines:
        lines = [title]

    # Keep font selection simple and scale with image width.
    font_size = 500
    font = 500
    line_boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_heights = [max(1, b[3] - b[1]) for b in line_boxes]
    line_spacing = max(4, font_size // 8)
    total_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1)

    title_center_y = int(height * y_position)
    start_y = max(10, title_center_y - total_height // 2)
    current_y = start_y
    for i, line in enumerate(lines):
        lw = line_boxes[i][2] - line_boxes[i][0]
        x = (width - lw) // 2
        draw.text((x, current_y), line, font=font, fill=(0, 0, 0))
        current_y += line_heights[i] + line_spacing

    out = BytesIO()
    image.save(out, format="JPEG", quality=90, optimize=True)
    return out.getvalue()


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB to (r, g, b). Default to black if invalid."""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 6:
        try:
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return (0, 0, 0)


def overlay_text_on_image(
    image_data: bytes,
    text_blocks: List[Dict[str, Any]],
    logo_url: Optional[str] = None,
) -> bytes:
    """
    Draw decorated text blocks onto an image. Each block can have:
    text, font_size, color_hex, y_position (0-1), alignment (center|left|right), shadow (bool or dict).
    If logo_url is provided, paste the logo at bottom-left before drawing text.
    """
    image = PILImage.open(BytesIO(image_data)).convert("RGB")
    width, height = image.size
    padding = max(width, height) // 25  # ~4% padding

    # Paste logo at bottom-left first (text will be drawn on top)
    if logo_url:
        try:
            logo_data = download_image_from_url(logo_url)
            logo_im = PILImage.open(BytesIO(logo_data))
            if logo_im.mode in ("RGBA", "LA", "P"):
                background = PILImage.new("RGB", logo_im.size, (255, 255, 255))
                if logo_im.mode == "P":
                    logo_im = logo_im.convert("RGBA")
                if logo_im.mode in ("RGBA", "LA"):
                    background.paste(logo_im, mask=logo_im.split()[-1])
                    logo_im = background
            elif logo_im.mode != "RGB":
                logo_im = logo_im.convert("RGB")
            max_logo_w = int(width * 0.28)
            max_logo_h = int(height * 0.12)
            logo_im.thumbnail((max_logo_w, max_logo_h), PILImage.Resampling.LANCZOS)
            lw, lh = logo_im.size
            y_logo = height - padding - lh
            image.paste(logo_im, (padding, y_logo))
        except Exception as e:
            logger.warning(f"Could not paste logo on text-overlay image: {e}")

    draw = ImageDraw.Draw(image)

    for block in text_blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        font_size = max(12, min(200, int(block.get("font_size", 48))))
        color_hex = block.get("color_hex", "#1a1a1a")
        color = _hex_to_rgb(color_hex)
        y_position = float(block.get("y_position", 0.5))
        alignment = (block.get("alignment") or "center").lower()
        if alignment not in ("center", "left", "right"):
            alignment = "center"
        use_shadow = block.get("shadow", True)
        shadow_color = _hex_to_rgb(block.get("shadow_color", "#000000"))
        shadow_offset = block.get("shadow_offset", 2)

        font = _get_font_for_size(font_size)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        # Measure all lines to compute bounding box and line height
        line_heights = []
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            line_heights.append((lw, lh))
            max_line_width = max(max_line_width, lw)
        total_height = sum(h for _, h in line_heights) + (len(line_heights) - 1) * (font_size // 4)

        # Vertical position: y_position 0 = top, 1 = bottom
        block_top = int(height * y_position - total_height / 2)
        block_top = max(padding, min(block_top, height - total_height - padding))

        for i, line in enumerate(lines):
            lw, lh = line_heights[i]
            if alignment == "center":
                x = (width - lw) // 2
            elif alignment == "right":
                x = width - lw - padding
            else:
                x = padding
            y = block_top + sum(line_heights[j][1] for j in range(i)) + i * (font_size // 4)

            if use_shadow:
                draw.text((x + shadow_offset, y + shadow_offset), line, font=font, fill=shadow_color)
            draw.text((x, y), line, font=font, fill=color)

    out = BytesIO()
    image.save(out, format="JPEG", quality=90, optimize=True)
    return out.getvalue()


def _generate_isbn13_barcode_image(isbn13: str) -> Optional[bytes]:
    """Generate an ISBN-13 barcode as PNG bytes. isbn13 can be 12 or 13 digits."""
    try:
        from barcode import ISBN13
        from barcode.writer import ImageWriter
        digits = re.sub(r"\D", "", isbn13)[:13]
        if len(digits) < 12:
            digits = (digits + "000000000000")[:12]
        if len(digits) == 13:
            digits = digits[:12]
        buf = BytesIO()
        ISBN13(digits, writer=ImageWriter()).write(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning(f"Could not generate barcode: {e}")
        return None


def overlay_back_cover(
    image_data: bytes,
    text_blocks: List[Dict[str, Any]],
    logo_url: Optional[str] = None,
    barcode_isbn: Optional[str] = None,
) -> bytes:
    """
    Composite back cover: paste logo (bottom-left) and barcode (bottom-right) first,
    then draw text blocks on top so title, description, tagline, website are visible.
    """
    image = PILImage.open(BytesIO(image_data)).convert("RGB")
    width, height = image.size
    padding = max(width, height) // 20  # 5% padding

    # Paste logo at bottom-left first (text will be drawn on top)
    if logo_url:
        try:
            logo_data = download_image_from_url(logo_url)
            logo_im = PILImage.open(BytesIO(logo_data))
            if logo_im.mode in ("RGBA", "LA", "P"):
                background = PILImage.new("RGB", logo_im.size, (255, 255, 255))
                if logo_im.mode == "P":
                    logo_im = logo_im.convert("RGBA")
                if logo_im.mode in ("RGBA", "LA"):
                    background.paste(logo_im, mask=logo_im.split()[-1])
                    logo_im = background
            elif logo_im.mode != "RGB":
                logo_im = logo_im.convert("RGB")
            max_logo_w = int(width * 0.28)
            max_logo_h = int(height * 0.12)
            logo_im.thumbnail((max_logo_w, max_logo_h), PILImage.Resampling.LANCZOS)
            lw, lh = logo_im.size
            y_logo = height - padding - lh
            image.paste(logo_im, (padding, y_logo))
        except Exception as e:
            logger.warning(f"Could not paste logo on back cover: {e}")

    # Generate and paste barcode at bottom-right
    if barcode_isbn:
        barcode_bytes = _generate_isbn13_barcode_image(barcode_isbn)
        if barcode_bytes:
            try:
                barcode_im = PILImage.open(BytesIO(barcode_bytes)).convert("RGB")
                max_barcode_h = int(height * 0.14)
                w, h = barcode_im.size
                if h > max_barcode_h:
                    ratio = max_barcode_h / h
                    new_w = int(w * ratio)
                    barcode_im = barcode_im.resize((new_w, max_barcode_h), PILImage.Resampling.LANCZOS)
                bw, bh = barcode_im.size
                x_barcode = width - padding - bw
                y_barcode = height - padding - bh
                image.paste(barcode_im, (x_barcode, y_barcode))
            except Exception as e:
                logger.warning(f"Could not paste barcode on back cover: {e}")

    # Draw all text on top (title, description, tagline, website, ISBN, age)
    with_text_bytes = BytesIO()
    image.save(with_text_bytes, format="JPEG", quality=90, optimize=True)
    with_text_bytes.seek(0)
    return overlay_text_on_image(with_text_bytes.getvalue(), text_blocks)


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

def validate_character_consistency(
    scene_image_data: bytes,
    reference_image_data: bytes,
    page_number: int,
    timeout_seconds: int = 15,
    story_id: Optional[int] = None,
    character_id: Optional[int] = None,
    scene_image_url: Optional[str] = None,
    reference_image_url: Optional[str] = None,
) -> ConsistencyValidationResult:
    """Wrapper for validation_utils.validate_character_consistency; logs result to consistency_validation table."""
    from validation_utils import validate_character_consistency as _validate_character_consistency
    result = _validate_character_consistency(
        scene_image_data=scene_image_data,
        reference_image_data=reference_image_data,
        page_number=page_number,
        gemini_client=gemini_client,
        gemini_text_model=GEMINI_TEXT_MODEL,
        timeout_seconds=timeout_seconds
    )
    log_consistency_validation(
        result=result,
        page_number=page_number,
        story_id=story_id,
        character_id=character_id,
        scene_image_url=scene_image_url,
        reference_image_url=reference_image_url,
    )
    return result


def log_consistency_validation(
    result: ConsistencyValidationResult,
    page_number: int,
    story_id: Optional[int] = None,
    character_id: Optional[int] = None,
    scene_image_url: Optional[str] = None,
    reference_image_url: Optional[str] = None,
) -> None:
    """Log every character comparison to consistency_validation table for analytics and workflow lookup."""
    if not supabase:
        return
    try:
        confidence = None
        if result.details and isinstance(result.details, dict):
            confidence = result.details.get("confidence")
        row = {
            "page_number": page_number,
            "similarity_score": round(float(result.similarity_score), 4),
            "is_consistent": result.is_consistent,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "scene_image_url": scene_image_url,
            "reference_image_url": reference_image_url,
            "details_json": result.details,
        }
        if story_id is not None:
            row["story_id"] = story_id
        if character_id is not None:
            row["character_id"] = character_id
        supabase.table("consistency_validation").insert(row).execute()
        logger.debug(f"Logged consistency validation for page {page_number} (score={result.similarity_score:.3f})")
    except Exception as e:
        logger.warning(f"Failed to log consistency validation: {e}")


@app.post("/api/character/vision/extract")
@limiter.limit("30/minute")
async def character_vision_extract(
    request: Request,
    image: UploadFile = File(...),
    character_id: Optional[int] = Query(None),
    source_image_url: Optional[str] = Query(None),
):
    """
    Character feature extraction via Google Vision API.
    Accepts an uploaded drawing image, runs Vision API (labels + dominant colors),
    stores results in character_features and optionally updates character extraction_data.
    Retries up to 2 times on timeout. Logs API response times.
    """
    if not vision_client:
        raise HTTPException(
            status_code=503,
            detail="Vision API not configured. Set GOOGLE_VISION_API_KEY or GOOGLE_SERVICE_ACCOUNT_JSON_B64.",
        )
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available.")
    # Validate content type
    ct = (image.content_type or "").lower()
    if ct and not ct.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")
    if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image empty or too large (max 10MB).")
    try:
        features, response_time_ms = extract_character_features(image_bytes, vision_client)
    except VisionNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except VisionAPIError as e:
        logger.error("Vision character feature extraction failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    # Store in character_features
    row = {
        "features_json": features,
        "extraction_model": features.get("extraction_model", "google_vision"),
        "response_time_ms": response_time_ms,
        "source_image_url": source_image_url,
    }
    if character_id is not None:
        row["character_id"] = character_id
    try:
        ins = supabase.table("character_features").insert(row).execute()
        feature_row = (ins.data or [None])[0]
        feature_id = feature_row.get("id") if feature_row else None
    except Exception as e:
        logger.error(f"Failed to insert character_features: {e}")
        raise HTTPException(status_code=500, detail="Failed to store extraction result.")
    # Optionally sync to characters.extraction_data for generation workflow
    extraction_for_character = {
        "labels": features.get("labels", []),
        "dominant_colors": features.get("dominant_colors", []),
        "extraction_model": "google_vision",
        "extraction_timestamp": datetime.utcnow().isoformat(),
    }
    if character_id is not None:
        try:
            supabase.table("characters").update({"extraction_data": extraction_for_character}).eq("id", character_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update character extraction_data: {e}")
    return {
        "success": True,
        "features": features,
        "character_feature_id": feature_id,
        "character_id": character_id,
        "response_time_ms": response_time_ms,
    }


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
        
        from prompt_loader import get_prompt
        validation_prompt = get_prompt("imageQualityValidation")
        
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

    The caller must provide a fully composed scene_prompt from the frontend.
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
        
        prompt = (scene_prompt or "").strip()
        if not prompt:
            raise ValueError("scene_prompt is required for story scene image generation")
        logger.info(f"Using scene prompt from frontend for page {page_number}")

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
        "sub": user_id,  # Standard claim for compatibility (e.g. extract_user_from_token)
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
    
    logger.warning("JWT verification failed for all configured secrets")
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
    purchase_type: str  # "single_story", "story_bundle", or "gift"
    story_id: Optional[str] = None  # Story ID to mark as purchased after payment
    gift_id: Optional[str] = None  # Gift ID for gift purchases
    user_email: Optional[str] = None
    user_id: Optional[str] = None
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class CreatePaymentIntentRequest(BaseModel):
    """Request model for creating a payment intent for Stripe Elements"""
    purchase_type: str  # "single_story", "story_bundle", or "gift"
    amount: int  # Amount in cents
    story_id: Optional[str] = None  # Story ID to mark as purchased after payment
    gift_id: Optional[str] = None  # Gift ID for gift purchases
    user_email: Optional[str] = None
    user_id: Optional[str] = None

class PaymentIntentResponse(BaseModel):
    """Response model for payment intent creation"""
    success: bool
    clientSecret: str
    payment_intent_id: Optional[str] = None

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

class DeductCreditRequest(BaseModel):
    """Request model for deducting user credit"""
    amount: int = 1  # Default to 1 credit

class DeductCreditResponse(BaseModel):
    """Response model for credit deduction"""
    success: bool
    message: Optional[str] = None
    remaining_credits: Optional[int] = None


class GiftCheckNotificationRequest(BaseModel):
    """Request model for checking gift notification and adding credit"""
    gift_id: str


class AddRecipientCreditOnSendRequest(BaseModel):
    """Request model for adding recipient credit when sender completes link gift (with notification send + deduct)"""
    gift_id: str


class AddRecipientCreditOnSendResponse(BaseModel):
    """Response model for add recipient credit on send"""
    success: bool
    message: Optional[str] = None


@app.post("/api/gifts/add-recipient-credit-on-send", response_model=AddRecipientCreditOnSendResponse)
@limiter.limit("30/minute")
async def add_recipient_credit_on_send(request: Request, body: AddRecipientCreditOnSendRequest):
    """
    When sender completes a link gift (notification email sent + deduct sender credit),
    add 1 credit to the recipient (to_user_id). Only for gift_type 'link'.
    Caller must be the sender (from_user_id).
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database is not configured")
    try:
        authorization = request.headers.get("Authorization")
        current_user_id = extract_user_from_token(authorization)
        if not current_user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        gift_id = body.gift_id.strip()
        if not gift_id:
            raise HTTPException(status_code=400, detail="gift_id is required")
        gift_result = supabase.table("gifts").select("*").eq("id", gift_id).execute()
        if not gift_result.data or len(gift_result.data) == 0:
            raise HTTPException(status_code=404, detail="Gift not found")
        gift = gift_result.data[0]
        gift_type = (gift.get("gift_type") or "").strip().lower()
        if gift_type != "link":
            return AddRecipientCreditOnSendResponse(
                success=True,
                message="Not a link gift; no recipient credit added"
            )
        from_user_id = gift.get("from_user_id")
        if not from_user_id or str(from_user_id) != str(current_user_id):
            raise HTTPException(status_code=403, detail="Only the sender can trigger recipient credit for this gift")
        to_user_id = gift.get("to_user_id")
        if not to_user_id:
            return AddRecipientCreditOnSendResponse(
                success=True,
                message="Recipient not yet in system (no to_user_id); no credit added"
            )
        credit_result = supabase.table("users").select("credit").eq("id", to_user_id).execute()
        if not credit_result.data:
            return AddRecipientCreditOnSendResponse(success=True, message="Recipient user not found")
        current_credit = credit_result.data[0].get("credit")
        if current_credit is None:
            current_credit = 0
        else:
            try:
                current_credit = int(current_credit) if isinstance(current_credit, str) else current_credit
            except (ValueError, TypeError):
                current_credit = 0
        new_credit = current_credit + 1
        supabase.table("users").update({"credit": new_credit}).eq("id", to_user_id).execute()
        logger.info(f"Link gift {gift_id}: added 1 credit to recipient {to_user_id}. Credits: {new_credit}")
        return AddRecipientCreditOnSendResponse(success=True, message="Recipient credit added")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_recipient_credit_on_send: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GiftCheckNotificationResponse(BaseModel):
    """Response model for gift check notification"""
    success: bool
    message: Optional[str] = None
    credit_added: bool = False
    remaining_credits: Optional[int] = None


@app.post("/api/gifts/check-notification-and-add-credit", response_model=GiftCheckNotificationResponse)
@limiter.limit("30/minute")
async def check_gift_notification_and_add_credit(request: Request, body: GiftCheckNotificationRequest):
    """
    When the recipient clicks a gift notification on the dashboard:
    1) Mark the gift as checked
    2) Add 1 credit to the recipient (to_user_id / current user)
    Only adds credit once per gift (when first marking as checked).
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database is not configured")

    try:
        authorization = request.headers.get("Authorization")
        current_user_id = extract_user_from_token(authorization)
        if not current_user_id:
            raise HTTPException(status_code=401, detail="Authentication required")

        gift_id = body.gift_id.strip()
        if not gift_id:
            raise HTTPException(status_code=400, detail="gift_id is required")

        # Get gift
        gift_result = supabase.table("gifts").select("*").eq("id", gift_id).execute()
        if not gift_result.data or len(gift_result.data) == 0:
            raise HTTPException(status_code=404, detail="Gift not found")

        gift = gift_result.data[0]
        to_user_id = gift.get("to_user_id")
        delivery_email = (gift.get("delivery_email") or "").strip().lower()

        # Get current user email for matching
        user_result = supabase.table("users").select("id, email").eq("id", current_user_id).execute()
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        current_user_email = (user_result.data[0].get("email") or "").strip().lower()

        # Verify current user is the recipient (to_user_id or delivery_email)
        is_recipient = (
            (to_user_id and str(to_user_id) == str(current_user_id))
            or (delivery_email and current_user_email and delivery_email == current_user_email)
        )
        if not is_recipient:
            raise HTTPException(status_code=403, detail="You are not the recipient of this gift")

        # If already checked, do not add credit again (idempotent)
        if gift.get("checked") is True:
            return GiftCheckNotificationResponse(
                success=True,
                message="Gift already checked",
                credit_added=False
            )

        # Mark gift as checked
        supabase.table("gifts").update({"checked": True}).eq("id", gift_id).execute()

        # Do not add credit when clicking notification: link gift recipient credit is added at send time (add-recipient-credit-on-send). Story gifts never get recipient credit.
        return GiftCheckNotificationResponse(
            success=True,
            message="Gift marked as checked",
            credit_added=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_gift_notification_and_add_credit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stripe/create-payment-intent", response_model=PaymentIntentResponse)
async def create_payment_intent(request: CreatePaymentIntentRequest):
    """
    Create a Stripe payment intent for use with Stripe Elements.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    try:
        logger.info(f"Creating payment intent for purchase_type: {request.purchase_type}, amount: {request.amount}, user_id: {request.user_id}")
        
        # Create payment intent
        payment_intent = stripe.PaymentIntent.create(
            amount=request.amount,
            currency='usd',
            payment_method_types=['card'],
            metadata={
                "user_id": request.user_id or "unknown",
                "purchase_type": request.purchase_type,
                "story_id": request.story_id or "none",
                "gift_id": request.gift_id or "none"
            }
        )
        
        logger.info(f"Created payment intent {payment_intent.id} for {request.purchase_type}")
        
        return PaymentIntentResponse(
            success=True,
            clientSecret=payment_intent.client_secret,
            payment_intent_id=payment_intent.id
        )
        
    except stripe.error.StripeError as e:
        error_detail = f"Stripe error: {str(e)}"
        if hasattr(e, 'user_message'):
            error_detail += f" - {e.user_message}"
        logger.error(f"Stripe error creating payment intent: {error_detail}")
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        logger.error(f"Error creating payment intent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")


@app.post("/api/stripe/create-onetime-checkout", response_model=SubscriptionResponse)
async def create_onetime_checkout(request: CreateOnetimeCheckoutRequest):
    """
    Create a Stripe checkout session for one-time purchases (single story or story bundle).
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    try:
        logger.info(f"Creating checkout session for purchase_type: {request.purchase_type}, user_id: {request.user_id}")
        
        # Determine price ID based on purchase type
        if request.purchase_type == "single_story":
            price_id = STRIPE_PRICE_ID_SINGLE_STORY
        elif request.purchase_type == "story_bundle":
            price_id = STRIPE_PRICE_ID_STORY_BUNDLE
        elif request.purchase_type == "gift":
            price_id = STRIPE_PRICE_ID_GIFT
        else:
            raise HTTPException(status_code=400, detail=f"Invalid purchase_type: {request.purchase_type}")
        
        if not price_id:
            logger.error(f"Price ID not configured for purchase_type: {request.purchase_type}")
            raise HTTPException(status_code=503, detail=f"Price ID not configured for {request.purchase_type}")
        
        logger.info(f"Using price_id: {price_id} for purchase_type: {request.purchase_type}")
        
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
                "story_id": request.story_id or "none",
                "gift_id": request.gift_id or "none"
            }
        )
        
        logger.info(f"Created one-time checkout session {checkout_session.id} for {request.purchase_type}")
        
        return SubscriptionResponse(
            success=True,
            checkout_url=checkout_session.url,
            session_id=checkout_session.id
        )
        
    except stripe.error.StripeError as e:
        error_detail = f"Stripe error: {str(e)}"
        if hasattr(e, 'user_message'):
            error_detail += f" - {e.user_message}"
        logger.error(f"Stripe error creating one-time checkout: {error_detail}")
        logger.error(f"Request details - purchase_type: {request.purchase_type}, price_id: {STRIPE_PRICE_ID_GIFT if request.purchase_type == 'gift' else 'N/A'}")
        raise HTTPException(status_code=400, detail=error_detail)
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


@app.post("/api/users/deduct-credit", response_model=DeductCreditResponse)
@limiter.limit("20/minute")
async def deduct_credit(request: Request, deduct_request: DeductCreditRequest):
    """
    Deduct credit from user's account.
    Requires authentication via Bearer token.
    """
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
        
        # Get current user credit
        user_result = supabase.table("users").select("credit").eq("id", user_id).execute()
        
        if not user_result.data or len(user_result.data) == 0:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        current_credit = user_result.data[0].get("credit")
        
        # Parse credit as integer (handle string or number)
        if current_credit is None:
            current_credit = 0
        else:
            try:
                current_credit = int(current_credit) if isinstance(current_credit, str) else current_credit
            except (ValueError, TypeError):
                current_credit = 0
        
        # Check if user has enough credit
        if current_credit < deduct_request.amount:
            return DeductCreditResponse(
                success=False,
                message="Insufficient credits",
                remaining_credits=current_credit
            )
        
        # Deduct credit
        new_credit = current_credit - deduct_request.amount
        
        # Update user credit in database
        update_result = supabase.table("users").update({
            "credit": new_credit
        }).eq("id", user_id).execute()
        
        if not update_result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update user credit"
            )
        
        logger.info(f"Deducted {deduct_request.amount} credit(s) from user {user_id}. Remaining: {new_credit}")
        
        return DeductCreditResponse(
            success=True,
            message=f"Successfully deducted {deduct_request.amount} credit(s)",
            remaining_credits=new_credit
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deducting credit: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deduct credit: {str(e)}")


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


def get_product_metadata_from_checkout_session(session):
    """
    Extract product metadata (credit/credits) from a Stripe checkout session (one-time payment).
    Uses session line items -> price -> product metadata.
    Returns dict with 'credit' (int) if found, None otherwise.
    """
    try:
        session_id = session.get("id")
        if not session_id:
            return None
        line_items = stripe.checkout.Session.list_line_items(session_id, expand=["data.price.product"])
        data = line_items.get("data", [])
        if not data:
            logger.warning("No line items found in checkout session")
            return None
        price = data[0].get("price") or {}
        product = price.get("product")
        if not product:
            return None
        if isinstance(product, str):
            product = stripe.Product.retrieve(product)
        product_metadata = product.get("metadata", {})
        credit = product_metadata.get("credit") or product_metadata.get("credits")
        if credit is None:
            return None
        try:
            credit = int(credit)
        except (TypeError, ValueError):
            logger.warning(f"Product metadata credit not a number: {credit}")
            return None
        if credit < 1:
            return None
        logger.info(f"Checkout session product metadata: credit={credit}")
        return {"credit": credit}
    except Exception as e:
        logger.error(f"Error retrieving product metadata from checkout session: {e}")
        return None


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


def mark_story_as_purchased_for_user(
    story_id_raw: Optional[str],
    user_id_raw: Optional[str],
    customer_email: Optional[str] = None
) -> bool:
    """
    Mark stories.purchased=True for the specific story and owner user.
    Returns True when an update is applied, otherwise False.
    """
    if not supabase:
        logger.warning("Cannot mark story as purchased: Supabase is not configured")
        return False

    if not story_id_raw or str(story_id_raw).strip().lower() in ("none", "null", ""):
        return False

    story_identifier = str(story_id_raw).strip()
    story_id_int: Optional[int] = None
    use_uid = False

    # Frontend preview flow typically sends stories.uid in query params; support both id and uid.
    try:
        story_id_int = int(story_identifier)
    except (TypeError, ValueError):
        use_uid = True

    target_user_id: Optional[str] = None
    if user_id_raw and str(user_id_raw).strip().lower() not in ("unknown", "none", "null", ""):
        target_user_id = str(user_id_raw).strip()
    elif customer_email:
        try:
            user_result = supabase.table("users").select("id").eq("email", customer_email).limit(1).execute()
            if user_result.data and len(user_result.data) > 0:
                target_user_id = user_result.data[0].get("id")
        except Exception as e:
            logger.warning(f"Could not resolve user by email {customer_email} for purchase update: {e}")

    if not target_user_id:
        story_ref = f"uid={story_identifier}" if use_uid else f"id={story_id_int}"
        logger.warning(
            f"Skipping purchased=true update for story ({story_ref}): user could not be resolved "
            f"(user_id={user_id_raw}, email={customer_email})"
        )
        return False

    try:
        update_query = supabase.table("stories").update({"purchased": True})
        if use_uid:
            update_query = update_query.eq("uid", story_identifier)
        else:
            update_query = update_query.eq("id", story_id_int)

        response = update_query.eq("user_id", target_user_id).execute()
        if response.data and len(response.data) > 0:
            story_ref = f"uid={story_identifier}" if use_uid else f"id={story_id_int}"
            logger.info(f"✅ Marked story ({story_ref}) as purchased for user {target_user_id}")
            return True

        logger.warning(
            f"No matching story row updated for purchased=true "
            f"(story_ref={'uid=' + story_identifier if use_uid else 'id=' + str(story_id_int)}, user_id={target_user_id})"
        )
        return False
    except Exception as e:
        story_ref = f"uid={story_identifier}" if use_uid else f"id={story_id_int}"
        logger.error(f"Error updating stories.purchased for story ({story_ref}), user {target_user_id}: {e}")
        return False


async def handle_checkout_completed(session):
    """Handle successful checkout session completion"""
    try:
        mode = session.get("mode")
        metadata = session.get("metadata", {})
        logger.info(f"Checkout completed: {session}")
        
        # Handle one-time payment (story purchase or gift purchase)
        if mode == "payment":
            story_id = metadata.get("story_id")
            gift_id = metadata.get("gift_id")
            user_id = metadata.get("user_id")
            purchase_type = metadata.get("purchase_type")
            payment_status = session.get("payment_status")
            
            logger.info(f"Checkout completed for one-time payment: purchase_type={purchase_type}, story_id={story_id}, gift_id={gift_id}, user_id={user_id}")
            
            # Handle gift purchase
            if purchase_type == "gift" and payment_status == "paid":
                logger.info(f"✅ Gift purchase completed successfully: gift_id={gift_id}, user_id={user_id}")
                # Gift will be created/updated by the frontend after payment verification
                # The webhook just logs the successful payment
                return
            
            # Single story or story bundle: update user credits from Stripe product metadata
            if purchase_type in ("single_story", "story_bundle") and payment_status == "paid" and supabase:
                product_metadata = get_product_metadata_from_checkout_session(session)
                if product_metadata:
                    credits_to_add = product_metadata.get("credit", 0) or 0
                    if credits_to_add > 0:
                        customer_email = session.get("customer_email") or session.get("customer_details", {}).get("email")
                        target_user_id = None
                        if user_id and str(user_id).strip() and str(user_id) != "unknown":
                            target_user_id = user_id
                        elif customer_email:
                            try:
                                user_result = supabase.table("users").select("id").eq("email", customer_email).execute()
                                if user_result.data and len(user_result.data) > 0:
                                    target_user_id = user_result.data[0].get("id")
                            except Exception as e:
                                logger.warning(f"Could not find user by email {customer_email}: {e}")
                        if target_user_id:
                            try:
                                credit_result = supabase.table("users").select("credit").eq("id", target_user_id).execute()
                                if credit_result.data:
                                    current_credit = credit_result.data[0].get("credit")
                                    if current_credit is None:
                                        current_credit = 0
                                    try:
                                        current_credit = int(current_credit) if isinstance(current_credit, str) else current_credit
                                    except (TypeError, ValueError):
                                        current_credit = 0
                                    new_credit = current_credit + credits_to_add
                                    supabase.table("users").update({"credit": new_credit}).eq("id", target_user_id).execute()
                                    logger.info(f"Updated user {target_user_id} credits: +{credits_to_add} (was {current_credit}, now {new_credit}) from {purchase_type} purchase (Stripe product metadata)")
                                else:
                                    logger.warning(f"User {target_user_id} not found when updating credits")
                            except Exception as e:
                                logger.error(f"Error adding credits for {purchase_type} purchase: {e}")
                        else:
                            logger.warning(f"Cannot add credits: no user_id or customer_email for {purchase_type} purchase")
                else:
                    logger.warning(f"No product metadata (credit) for {purchase_type} purchase; set Stripe product metadata 'credit' or 'credits'")
            
            # Mark story as purchased in stories table for current user + story
            if story_id and payment_status == "paid" and supabase:
                try:
                    customer_email = session.get("customer_email") or session.get("customer_details", {}).get("email")
                    mark_story_as_purchased_for_user(
                        story_id_raw=story_id,
                        user_id_raw=user_id,
                        customer_email=customer_email
                    )
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
async def get_checkout_session(
    session_id: str,
    fallback_story_id: Optional[str] = None,
    fallback_user_id: Optional[str] = None
):
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
        story_id = None
        metadata = session.get("metadata", {})
        if mode == "payment":
            purchase_type = metadata.get("purchase_type", "single_story")
            metadata_story_id = metadata.get("story_id")
            metadata_user_id = metadata.get("user_id")

            # Fallback to query params when Stripe metadata is missing/placeholder.
            story_id = metadata_story_id
            if not story_id or str(story_id).strip().lower() in ("none", "null", ""):
                story_id = fallback_story_id

            resolved_user_id = metadata_user_id
            if not resolved_user_id or str(resolved_user_id).strip().lower() in ("unknown", "none", "null", ""):
                resolved_user_id = fallback_user_id

            if payment_status == "paid" and purchase_type in ("single_story", "story_bundle"):
                # Also update here so frontend verification path guarantees purchased=true
                mark_story_as_purchased_for_user(
                    story_id_raw=story_id,
                    user_id_raw=resolved_user_id,
                    customer_email=customer_email
                )
        
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
            "story_id": story_id,
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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: Optional[str] = None
    auth_provider: Optional[str] = None
    google_id: Optional[str] = None


# --- Passwordless auth (Twilio Verify) ---
class RequestOtpRequest(BaseModel):
    phone: str  # E.164 format, e.g. +15551234567


class VerifyOtpRequest(BaseModel):
    phone: str
    code: str  # 4–10 digit verification code


@app.post("/api/auth/request-otp")
@limiter.limit("6/minute")  # Prevent SMS abuse
async def auth_request_otp(request: Request, body: RequestOtpRequest):
    """
    Send SMS verification code via Twilio Verify.
    Used for passwordless phone login/signup.
    """
    if not twilio_verify_client:
        raise HTTPException(
            status_code=503,
            detail="SMS verification is not configured. Please contact support."
        )
    phone = (body.phone or "").strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")
    # Normalize to E.164: ensure leading +
    if not phone.startswith("+"):
        phone = "+" + re.sub(r"\D", "", phone)
    if not re.match(r"^\+\d{10,15}$", phone):
        raise HTTPException(status_code=400, detail="Invalid phone number format")
    try:
        verification = twilio_verify_client.verify.v2.services(
            TWILIO_VERIFY_SERVICE_SID
        ).verifications.create(to=phone, channel="sms")
        return {"success": True, "message": "Verification code sent"}
    except Exception as e:
        err_msg = str(e).lower()
        if "rate" in err_msg or "limit" in err_msg:
            raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
        if "invalid" in err_msg or "unverified" in err_msg:
            raise HTTPException(status_code=400, detail="This phone number cannot receive SMS.")
        logger.exception("Twilio Verify send failed")
        raise HTTPException(status_code=500, detail="Failed to send verification code.")


@app.post("/api/auth/verify-otp")
@limiter.limit("10/minute")
async def auth_verify_otp(request: Request, body: VerifyOtpRequest):
    """
    Verify SMS code with Twilio, then get-or-create user and return JWT.
    Passwordless login/signup: new users are created automatically.
    """
    if not twilio_verify_client or not supabase:
        raise HTTPException(
            status_code=503,
            detail="Verification or database is not available."
        )
    phone = (body.phone or "").strip()
    if not phone.startswith("+"):
        phone = "+" + re.sub(r"\D", "", phone)
    if not re.match(r"^\+\d{10,15}$", phone):
        raise HTTPException(status_code=400, detail="Invalid phone number format")
    code = (body.code or "").strip()
    if not code or not code.isdigit():
        raise HTTPException(status_code=400, detail="Verification code is required")
    try:
        check = twilio_verify_client.verify.v2.services(
            TWILIO_VERIFY_SERVICE_SID
        ).verification_checks.create(to=phone, code=code)
        if check.status != "approved":
            raise HTTPException(status_code=400, detail="Invalid or expired code.")
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e).lower()
        if "invalid" in err_msg or "expired" in err_msg or "404" in err_msg:
            raise HTTPException(status_code=400, detail="Invalid or expired code.")
        logger.exception("Twilio Verify check failed")
        raise HTTPException(status_code=500, detail="Verification failed.")
    # Get or create user in our users table (phone-only, no Supabase Auth user)
    try:
        existing = supabase.table("users").select("id, first_name, last_name, email, phone, role").eq("phone", phone).execute()
        if existing.data and len(existing.data) > 0:
            row = existing.data[0]
            user_id = row["id"]
            is_new = False
        else:
            user_id = str(uuid.uuid4())
            today = datetime.utcnow().strftime("%Y-%m-%d")
            supabase.table("users").insert({
                "id": user_id,
                "phone": phone,
                "role": "adult",
                "last_login": today,
                "upload_cnt": 10,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }).execute()
            is_new = True
            row = {"id": user_id, "first_name": None, "last_name": None, "email": None, "phone": phone, "role": "adult"}
    except Exception as e:
        logger.exception("Get/create user failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not sign you in.")
    # Update last_login for existing users
    if not is_new:
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            supabase.table("users").update({
                "last_login": today,
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", user_id).execute()
        except Exception:
            pass
    access_token = create_jwt_token(user_id)
    return {
        "success": True,
        "access_token": access_token,
        "user": {
            "id": row.get("id"),
            "email": row.get("email"),
            "phone": row.get("phone"),
            "first_name": row.get("first_name"),
            "last_name": row.get("last_name"),
            "role": row.get("role", "adult"),
        },
        "is_new_user": is_new,
    }


@app.get("/api/auth/me")
@limiter.limit("60/minute")
async def auth_me(request: Request, authorization: Optional[str] = Header(None)):
    """
    Return current user profile for the given JWT (e.g. passwordless phone session).
    """
    user_id = extract_user_from_token(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        result = supabase.table("users").select("id, email, phone, first_name, last_name, role").eq("id", user_id).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        row = result.data[0]
        return {
            "success": True,
            "user": {
                "id": row.get("id"),
                "email": row.get("email"),
                "phone": row.get("phone"),
                "first_name": row.get("first_name"),
                "last_name": row.get("last_name"),
                "role": row.get("role", "adult"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Auth me failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not load profile.")


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
                        "gift_message": gift.get("special_msg", "Enjoy your special story!"),
                        "gift_order_id": gift_id,
                        "scenario": "giver_creating",
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
    Sync Supabase Auth users into the app's public.users table.

    This endpoint is intentionally server-side because it writes trusted profile
    fields with the service-role Supabase client. The caller must send the
    current Supabase access token in the Authorization header.
    """
    try:
        if not supabase:
            raise HTTPException(
                status_code=500,
                detail="Database service not available"
            )

        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")

        access_token = authorization[7:]
        token_user_id = None
        auth_user = None

        try:
            auth_response = supabase.auth.get_user(access_token)
            auth_user = getattr(auth_response, "user", None)
            token_user_id = getattr(auth_user, "id", None)
        except Exception as auth_error:
            logger.warning("Supabase auth.get_user failed during auth sync: %s", auth_error)
            token_payload = verify_jwt_token(access_token)
            token_user_id = token_payload.get("sub") if token_payload else None

        if not token_user_id:
            raise HTTPException(status_code=401, detail="Invalid authentication token")

        if body.user_id and body.user_id != token_user_id:
            raise HTTPException(status_code=403, detail="Cannot sync a different user")

        def read_auth_value(value: Any, key: str, default: Any = None) -> Any:
            if isinstance(value, dict):
                return value.get(key, default)
            return getattr(value, key, default)

        auth_metadata = getattr(auth_user, "user_metadata", None) or {}
        app_metadata = getattr(auth_user, "app_metadata", None) or {}
        identities = getattr(auth_user, "identities", None) or []
        google_identity = next(
            (identity for identity in identities if read_auth_value(identity, "provider") == "google"),
            None
        )
        google_identity_data = read_auth_value(google_identity, "identity_data", {}) or {}

        email = (
            (body.email or "").strip().lower()
            or (getattr(auth_user, "email", None) or "").strip().lower()
            or str(auth_metadata.get("email") or google_identity_data.get("email") or "").strip().lower()
        )
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")

        full_name = (
            (body.name or "").strip()
            or str(auth_metadata.get("full_name") or auth_metadata.get("name") or "").strip()
            or str(google_identity_data.get("full_name") or google_identity_data.get("name") or "").strip()
        )
        name_parts = full_name.split()
        first_name = (
            (body.first_name or "").strip()
            or str(auth_metadata.get("given_name") or google_identity_data.get("given_name") or "").strip()
            or (name_parts[0] if name_parts else "")
        )
        last_name = (
            (body.last_name or "").strip()
            or str(auth_metadata.get("family_name") or google_identity_data.get("family_name") or "").strip()
            or (" ".join(name_parts[1:]) if len(name_parts) > 1 else "")
        )
        avatar_url = (
            (body.avatar_url or "").strip()
            or str(auth_metadata.get("avatar_url") or auth_metadata.get("picture") or "").strip()
            or str(google_identity_data.get("avatar_url") or google_identity_data.get("picture") or "").strip()
            or None
        )
        auth_provider = (
            (body.auth_provider or "").strip()
            or str(app_metadata.get("provider") or "").strip()
            or (read_auth_value(google_identity, "provider") if google_identity else "")
            or "email"
        )
        google_id = (
            (body.google_id or "").strip()
            or str(auth_metadata.get("provider_id") or google_identity_data.get("provider_id") or "").strip()
            or (read_auth_value(google_identity, "id") if google_identity else "")
            or (token_user_id if auth_provider == "google" else "")
            or None
        )
        role = (body.role or "").strip() or "adult"
        today = datetime.utcnow().strftime("%Y-%m-%d")
        now = datetime.utcnow().isoformat()

        def is_google_id_type_error(error: Exception) -> bool:
            error_text = str(error).lower()
            return (
                google_id is not None
                and "22003" in error_text
                and "bigint" in error_text
            )

        logger.info(f"Auth sync requested for user: {token_user_id} ({email}, provider={auth_provider})")

        existing_response = supabase.table("users").select("*").eq("id", token_user_id).execute()
        existing_user = existing_response.data[0] if existing_response.data else None

        if not existing_user:
            email_response = supabase.table("users").select("*").eq("email", email).execute()
            existing_user = email_response.data[0] if email_response.data else None

        is_new_user = existing_user is None
        welcome_email_sent = False

        user_data = {
            "id": token_user_id,
            "email": email,
            "first_name": first_name or None,
            "last_name": last_name or None,
            "avatar_url": avatar_url,
            "role": role,
            "google_id": google_id,
            "last_login": today,
            "updated_at": now,
        }
        
        if is_new_user:
            insert_data = {
                **user_data,
                "created_at": now,
                "upload_cnt": 10,
                "subscription_status": "free",
            }
            try:
                supabase.table("users").insert(insert_data).execute()
            except Exception as db_error:
                if not is_google_id_type_error(db_error):
                    raise
                logger.warning(
                    "users.google_id appears to be BIGINT and cannot store Google provider id %s. "
                    "Retrying auth sync without google_id; apply the google_id TEXT migration.",
                    google_id,
                )
                insert_data["google_id"] = None
                user_data["google_id"] = None
                supabase.table("users").insert(insert_data).execute()
            logger.info(f"New user synced to public.users: {token_user_id}")
        else:
            update_data = {
                **user_data,
                "first_name": first_name or existing_user.get("first_name"),
                "last_name": last_name or existing_user.get("last_name"),
                "avatar_url": avatar_url or existing_user.get("avatar_url"),
                "role": existing_user.get("role") or role,
            }
            try:
                supabase.table("users").update(update_data).eq("id", existing_user["id"]).execute()
            except Exception as db_error:
                if not is_google_id_type_error(db_error):
                    raise
                logger.warning(
                    "users.google_id appears to be BIGINT and cannot store Google provider id %s. "
                    "Retrying auth sync without google_id; apply the google_id TEXT migration.",
                    google_id,
                )
                update_data["google_id"] = existing_user.get("google_id")
                user_data["google_id"] = existing_user.get("google_id")
                supabase.table("users").update(update_data).eq("id", existing_user["id"]).execute()
            logger.info(f"Existing user synced to public.users: {existing_user['id']} -> {token_user_id}")

        if auth_provider == "google" and is_new_user:
            try:
                supabase.table("user_auth_history").insert({
                    "user_id": token_user_id,
                    "event_type": "register",
                    "auth_provider": "google_oauth",
                }).execute()
            except Exception as history_error:
                logger.warning("Failed to write Google auth history: %s", history_error)

        if is_new_user:
            logger.info(f"New user detected: {token_user_id}, sending welcome email")
            
            # Get user's name for the email
            customer_name = full_name or " ".join([first_name, last_name]).strip() or None
            
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
            logger.info(f"Existing user {token_user_id}, skipping welcome email")
        
        return {
            "success": True,
            "is_new_user": is_new_user,
            "welcome_email_sent": welcome_email_sent,
            "user": user_data,
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
