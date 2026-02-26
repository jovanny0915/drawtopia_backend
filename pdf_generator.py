"""
PDF Generation System
Generates print-ready PDFs for both Interactive Search and Story Adventure formats
"""

import logging
import time
from io import BytesIO
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image as PILImage, ImageFilter, ImageDraw
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

logger = logging.getLogger(__name__)

# PDF Settings
PDF_DPI = 300  # Print-ready DPI
PAGE_WIDTH = letter[0]  # 8.5 inches
PAGE_HEIGHT = letter[1]  # 11 inches
MARGIN = 0.5 * inch  # 0.5 inch margins

# Branding
BRAND_NAME = "Drawtopia"
BRAND_COLOR = HexColor("#4A90E2")  # Blue color (adjust as needed)


def download_image_from_url(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download image from URL and return bytes"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def resize_image_for_pdf(image_data: bytes, target_width: float, target_height: float, dpi: int = 300) -> Optional[PILImage.Image]:
    """
    Resize image to fit PDF dimensions at specified DPI
    Maintains aspect ratio and ensures high quality
    """
    try:
        image = PILImage.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            background = PILImage.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate target size in pixels (DPI * inches)
        target_width_px = int(target_width * dpi / 72)  # Convert points to pixels
        target_height_px = int(target_height * dpi / 72)
        
        # Resize with high-quality resampling
        image = image.resize((target_width_px, target_height_px), PILImage.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None


def add_branding_footer(c: canvas.Canvas, page_num: int, total_pages: int):
    """Add branding footer to PDF pages"""
    footer_y = 0.3 * inch
    footer_text = f"{BRAND_NAME} | Page {page_num} of {total_pages}"
    
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#666666"))
    text_width = c.stringWidth(footer_text, "Helvetica", 8)
    c.drawString((PAGE_WIDTH - text_width) / 2, footer_y, footer_text)


def create_interactive_search_pdf(
    character_name: str,
    story_title: str,
    character_image_url: Optional[str],
    scene_urls: List[str],
    output_buffer: BytesIO
) -> bool:
    """
    Create Interactive Search PDF format:
    - Cover page with title and character
    - 4 full-page scene spreads
    - Back cover with branding
    """
    try:
        start_time = time.time()
        logger.info(f"Creating Interactive Search PDF for {character_name}")
        
        # Create PDF canvas
        c = canvas.Canvas(output_buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
        
        # Calculate image dimensions (full page minus margins)
        image_width = PAGE_WIDTH - (2 * MARGIN)
        image_height = PAGE_HEIGHT - (2 * MARGIN)
        
        page_num = 1
        total_pages = 6  # Cover + 4 scenes + Back cover
        
        # === COVER PAGE ===
        logger.info("Creating cover page...")
        c.setFillColor(white)
        c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        
        # Title
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 36)
        title_y = PAGE_HEIGHT - 2 * inch
        title_width = c.stringWidth(story_title, "Helvetica-Bold", 36)
        c.drawString((PAGE_WIDTH - title_width) / 2, title_y, story_title)
        
        # Character image (if available)
        if character_image_url:
            char_image_data = download_image_from_url(character_image_url)
            if char_image_data:
                char_image = resize_image_for_pdf(char_image_data, 4 * inch, 4 * inch, PDF_DPI)
                if char_image:
                    char_img_reader = ImageReader(char_image)
                    char_x = (PAGE_WIDTH - 4 * inch) / 2
                    char_y = PAGE_HEIGHT - 6.5 * inch
                    c.drawImage(char_img_reader, char_x, char_y, width=4 * inch, height=4 * inch)
        
        # Character name
        c.setFont("Helvetica", 24)
        char_name_y = 2 * inch
        char_name_width = c.stringWidth(f"Starring {character_name}", "Helvetica", 24)
        c.drawString((PAGE_WIDTH - char_name_width) / 2, char_name_y, f"Starring {character_name}")
        
        add_branding_footer(c, page_num, total_pages)
        c.showPage()
        page_num += 1
        
        # === 4 FULL-PAGE SCENE SPREADS ===
        for i, scene_url in enumerate(scene_urls[:4], 1):
            logger.info(f"Adding scene {i}/4...")
            c.setFillColor(white)
            c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
            
            scene_image_data = download_image_from_url(scene_url)
            if scene_image_data:
                scene_image = resize_image_for_pdf(scene_image_data, image_width, image_height, PDF_DPI)
                if scene_image:
                    scene_img_reader = ImageReader(scene_image)
                    c.drawImage(scene_img_reader, MARGIN, MARGIN, width=image_width, height=image_height)
                else:
                    logger.warning(f"Failed to resize scene {i} image")
            else:
                logger.warning(f"Failed to download scene {i} image from {scene_url}")
            
            add_branding_footer(c, page_num, total_pages)
            c.showPage()
            page_num += 1
        
        # === BACK COVER ===
        logger.info("Creating back cover...")
        c.setFillColor(white)
        c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        
        # Branding
        c.setFillColor(BRAND_COLOR)
        c.setFont("Helvetica-Bold", 32)
        brand_y = PAGE_HEIGHT - 3 * inch
        brand_width = c.stringWidth(BRAND_NAME, "Helvetica-Bold", 32)
        c.drawString((PAGE_WIDTH - brand_width) / 2, brand_y, BRAND_NAME)
        
        # Tagline or additional info
        c.setFillColor(HexColor("#666666"))
        c.setFont("Helvetica", 14)
        tagline = "Creating magical stories for children"
        tagline_y = PAGE_HEIGHT - 4.5 * inch
        tagline_width = c.stringWidth(tagline, "Helvetica", 14)
        c.drawString((PAGE_WIDTH - tagline_width) / 2, tagline_y, tagline)
        
        add_branding_footer(c, page_num, total_pages)
        c.showPage()
        
        # Save PDF
        c.save()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Interactive Search PDF created successfully in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error creating Interactive Search PDF: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return False


def create_story_adventure_pdf(
    character_name: str,
    story_title: str,
    character_image_url: Optional[str],
    story_pages: List[Dict[str, Any]],  # List of pages with 'text' and 'scene' (URL)
    audio_urls: Optional[List[Optional[str]]] = None,
    output_buffer: BytesIO = None
) -> bool:
    """
    Create Story Adventure PDF format:
    - Cover page with title and character
    - 5 illustrated pages with text
    - Audio access information
    - Back cover with branding
    """
    try:
        start_time = time.time()
        logger.info(f"Creating Story Adventure PDF for {character_name}")
        
        if output_buffer is None:
            output_buffer = BytesIO()
        
        # Create PDF canvas
        c = canvas.Canvas(output_buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
        
        # Calculate dimensions
        image_width = PAGE_WIDTH - (2 * MARGIN)
        image_height = (PAGE_HEIGHT - (2 * MARGIN)) * 0.6  # 60% for image
        text_area_height = (PAGE_HEIGHT - (2 * MARGIN)) * 0.35  # 35% for text
        
        total_pages = 2 + len(story_pages) + 1  # Cover + story pages + back cover
        page_num = 1
        
        # === COVER PAGE ===
        logger.info("Creating cover page...")
        c.setFillColor(white)
        c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        
        # Title
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 36)
        title_y = PAGE_HEIGHT - 2 * inch
        title_width = c.stringWidth(story_title, "Helvetica-Bold", 36)
        c.drawString((PAGE_WIDTH - title_width) / 2, title_y, story_title)
        
        # Character image (if available)
        if character_image_url:
            char_image_data = download_image_from_url(character_image_url)
            if char_image_data:
                char_image = resize_image_for_pdf(char_image_data, 4 * inch, 4 * inch, PDF_DPI)
                if char_image:
                    char_img_reader = ImageReader(char_image)
                    char_x = (PAGE_WIDTH - 4 * inch) / 2
                    char_y = PAGE_HEIGHT - 6.5 * inch
                    c.drawImage(char_img_reader, char_x, char_y, width=4 * inch, height=4 * inch)
        
        # Character name
        c.setFont("Helvetica", 24)
        char_name_y = 2 * inch
        char_name_width = c.stringWidth(f"Starring {character_name}", "Helvetica", 24)
        c.drawString((PAGE_WIDTH - char_name_width) / 2, char_name_y, f"Starring {character_name}")
        
        add_branding_footer(c, page_num, total_pages)
        c.showPage()
        page_num += 1
        
        # === 5 ILLUSTRATED PAGES WITH TEXT ===
        for i, page_data in enumerate(story_pages[:5], 1):
            logger.info(f"Adding story page {i}/5...")
            c.setFillColor(white)
            c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
            
            page_text = page_data.get('text', '')
            scene_url = page_data.get('scene')
            
            # Scene image (top 60% of page)
            if scene_url:
                scene_image_data = download_image_from_url(str(scene_url))
                if scene_image_data:
                    scene_image = resize_image_for_pdf(scene_image_data, image_width, image_height, PDF_DPI)
                    if scene_image:
                        scene_img_reader = ImageReader(scene_image)
                        img_y = PAGE_HEIGHT - MARGIN - image_height
                        c.drawImage(scene_img_reader, MARGIN, img_y, width=image_width, height=image_height)
            
            # Story text (bottom 35% of page)
            text_y = MARGIN + text_area_height
            text_x = MARGIN + 0.2 * inch
            text_width = PAGE_WIDTH - (2 * MARGIN) - 0.4 * inch
            
            # Draw text with word wrapping
            c.setFillColor(black)
            c.setFont("Helvetica", 14)
            
            # Simple text wrapping (split into lines)
            words = page_text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if c.stringWidth(test_line, "Helvetica", 14) <= text_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw lines
            line_height = 18
            for j, line in enumerate(lines):
                y_pos = text_y - (j * line_height)
                if y_pos < MARGIN:
                    break  # Don't draw below margin
                c.drawString(text_x, y_pos, line)
            
            add_branding_footer(c, page_num, total_pages)
            c.showPage()
            page_num += 1
        
        # === AUDIO ACCESS INFORMATION PAGE ===
        if audio_urls and any(audio_urls):
            logger.info("Adding audio access information page...")
            c.setFillColor(white)
            c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
            
            c.setFillColor(black)
            c.setFont("Helvetica-Bold", 24)
            audio_title = "Audio Version Available"
            title_width = c.stringWidth(audio_title, "Helvetica-Bold", 24)
            c.drawString((PAGE_WIDTH - title_width) / 2, PAGE_HEIGHT - 2 * inch, audio_title)
            
            c.setFont("Helvetica", 14)
            info_text = "Scan the QR code or visit the link below to access the audio version of this story:"
            info_y = PAGE_HEIGHT - 3.5 * inch
            info_width = c.stringWidth(info_text, "Helvetica", 14)
            c.drawString((PAGE_WIDTH - info_width) / 2, info_y, info_text)
            
            # List audio URLs (if available)
            audio_y = PAGE_HEIGHT - 5 * inch
            c.setFont("Helvetica", 12)
            for idx, audio_url in enumerate(audio_urls[:5], 1):
                if audio_url:
                    url_text = f"Page {idx}: {audio_url}"
                    # Truncate if too long
                    if c.stringWidth(url_text, "Helvetica", 12) > PAGE_WIDTH - (2 * MARGIN):
                        url_text = url_text[:50] + "..."
                    c.drawString(MARGIN, audio_y, url_text)
                    audio_y -= 0.3 * inch
            
            add_branding_footer(c, page_num, total_pages)
            c.showPage()
            page_num += 1
            total_pages += 1
        
        # === BACK COVER ===
        logger.info("Creating back cover...")
        c.setFillColor(white)
        c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        
        # Branding
        c.setFillColor(BRAND_COLOR)
        c.setFont("Helvetica-Bold", 32)
        brand_y = PAGE_HEIGHT - 3 * inch
        brand_width = c.stringWidth(BRAND_NAME, "Helvetica-Bold", 32)
        c.drawString((PAGE_WIDTH - brand_width) / 2, brand_y, BRAND_NAME)
        
        # Tagline
        c.setFillColor(HexColor("#666666"))
        c.setFont("Helvetica", 14)
        tagline = "Creating magical stories for children"
        tagline_y = PAGE_HEIGHT - 4.5 * inch
        tagline_width = c.stringWidth(tagline, "Helvetica", 14)
        c.drawString((PAGE_WIDTH - tagline_width) / 2, tagline_y, tagline)
        
        add_branding_footer(c, page_num, total_pages)
        c.showPage()
        
        # Save PDF
        c.save()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Story Adventure PDF created successfully in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error creating Story Adventure PDF: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return False


def create_simple_scene_pdf(
    story_title: str,
    scene_urls: List[str],
    output_buffer: BytesIO
) -> bool:
    """
    Create a simple PDF where each scene image is a full page
    
    Format:
    - Each image on its own page with 1 inch margins
    - A4 pagesize
    - preserveAspectRatio=True
    """
    try:
        start_time = time.time()
        logger.info(f"Creating simple scene PDF: {story_title} with {len(scene_urls)} scenes")
        
        # Create PDF canvas with A4 pagesize
        c = canvas.Canvas(output_buffer, pagesize=A4)
        width, height = A4
        
        # Use 1 inch margins as specified
        margin = 1 * inch
        image_width = width - (2 * margin)
        image_height = height - (2 * margin)
        
        # === ONE FULL PAGE PER SCENE IMAGE ===
        for i, scene_url in enumerate(scene_urls, 1):
            logger.info(f"Adding scene {i}/{len(scene_urls)}...")
            
            scene_image_data = download_image_from_url(scene_url)
            if scene_image_data:
                # Download and prepare image
                try:
                    image = PILImage.open(BytesIO(scene_image_data))
                    
                    # Convert to RGB if necessary
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = PILImage.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        if image.mode in ('RGBA', 'LA'):
                            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                            image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    img_reader = ImageReader(image)
                    
                    # Draw image with 1 inch margins and preserveAspectRatio
                    c.drawImage(
                        img_reader,
                        x=margin,
                        y=margin,
                        width=image_width,
                        height=image_height,
                        preserveAspectRatio=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to process scene {i} image: {e}")
            else:
                logger.warning(f"Failed to download scene {i} image from {scene_url}")
            
            c.showPage()
        
        # Save PDF
        c.save()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Simple scene PDF created successfully in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error creating simple scene PDF: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return False


def _load_image_rgb(url: Optional[str] = None, image_data: Optional[bytes] = None) -> Optional[PILImage.Image]:
    """Load image from URL or bytes and return as RGB PIL Image, or None on failure."""
    if url and not image_data:
        image_data = download_image_from_url(url)
    if not image_data:
        return None
    try:
        image = PILImage.open(BytesIO(image_data))
        if image.mode in ('RGBA', 'LA', 'P'):
            background = PILImage.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.warning(f"Failed to load/convert image: {e}")
        return None


def _draw_image_cover_page(
    c: canvas.Canvas,
    image: PILImage.Image,
    page_width: float,
    page_height: float,
) -> bool:
    """
    Draw image so it covers the entire page (like CSS object-fit: cover).
    Image is scaled to fill the page and centered; no margins. One image = one full page.
    """
    if not image or not image.size[0] or not image.size[1]:
        return False
    try:
        iw, ih = image.size[0], image.size[1]
        scale = max(page_width / iw, page_height / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        x = (page_width - draw_w) / 2
        y = (page_height - draw_h) / 2
        img_reader = ImageReader(image)
        c.drawImage(img_reader, x, y, width=draw_w, height=draw_h)
        return True
    except Exception as e:
        logger.warning(f"Failed to draw image on page: {e}")
        return False


def _draw_full_page_image(
    c: canvas.Canvas,
    url: str,
    page_width: float,
    page_height: float,
    page_label: str = "page",
) -> bool:
    """Download image from URL and draw it to cover the full page (no margins). Returns True if drawn."""
    if not url:
        return False
    image = _load_image_rgb(url=url)
    if not image:
        logger.warning(f"Failed to download {page_label} image from {url}")
        return False
    return _draw_image_cover_page(c, image, page_width, page_height)


def _split_image_left_right(image: PILImage.Image):
    """Split image into left half and right half. Returns (left_pil, right_pil)."""
    w, h = image.size[0], image.size[1]
    mid = w // 2
    left = image.crop((0, 0, mid, h))
    right = image.crop((mid, 0, w, h))
    return left, right


# --- Text overlay styling (match /preview/default) ---
TEXT_WHITE = HexColor("#FFFFFF")
TEXT_WHITE_92 = HexColor("#E6E6E6")  # rgba(255,255,255,0.92)
TEXT_WHITE_85 = HexColor("#D9D9D9")  # rgba(255,255,255,0.85)

_SPECIAL_PAGE_FONT_STATE = {
    "initialized": False,
    "regular": "Helvetica",
    "medium": "Helvetica",
    "semibold": "Helvetica-Bold",
    "bold": "Helvetica-Bold",
    "italic": "Helvetica-Oblique",
    "display": "Times-Roman",
}


def _register_first_available_font(font_name: str, candidates: List[Path]) -> bool:
    """Register the first available TTF font file under font_name."""
    for candidate in candidates:
        try:
            if candidate.exists():
                pdfmetrics.registerFont(TTFont(font_name, str(candidate)))
                return True
        except Exception as e:
            logger.debug(f"Failed to register font {font_name} from {candidate}: {e}")
    return False


def _ensure_special_page_fonts() -> None:
    """Initialize special-page fonts once, with robust fallback fonts."""
    if _SPECIAL_PAGE_FONT_STATE["initialized"]:
        return

    backend_fonts_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    windows_fonts = Path("C:/Windows/Fonts")

    if _register_first_available_font(
        "DrawtopiaQuicksandRegular",
        [
            backend_fonts_dir / "Quicksand-Regular.ttf",
            backend_fonts_dir / "Quicksand-VariableFont_wght.ttf",
            windows_fonts / "Quicksand-Regular.ttf",
            windows_fonts / "Quicksand-VariableFont_wght.ttf",
        ],
    ):
        _SPECIAL_PAGE_FONT_STATE["regular"] = "DrawtopiaQuicksandRegular"

    if _register_first_available_font(
        "DrawtopiaQuicksandMedium",
        [
            backend_fonts_dir / "Quicksand-Medium.ttf",
            windows_fonts / "Quicksand-Medium.ttf",
        ],
    ):
        _SPECIAL_PAGE_FONT_STATE["medium"] = "DrawtopiaQuicksandMedium"
    else:
        _SPECIAL_PAGE_FONT_STATE["medium"] = _SPECIAL_PAGE_FONT_STATE["regular"]

    if _register_first_available_font(
        "DrawtopiaQuicksandSemiBold",
        [
            backend_fonts_dir / "Quicksand-SemiBold.ttf",
            windows_fonts / "Quicksand-SemiBold.ttf",
        ],
    ):
        _SPECIAL_PAGE_FONT_STATE["semibold"] = "DrawtopiaQuicksandSemiBold"
    else:
        _SPECIAL_PAGE_FONT_STATE["semibold"] = _SPECIAL_PAGE_FONT_STATE["bold"]

    if _register_first_available_font(
        "DrawtopiaQuicksandBold",
        [
            backend_fonts_dir / "Quicksand-Bold.ttf",
            windows_fonts / "Quicksand-Bold.ttf",
        ],
    ):
        _SPECIAL_PAGE_FONT_STATE["bold"] = "DrawtopiaQuicksandBold"

    if _register_first_available_font(
        "DrawtopiaDMSerifDisplay",
        [
            backend_fonts_dir / "DMSerifDisplay-Regular.ttf",
            windows_fonts / "DMSerifDisplay-Regular.ttf",
        ],
    ):
        _SPECIAL_PAGE_FONT_STATE["display"] = "DrawtopiaDMSerifDisplay"

    _SPECIAL_PAGE_FONT_STATE["initialized"] = True

_BACK_COVER_LOGO_CANDIDATES = [
    Path(__file__).resolve().parent / "assets" / "white-logo.png",
    Path(__file__).resolve().parents[1] / "drawtopia_frontend" / "src" / "assets" / "white-logo.png",
]

_CTA_LINK_ICON_CANDIDATES = [
    Path(__file__).resolve().parent / "assets" / "Link.svg",
    Path(__file__).resolve().parents[1] / "drawtopia_frontend" / "src" / "assets" / "Link.svg",
]


def _draw_cover_bottom_logo(c: canvas.Canvas, width: float, height: float) -> None:
    """Draw Drawtopia logo centered near the bottom of the cover page."""
    logo_w = width * 0.25
    logo_h = logo_w * 0.223  # Match white-logo.png aspect ratio
    logo_x = (width - logo_w) / 2
    logo_y = max(height * 0.035, 18)

    for logo_path in _BACK_COVER_LOGO_CANDIDATES:
        if logo_path.exists():
            try:
                c.drawImage(
                    ImageReader(str(logo_path)),
                    logo_x,
                    logo_y,
                    width=logo_w,
                    height=logo_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                return
            except Exception as e:
                logger.warning(f"Failed to draw cover logo from {logo_path}: {e}")

    logger.warning("Cover logo not found; skipping Drawtopia logo on cover")


def _draw_cta_link_icon(c: canvas.Canvas, x: float, y: float, size: float) -> None:
    """Draw a compact white link icon for the CTA button."""
    c.saveState()
    c.setStrokeColor(TEXT_WHITE)
    c.setLineWidth(max(1.1, size * 0.105))
    c.setLineCap(1)
    # Two linked loops
    r = size * 0.27
    c.circle(x + size * 0.38, y + size * 0.62, r, stroke=1, fill=0)
    c.circle(x + size * 0.62, y + size * 0.38, r, stroke=1, fill=0)
    # Connector stroke
    c.line(x + size * 0.45, y + size * 0.55, x + size * 0.55, y + size * 0.45)
    c.restoreState()


def _wrap_lines(c: canvas.Canvas, text: str, max_width: float, font_name: str = "Helvetica", font_size: int = 11) -> List[str]:
    """Wrap text into lines that fit within max_width. Returns list of lines."""
    words = text.replace("\n", " \n ").split()
    lines = []
    current = []
    for w in words:
        if w == "\n":
            if current:
                lines.append(" ".join(current))
                current = []
            continue
        trial = " ".join(current + [w]) if current else w
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w] if c.stringWidth(w, font_name, font_size) <= max_width else []
            if current:
                continue
            # word longer than line: break by character not supported here, keep word
            lines.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_centered_text_block(
    c: canvas.Canvas, x_center: float, y_start: float, lines: List[str],
    font_name: str = "Helvetica", font_size: int = 11, leading: float = 14, color: Any = None
) -> float:
    """Draw lines centered at x_center, from y_start downward. Returns final y after last line."""
    if color is not None:
        c.setFillColor(color)
    for line in lines:
        w = c.stringWidth(line, font_name, font_size)
        c.setFont(font_name, font_size)
        c.drawString(x_center - w / 2, y_start, line)
        y_start -= leading
    return y_start


def _draw_styled_centered_text_line(
    c: canvas.Canvas,
    text: str,
    x_center: float,
    y: float,
    font_name: str,
    font_size: float,
    fill_color: Any,
    stroke_color: Any,
    stroke_width: float,
) -> None:
    """
    Draw one centered line with layered shadow + stroke/fill.
    This approximates the preview CSS text style in printable PDFs.
    """
    text_width = c.stringWidth(text, font_name, font_size)
    x = x_center - text_width / 2

    # Soft glow layers (approximation of CSS blur shadows in preview)
    glow_offsets = [
        (0.0, 0.0, 0.22),
        (0.0, -0.9, 0.20),
        (0.0, 0.9, 0.18),
        (-1.0, 0.0, 0.12),
        (1.0, 0.0, 0.12),
    ]
    for dx, dy, alpha in glow_offsets:
        c.saveState()
        c.setFont(font_name, font_size)
        c.setFillColor(Color(1, 1, 1, alpha=alpha))
        c.drawString(x + dx, y + dy, text)
        c.restoreState()

    # Thicker drop shadow layers (requested to match preview)
    c.saveState()
    c.setFont(font_name, font_size)
    c.setFillColor(Color(15 / 255, 10 / 255, 59 / 255, alpha=0.50))
    c.drawString(x, y - max(1.4, font_size * 0.11), text)
    c.setFillColor(Color(15 / 255, 10 / 255, 59 / 255, alpha=0.36))
    c.drawString(x, y - max(2.0, font_size * 0.17), text)
    c.setFillColor(Color(15 / 255, 10 / 255, 59 / 255, alpha=0.28))
    c.drawString(x + 1.0, y - max(1.8, font_size * 0.14), text)
    c.drawString(x - 1.0, y - max(1.8, font_size * 0.14), text)
    c.restoreState()

    # Manual outline pass for broader compatibility and cleaner look.
    # Using multiple offset draws avoids reportlab text-render inconsistencies
    # where thick stroke can overpower the white fill.
    c.saveState()
    c.setFont(font_name, font_size)
    outline = max(1.4, stroke_width * 0.8)
    outer_outline = outline * 1.55
    stroke_r = getattr(stroke_color, "red", 28 / 255)
    stroke_g = getattr(stroke_color, "green", 89 / 255)
    stroke_b = getattr(stroke_color, "blue", 111 / 255)
    # Outer soft ring
    c.setFillColor(Color(stroke_r, stroke_g, stroke_b, alpha=0.42))
    outer_offsets = [
        (-outer_outline, 0.0), (outer_outline, 0.0), (0.0, -outer_outline), (0.0, outer_outline),
        (-outer_outline * 0.72, -outer_outline * 0.72), (-outer_outline * 0.72, outer_outline * 0.72),
        (outer_outline * 0.72, -outer_outline * 0.72), (outer_outline * 0.72, outer_outline * 0.72),
    ]
    for dx, dy in outer_offsets:
        c.drawString(x + dx, y + dy, text)
    # Inner solid outline
    c.setFillColor(stroke_color)
    outline_offsets = [
        (-outline, 0.0), (outline, 0.0), (0.0, -outline), (0.0, outline),
        (-outline * 0.75, -outline * 0.75), (-outline * 0.75, outline * 0.75),
        (outline * 0.75, -outline * 0.75), (outline * 0.75, outline * 0.75),
    ]
    for dx, dy in outline_offsets:
        c.drawString(x + dx, y + dy, text)
    c.restoreState()

    # Bright fill on top (preview has strong white interior)
    c.saveState()
    c.setFont(font_name, font_size)
    c.setFillColor(fill_color)
    c.drawString(x, y, text)
    c.restoreState()


def _draw_copyright_page_text(
    c: canvas.Canvas, width: float, height: float,
    child_name: str, character_name: str
) -> None:
    """Draw copyright page text overlay (same content and style as preview)."""
    _ensure_special_page_fonts()
    semibold_font = _SPECIAL_PAGE_FONT_STATE["semibold"]
    regular_font = _SPECIAL_PAGE_FONT_STATE["regular"]
    margin_x = width * 0.11
    max_w = width - 2 * margin_x
    cx = width / 2
    y = height * 0.64
    body_size = 16
    line_height = body_size * 1.34
    c.setFillColor(TEXT_WHITE_92)
    c.setFont(semibold_font, body_size)
    paras = [
        f"This one-of-a-kind adventure story was created just for {child_name}.",
        "Beyond these pages lies a magical world filled with wonder, mystery, and brave moments. Every scene unfolds a new chapter in the journey.",
        f"Follow {character_name} through lands of shadow and light, where courage is tested and imagination guides the way forward.",
        f"This story celebrates {child_name}'s creativity and courage. Turn the page and begin your adventure into the unknown—where magic awaits.",
    ]
    for p in paras:
        lines = _wrap_lines(c, p, max_w, semibold_font, body_size)
        for line in lines:
            w = c.stringWidth(line, semibold_font, body_size)
            c.drawString(cx - w / 2, y, line)
            y -= line_height
        y -= line_height * 0.62

    y = height * 0.12
    c.setFillColor(TEXT_WHITE_85)
    footer_size = 14
    c.setFont(regular_font, footer_size)
    footer = "© 2026 Drawtopia. All rights reserved.\nPublished by Drawtopia | drawtopia.ai"
    for line in footer.split("\n"):
        w = c.stringWidth(line, regular_font, footer_size)
        c.drawString(cx - w / 2, y, line)
        y -= footer_size * 1.5


def _draw_dedication_page_text(
    c: canvas.Canvas, width: float, height: float,
    child_name: str, body: str, signature: str
) -> None:
    """Draw dedication page text overlay (same content and style as preview)."""
    _ensure_special_page_fonts()
    medium_font = _SPECIAL_PAGE_FONT_STATE["medium"]
    regular_font = _SPECIAL_PAGE_FONT_STATE["regular"]
    max_w = width * 0.70
    cx = width / 2
    y = height * 0.62
    line_height = height * 0.052
    c.setFillColor(TEXT_WHITE)
    c.setFont(medium_font, 24)
    title = f"Dear {child_name}"
    w = c.stringWidth(title, medium_font, 24)
    c.drawString(cx - w / 2, y, title)
    y -= line_height * 1.42
    if body:
        lines = _wrap_lines(c, body, max_w, regular_font, 24)
        c.setFont(regular_font, 24)
        for line in lines:
            w = c.stringWidth(line, regular_font, 24)
            c.drawString(cx - w / 2, y, line)
            y -= line_height
    else:
        default = "In every tiny thing you do each day, never forget that you are loved enormously"
        lines = _wrap_lines(c, default, max_w, regular_font, 24)
        c.setFont(regular_font, 24)
        for line in lines:
            w = c.stringWidth(line, regular_font, 24)
            c.drawString(cx - w / 2, y, line)
            y -= line_height
    if signature:
        y -= line_height * 0.2
        c.setFont(regular_font, 22)
        w = c.stringWidth(signature, regular_font, 22)
        c.drawString(cx - w / 2, y, signature)


def _draw_last_words_page_text(
    c: canvas.Canvas, width: float, height: float, child_name: str
) -> None:
    """Draw last words page text overlay (same content and style as preview)."""
    _ensure_special_page_fonts()
    display_font = _SPECIAL_PAGE_FONT_STATE["display"]
    regular_font = _SPECIAL_PAGE_FONT_STATE["regular"]
    max_w = width * 0.70
    cx = width / 2
    y = height * 0.61
    line_height = height * 0.05
    c.setFillColor(TEXT_WHITE)
    title_font_size = 32
    c.setFont(display_font, title_font_size)
    title = "A Special Thank You"
    w = c.stringWidth(title, display_font, title_font_size)
    title_x = cx - w / 2
    # Draw multiple tightly-offset passes to increase perceived title thickness in PDF.
    c.drawString(title_x, y, title)
    c.drawString(title_x + 0.45, y, title)
    c.drawString(title_x + 0.9, y, title)
    c.drawString(title_x + 0.45, y + 0.25, title)
    y -= line_height * 1.60
    body = f"This magical adventure wouldn't exist without the incredible imagination of {child_name}. Thank you for sharing your creativity with the world!"
    lines = _wrap_lines(c, body, max_w, regular_font, 20)
    c.setFont(regular_font, 20)
    for line in lines:
        w = c.stringWidth(line, regular_font, 20)
        c.drawString(cx - w / 2, y, line)
        y -= line_height
    y -= line_height * 0.35
    tagline = "Every drawing tells a story. Yours told this one."
    c.setFont(regular_font, 20)
    w = c.stringWidth(tagline, regular_font, 20)
    c.drawString(cx - w / 2, y, tagline)


def _draw_last_admin_page_text(
    c: canvas.Canvas, width: float, height: float
) -> None:
    """Draw last admin page text overlay (same content and style as preview)."""
    _ensure_special_page_fonts()
    bold_font = _SPECIAL_PAGE_FONT_STATE["bold"]
    regular_font = _SPECIAL_PAGE_FONT_STATE["regular"]
    semibold_font = _SPECIAL_PAGE_FONT_STATE["semibold"]
    max_w = width * 0.76
    cx = width / 2
    line_height = height * 0.047

    logo_w = width * 0.30
    logo_h = logo_w * 0.223
    logo_x = cx - logo_w / 2
    logo_y = height * 0.885
    for logo_path in _BACK_COVER_LOGO_CANDIDATES:
        if logo_path.exists():
            try:
                c.drawImage(
                    ImageReader(str(logo_path)),
                    logo_x,
                    logo_y,
                    width=logo_w,
                    height=logo_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                break
            except Exception as e:
                logger.warning(f"Failed to draw last admin logo from {logo_path}: {e}")

    c.setFillColor(TEXT_WHITE)
    title = "Where Every Child Becomes a Storyteller"
    tagline = "Their imagination. Their characters. Their stories. Enhanced, not replaced."
    body = "At Drawtopia, we believe every child's drawing holds a story waiting to be told. We use the magic of AI to enhance - never replace - your child's authentic artwork, turning their imagination into adventures they'll treasure forever."

    title_lines = _wrap_lines(c, title, max_w, bold_font, 31)
    tagline_lines = _wrap_lines(c, tagline, max_w, regular_font, 21)
    body_lines = _wrap_lines(c, body, max_w, regular_font, 18)

    # Center only the text stack (title + tagline + body) vertically.
    # The logo and CTA button remain independently positioned.
    text_block_height = (
        len(title_lines) * line_height
        + line_height * 0.24
        + len(tagline_lines) * line_height
        + line_height * 0.24
        + len(body_lines) * line_height
    )
    y = (height + text_block_height) / 2

    c.setFont(bold_font, 31)
    for line in title_lines:
        w = c.stringWidth(line, bold_font, 31)
        c.drawString(cx - w / 2, y, line)
        y -= line_height

    y -= line_height * 0.24
    c.setFont(regular_font, 21)
    for line in tagline_lines:
        w = c.stringWidth(line, regular_font, 21)
        line_x = cx - w / 2
        c.drawString(line_x, y, line)
        c.saveState()
        c.setStrokeColor(TEXT_WHITE_92)
        c.setLineWidth(1.6)
        c.line(line_x, y - 2.4, line_x + w, y - 2.4)
        c.restoreState()
        y -= line_height

    y -= line_height * 0.24
    c.setFont(regular_font, 18)
    for line in body_lines:
        w = c.stringWidth(line, regular_font, 18)
        c.drawString(cx - w / 2, y, line)
        y -= line_height

    button_text = "Drawtopia.ai"
    button_font_size = 14.5
    c.setFont(semibold_font, button_font_size)
    text_w = c.stringWidth(button_text, semibold_font, button_font_size)
    icon_size = button_font_size + 2
    icon_gap = 7
    content_w = icon_size + icon_gap + text_w
    button_pad_x = 24
    button_pad_y = 10
    button_w = content_w + button_pad_x * 2
    button_h = button_font_size + button_pad_y * 2
    button_x = cx - button_w / 2
    button_y = height * 0.08
    c.saveState()
    c.setFillColor(HexColor("#438BFF"))
    c.setStrokeColor(HexColor("#438BFF"))
    c.roundRect(button_x, button_y, button_w, button_h, 14, fill=1, stroke=0)
    c.setFillColor(TEXT_WHITE)
    content_x = button_x + (button_w - content_w) / 2
    icon_x = content_x
    icon_y = button_y + (button_h - icon_size) / 2 - 0.2
    _draw_cta_link_icon(c, icon_x, icon_y, icon_size)
    text_x = content_x + icon_size + icon_gap
    text_y = button_y + button_pad_y + 1.15
    c.drawString(text_x, text_y, button_text)
    c.setStrokeColor(TEXT_WHITE)
    c.setLineWidth(1.2)
    c.line(text_x, text_y - 1.5, text_x + text_w, text_y - 1.5)
    c.restoreState()
    c.linkURL(
        "https://app.drawtopia.ai",
        (button_x, button_y, button_x + button_w, button_y + button_h),
        relative=0,
        thickness=0,
    )


def _draw_vertical_gradient_overlay(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    rgb: tuple,
    alpha_start: float,
    alpha_end: float,
    blur_radius: float = 4.0,
) -> None:
    """Draw a smooth vertical alpha gradient using an RGBA image overlay."""
    if height <= 0 or width <= 0:
        return
    r, g, b = rgb
    width_px = max(8, int(round(width * 2.0)))
    height_px = max(8, int(round(height * 2.0)))
    alpha_strip = PILImage.new("L", (1, height_px))
    alpha_strip.putdata([
        int(max(0, min(255, round((alpha_start + (alpha_end - alpha_start) * (i / max(1, height_px - 1))) * 255))))
        for i in range(height_px)
    ])
    alpha_img = alpha_strip.resize((width_px, height_px), PILImage.Resampling.BICUBIC)
    if blur_radius > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    overlay = PILImage.new("RGBA", (width_px, height_px), (
        int(max(0, min(255, round(r * 255)))),
        int(max(0, min(255, round(g * 255)))),
        int(max(0, min(255, round(b * 255)))),
        0,
    ))
    overlay.putalpha(alpha_img)
    c.drawImage(ImageReader(overlay), x, y, width=width, height=height, mask="auto")


def _draw_horizontal_gradient_overlay(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    rgb: tuple,
    alpha_start: float,
    alpha_end: float,
    blur_radius: float = 4.0,
) -> None:
    """Draw a smooth horizontal alpha gradient using an RGBA image overlay."""
    if height <= 0 or width <= 0:
        return
    r, g, b = rgb
    width_px = max(8, int(round(width * 2.0)))
    height_px = max(8, int(round(height * 2.0)))
    alpha_strip = PILImage.new("L", (width_px, 1))
    alpha_strip.putdata([
        int(max(0, min(255, round((alpha_start + (alpha_end - alpha_start) * (i / max(1, width_px - 1))) * 255))))
        for i in range(width_px)
    ])
    alpha_img = alpha_strip.resize((width_px, height_px), PILImage.Resampling.BICUBIC)
    if blur_radius > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    overlay = PILImage.new("RGBA", (width_px, height_px), (
        int(max(0, min(255, round(r * 255)))),
        int(max(0, min(255, round(g * 255)))),
        int(max(0, min(255, round(b * 255)))),
        0,
    ))
    overlay.putalpha(alpha_img)
    c.drawImage(ImageReader(overlay), x, y, width=width, height=height, mask="auto")


def _draw_back_cover_blur_layers(c: canvas.Canvas, width: float, height: float) -> None:
    """Match preview back cover top and bottom blue blur layers."""
    blur_color = (91 / 255, 153 / 255, 175 / 255)  # #5B99AF
    blur_h = height * 0.40
    # Top blur: strongest at top edge, fades toward center.
    _draw_vertical_gradient_overlay(
        c=c,
        x=0,
        y=height - blur_h,
        width=width,
        height=blur_h,
        rgb=blur_color,
        alpha_start=0.58,
        alpha_end=0.0,
    )
    # Bottom blur: strongest at bottom edge, fades upward.
    _draw_vertical_gradient_overlay(
        c=c,
        x=0,
        y=0,
        width=width,
        height=blur_h,
        rgb=blur_color,
        alpha_start=0.0,
        alpha_end=0.58,
    )


def _draw_last_admin_left_blur_layer(c: canvas.Canvas, width: float, height: float) -> None:
    """Match preview last-admin left-side blue gradient blur layer."""
    blur_color = (68 / 255, 120 / 255, 159 / 255)  # #44789F
    blur_w = width * 0.50
    _draw_horizontal_gradient_overlay(
        c=c,
        x=0,
        y=0,
        width=blur_w,
        height=height,
        rgb=blur_color,
        alpha_start=0.56,
        alpha_end=0.0,
    )


def _draw_right_side_blur_layer(c: canvas.Canvas, width: float, height: float) -> None:
    """Match preview right-side blue gradient blur layer (copyright page)."""
    blur_color = (68 / 255, 120 / 255, 159 / 255)  # #44789F
    blur_w = width * 0.50
    _draw_horizontal_gradient_overlay(
        c=c,
        x=width - blur_w,
        y=0,
        width=blur_w,
        height=height,
        rgb=blur_color,
        alpha_start=0.0,
        alpha_end=0.56,
    )


def _draw_center_blur_layer(c: canvas.Canvas, width: float, height: float) -> None:
    """Match preview center blur decoration on copyright and last-words pages."""
    if width <= 0 or height <= 0:
        return
    blur_color = (68 / 255, 120 / 255, 158 / 255)  # #44789E
    layer_w = width * 1.0
    layer_h = height * 0.5
    scale = 2.0
    page_w_px = max(32, int(round(width * scale)))
    page_h_px = max(32, int(round(height * scale)))
    overlay = PILImage.new("RGBA", (page_w_px, page_h_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    blur_w_px = max(16, int(round(layer_w * scale)))
    blur_h_px = max(16, int(round(layer_h * scale)))
    x0 = int(round((page_w_px - blur_w_px) / 2))
    y0 = int(round((page_h_px - blur_h_px) / 2))
    x1 = x0 + blur_w_px
    y1 = y0 + blur_h_px
    draw.ellipse(
        [x0, y0, x1, y1],
        fill=(
            int(round(blur_color[0] * 255)),
            int(round(blur_color[1] * 255)),
            int(round(blur_color[2] * 255)),
            168,
        ),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=100))
    c.drawImage(ImageReader(overlay), 0, 0, width=width, height=height, mask="auto")


def _draw_story_main_text_blur_layer(
    c: canvas.Canvas,
    width: float,
    height: float,
    is_first_story_page: bool,
) -> None:
    """Add full-circle blue blur under story text (bottom-right on page 1, top-right on pages 2-5)."""
    if width <= 0 or height <= 0:
        return

    blur_color = (59 / 255, 119 / 255, 139 / 255)  # #3B778B
    layer_w = 420.0
    layer_h = 420.0

    # Keep this buffer moderate for speed while preserving smooth blur.
    scale = max(1.0, min(1.8, 1200.0 / max(width, height)))
    page_w_px = max(64, int(round(width * scale)))
    page_h_px = max(64, int(round(height * scale)))
    overlay = PILImage.new("RGBA", (page_w_px, page_h_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Keep the full circle fully inside the page (no corner clipping).
    # Position remains in the same right-side text zones:
    # - Page 1: bottom-right
    # - Pages 2-5: top-right
    circle_r = layer_w / 2.0
    cx = width * 0.78
    # NOTE: This ellipse is drawn on a PIL overlay (top-left origin), not directly on ReportLab.
    # So larger Y means lower (toward bottom) on the page.
    cy = height * 0.76 if is_first_story_page else height * 0.24
    # Clamp center so the full circle stays visible even on smaller page sizes.
    cx = min(max(circle_r, cx), max(circle_r, width - circle_r))
    cy = min(max(circle_r, cy), max(circle_r, height - circle_r))
    x0 = cx - circle_r
    y0 = cy - circle_r
    x1 = x0 + layer_w
    y1 = y0 + layer_h
    draw.ellipse(
        [
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        ],
        fill=(
            int(round(blur_color[0] * 255)),
            int(round(blur_color[1] * 255)),
            int(round(blur_color[2] * 255)),
            78,
        ),
    )

    blur_radius = max(8, int(round(min(180 * scale, 84))))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    c.drawImage(ImageReader(overlay), 0, 0, width=width, height=height, mask="auto")


def _draw_story_main_page_text(
    c: canvas.Canvas,
    width: float,
    height: float,
    text: str,
    is_first_story_page: bool,
) -> None:
    """Draw story text in right-side area (page 1 bottom-right, pages 2-5 top-right)."""
    clean_text = (text or "").strip()
    if not clean_text:
        return

    _ensure_special_page_fonts()
    text_font = _SPECIAL_PAGE_FONT_STATE["bold"]
    text_color = HexColor("#FDDAC6")
    font_size = max(10.0, min(16.0, width * 0.026))
    line_height = font_size * 1.25
    # Constrain text to a specific right-side box so placement matches preview intent.
    right_margin = width * 0.04
    text_box_width = width * 0.56
    text_box_left = width - right_margin - text_box_width
    max_width = text_box_width * 0.9
    max_lines = 6

    lines = _wrap_lines(c, clean_text, max_width, text_font, int(round(font_size)))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip() + "..."
    if not lines:
        return

    x_center = text_box_left + (text_box_width / 2.0)
    if is_first_story_page:
        # Page 1 -> bottom-right area.
        bottom_padding = height * 0.11
        y_start = bottom_padding + (len(lines) - 1) * line_height
    else:
        # Pages 2-5 -> top-right area.
        top_padding = height * 0.12
        y_start = height - top_padding - font_size

    c.setFillColor(text_color)
    c.setFont(text_font, font_size)
    y = y_start
    for line in lines:
        w = c.stringWidth(line, text_font, font_size)
        c.drawString(x_center - w / 2, y, line)
        y -= line_height


# ISBN barcode pattern from preview SVG (viewBox 0 0 120 70): black bars as (x, width) in SVG units
_BARCODE_BLACK_BARS = [
    (2, 2), (7, 2), (12, 1), (17, 1), (22, 1), (27, 1), (32, 2), (37, 2),
    (42, 1), (47, 1), (52, 1), (57, 1), (62, 2), (67, 2), (72, 1), (77, 1),
    (82, 1), (87, 2), (92, 2), (97, 1), (102, 1), (107, 2),
]


def _draw_back_cover_barcode(
    c: canvas.Canvas, width: float, height: float
) -> None:
    """Draw ISBN barcode block at bottom-right of back cover (same as preview)."""
    _ensure_special_page_fonts()
    regular_font = _SPECIAL_PAGE_FONT_STATE["regular"]
    # Position: bottom-right, matching preview (bottom 1.5rem, right 1.75rem)
    margin_right = width * 0.055
    margin_bottom = height * 0.03
    # Barcode block size (preview: 110px x 56px for the svg, with wrap padding)
    box_w = width * 0.19  # ~113pt on A4
    box_h = height * 0.068   # ~57pt on A4
    pad = 3  # padding inside white wrap
    # SVG viewBox 120x70 -> scale to (box_w - 2*pad) x (box_h - 2*pad - text_space)
    bar_area_h = box_h - 2 * pad - 14  # space for ISBN text below bars
    bar_area_w = box_w - 2 * pad
    scale_x = bar_area_w / 120.0
    scale_y = bar_area_h / 45.0  # bar height in SVG is 45
    left = width - margin_right - box_w
    bottom = margin_bottom
    # White background wrap (rounded rect as simple rect)
    c.setFillColor(white)
    c.setStrokeColor(HexColor("#cccccc"))
    c.setLineWidth(0.5)
    c.rect(left, bottom, box_w, box_h, fill=1, stroke=1)
    # Black bars (y=5 in SVG, height=45; we draw from bottom + text space)
    bar_y = bottom + pad + 12  # 12pt for text line below bars
    c.setFillColor(black)
    c.setStrokeColor(black)
    for x_svg, w_svg in _BARCODE_BLACK_BARS:
        x_pt = left + pad + x_svg * scale_x
        w_pt = max(0.5, w_svg * scale_x)
        h_pt = 45 * scale_y
        c.rect(x_pt, bar_y, w_pt, h_pt, fill=1, stroke=0)
    # ISBN text below bars (preview: "1 234567 890128>")
    isbn_text = "1 234567 890128>"
    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    tw = c.stringWidth(isbn_text, "Helvetica", 8)
    text_x = left + box_w / 2 - tw / 2
    text_y = bottom + pad + 2
    c.drawString(text_x, text_y, isbn_text)
    # "ISBN placeholder" above the barcode block (preview: back-cover-isbn)
    isbn_label = "ISBN placeholder"
    c.setFillColor(TEXT_WHITE)
    c.setFont(regular_font, 10.5)
    lw = c.stringWidth(isbn_label, regular_font, 10.5)
    c.drawString(width - margin_right - lw, bottom + box_h + 4, isbn_label)
    # Age label below barcode
    age_text = "[Age 6-12]"
    c.setFillColor(TEXT_WHITE)
    c.setFont(regular_font, 10.5)
    aw = c.stringWidth(age_text, regular_font, 10.5)
    c.drawString(width - margin_right - aw, bottom - 12, age_text)


def _draw_back_cover_text(
    c: canvas.Canvas, width: float, height: float
) -> None:
    """Draw back cover text overlay (same content and style as preview)."""
    _ensure_special_page_fonts()
    regular_font = _SPECIAL_PAGE_FONT_STATE["regular"]
    bold_font = _SPECIAL_PAGE_FONT_STATE["bold"]
    italic_font = _SPECIAL_PAGE_FONT_STATE["italic"]
    semibold_font = _SPECIAL_PAGE_FONT_STATE["semibold"]
    cx = width / 2
    margin_x = width * 0.06
    max_w = width - 2 * margin_x
    y = height * 0.79
    title_font_size = 35
    title_line_gap = height * 0.062
    stroke_width = max(2.0, title_font_size * 0.13)
    title_stroke = HexColor("#1C596F")
    title_lines = ["Drawtopia Makes", "Every Child a", "Storyteller"]
    for line in title_lines:
        _draw_styled_centered_text_line(
            c=c,
            text=line,
            x_center=cx,
            y=y,
            font_name=bold_font,
            font_size=title_font_size,
            fill_color=TEXT_WHITE,
            stroke_color=title_stroke,
            stroke_width=stroke_width,
        )
        y -= title_line_gap

    y -= height * 0.012
    desc_font = 16
    c.setFont(regular_font, desc_font)
    c.setFillColor(TEXT_WHITE)
    desc = "At Drawtopia, we believe every child's drawing holds a story waiting to be told. We use the magic of AI to enhance - never replace - your child's authentic artwork, turning their imagination into adventures they'll treasure forever."
    lines = _wrap_lines(c, desc, max_w * 0.78, regular_font, desc_font)
    line_height = desc_font * 1.52
    for line in lines:
        w = c.stringWidth(line, regular_font, desc_font)
        c.drawString(cx - w / 2, y, line)
        y -= line_height

    # Bottom-left block (logo + tagline + website), matching preview placement.
    left_x = width * 0.055
    bottom_margin = height * 0.035
    logo_w = width * 0.165
    logo_h = logo_w * 0.223  # Keep source aspect ratio close to white-logo.png
    logo_drawn = False
    for logo_path in _BACK_COVER_LOGO_CANDIDATES:
        if logo_path.exists():
            try:
                c.drawImage(
                    ImageReader(str(logo_path)),
                    left_x,
                    bottom_margin + 54,
                    width=logo_w,
                    height=logo_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                logo_drawn = True
                break
            except Exception as e:
                logger.warning(f"Failed to draw back cover logo from {logo_path}: {e}")

    c.setFillColor(TEXT_WHITE_92)
    c.setFont(italic_font, 10.8)
    c.drawString(left_x, bottom_margin + 36, "Their imagination. Their characters. Their")
    c.drawString(left_x, bottom_margin + 22, "stories. Enhanced, not replaced.")
    c.setFillColor(TEXT_WHITE)
    c.setFont(semibold_font, 11.5)
    c.drawString(left_x, bottom_margin + 8, "drawtopia.ai")

    if not logo_drawn:
        logger.warning("Back cover logo not found; rendered text-only bottom-left block")

    # ISBN label + barcode block at bottom-right (same as preview)
    _draw_back_cover_barcode(c, width, height)


def _normalize_scene_urls(scene_urls) -> List[str]:
    """Accept scene_urls as list or JSON/string representation of list."""
    if scene_urls is None:
        return []
    if isinstance(scene_urls, list):
        return [str(u) for u in scene_urls if u]
    if isinstance(scene_urls, str):
        try:
            import json
            parsed = json.loads(scene_urls)
            return [str(u) for u in (parsed if isinstance(parsed, list) else [parsed]) if u]
        except Exception:
            pass
        try:
            import ast
            parsed = ast.literal_eval(scene_urls)
            return [str(u) for u in (parsed if isinstance(parsed, list) else [parsed]) if u]
        except Exception:
            pass
    return []


def create_book_pdf_with_cover(
    story_title: str,
    story_cover_url: Optional[str],
    scene_urls,
    output_buffer: BytesIO,
    copyright_image_url: Optional[str] = None,
    dedication_image_url: Optional[str] = None,
    last_word_page_image_url: Optional[str] = None,
    last_admin_page_image_url: Optional[str] = None,
    back_cover_image_url: Optional[str] = None,
    copyright_child_name: Optional[str] = None,
    copyright_character_name: Optional[str] = None,
    dedication_body: Optional[str] = None,
    dedication_signature: Optional[str] = None,
    story_page_texts: Optional[List[str]] = None,
) -> bool:
    """
    Create a full book PDF. Each image covers the whole page (no margins, scale-to-fill).
    Story page images are split into left/right halves; each half is one PDF page.
    Copyright, dedication, last words, last admin, and back cover pages show image + text overlay (same style as preview).
    Page order: cover, copyright, dedication, story pages (left+right each), last words, last admin, back cover.
    """
    child_name = (copyright_child_name or "[CHILD_NAME]").strip() or "[CHILD_NAME]"
    character_name = (copyright_character_name or "[CHARACTER_NAME]").strip() or "[CHARACTER_NAME]"
    try:
        start_time = time.time()
        scene_list = _normalize_scene_urls(scene_urls)
        logger.info(
            f"Creating book PDF: {story_title} — cover, copyright, dedication, "
            f"{len(scene_list)} story images (left+right each), last words, last admin, back cover"
        )

        c = canvas.Canvas(output_buffer, pagesize=A4)
        width, height = A4
        page_count = 0
        story_text_list = story_page_texts or []

        # 1. Cover (one image = one full page, image covers whole page)
        if story_cover_url:
            logger.info("Adding cover page...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, story_cover_url, width, height, "cover"):
                page_count += 1
            _draw_cover_bottom_logo(c, width, height)
            c.showPage()

        # 2. Copyright page (image + text overlay, same style as preview)
        if copyright_image_url:
            logger.info("Adding copyright page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, copyright_image_url, width, height, "copyright"):
                page_count += 1
            _draw_center_blur_layer(c, width, height)
            _draw_right_side_blur_layer(c, width, height)
            _draw_copyright_page_text(c, width, height, child_name, character_name)
            c.showPage()

        # 3. Dedication page (image + text overlay, same style as preview)
        has_dedication_text = bool((dedication_body or "").strip() or (dedication_signature or "").strip())
        dedication_image_str = (dedication_image_url or "").strip()
        has_dedication_image = bool(
            dedication_image_str
            and dedication_image_str.lower() not in {"null", "none", "undefined"}
        )
        if has_dedication_image or has_dedication_text:
            logger.info("Adding dedication page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            image_drawn = False
            if has_dedication_image:
                image_drawn = _draw_full_page_image(c, dedication_image_str, width, height, "dedication")
            if not image_drawn:
                # Prevent blank white page when dedication image URL is invalid/missing.
                c.setFillColor(HexColor("#1A3540"))
                c.rect(0, 0, width, height, fill=1, stroke=0)
            _draw_last_admin_left_blur_layer(c, width, height)
            _draw_dedication_page_text(c, width, height, child_name, dedication_body or "", dedication_signature or "")
            c.showPage()
            page_count += 1

        # 4. Story pages: each scene image → left half page + right half page (each covers full PDF page)
        for i, scene_url in enumerate(scene_list, 1):
            if not scene_url:
                continue
            logger.info(f"Adding story image {i}/{len(scene_list)} as left + right pages...")
            image = _load_image_rgb(url=scene_url)
            if not image:
                logger.warning(f"Failed to load story image {i} from {scene_url}, skipping")
                continue
            left_half, right_half = _split_image_left_right(image)
            # Left half = one page
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_image_cover_page(c, left_half, width, height):
                page_count += 1
            c.showPage()
            # Right half = one page
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_image_cover_page(c, right_half, width, height):
                page_count += 1
            page_text = story_text_list[i - 1] if i - 1 < len(story_text_list) else ""
            if page_text.strip():
                _draw_story_main_text_blur_layer(
                    c,
                    width,
                    height,
                    is_first_story_page=(i == 1),
                )
                _draw_story_main_page_text(
                    c,
                    width,
                    height,
                    page_text,
                    is_first_story_page=(i == 1),
                )
            c.showPage()

        # 5. Last words page (image + text overlay, same style as preview)
        if last_word_page_image_url:
            logger.info("Adding last words page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, last_word_page_image_url, width, height, "last words"):
                page_count += 1
            _draw_center_blur_layer(c, width, height)
            _draw_last_words_page_text(c, width, height, child_name)
            c.showPage()

        # 6. Last admin page (image + text overlay, same style as preview)
        if last_admin_page_image_url:
            logger.info("Adding last admin page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, last_admin_page_image_url, width, height, "last admin"):
                page_count += 1
            _draw_last_admin_left_blur_layer(c, width, height)
            _draw_last_admin_page_text(c, width, height)
            c.showPage()

        # 7. Back cover (image + text overlay, same style as preview)
        if back_cover_image_url:
            logger.info("Adding back cover page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, back_cover_image_url, width, height, "back cover"):
                page_count += 1
            _draw_back_cover_blur_layers(c, width, height)
            _draw_back_cover_text(c, width, height)
            c.showPage()

        c.save()
        elapsed = time.time() - start_time
        logger.info(f"✅ Book PDF created successfully with {page_count} pages in {elapsed:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error creating book PDF: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return False


def generate_pdf(
    pdf_type: str,  # "interactive_search" or "story_adventure" or "simple_scenes"
    character_name: str,
    story_title: str,
    character_image_url: Optional[str] = None,
    scene_urls: Optional[List[str]] = None,
    story_pages: Optional[List[Dict[str, Any]]] = None,
    audio_urls: Optional[List[Optional[str]]] = None
) -> Optional[bytes]:
    """
    Main function to generate PDF based on type
    
    Returns:
        PDF bytes if successful, None otherwise
    """
    try:
        output_buffer = BytesIO()
        
        if pdf_type == "simple_scenes":
            if not scene_urls:
                logger.error("scene_urls required for simple_scenes PDF")
                return None
            
            success = create_simple_scene_pdf(
                story_title=story_title,
                scene_urls=scene_urls,
                output_buffer=output_buffer
            )
        elif pdf_type == "interactive_search":
            if not scene_urls:
                logger.error("scene_urls required for interactive_search PDF")
                return None
            
            success = create_interactive_search_pdf(
                character_name=character_name,
                story_title=story_title,
                character_image_url=character_image_url,
                scene_urls=scene_urls,
                output_buffer=output_buffer
            )
        elif pdf_type == "story_adventure":
            if not story_pages:
                logger.error("story_pages required for story_adventure PDF")
                return None
            
            success = create_story_adventure_pdf(
                character_name=character_name,
                story_title=story_title,
                character_image_url=character_image_url,
                story_pages=story_pages,
                audio_urls=audio_urls,
                output_buffer=output_buffer
            )
        else:
            logger.error(f"Unknown PDF type: {pdf_type}")
            return None
        
        if success:
            pdf_bytes = output_buffer.getvalue()
            logger.info(f"PDF generated: {len(pdf_bytes)} bytes")
            return pdf_bytes
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in generate_pdf: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None

