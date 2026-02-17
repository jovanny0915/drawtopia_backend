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
from PIL import Image as PILImage
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
# Preview uses Quicksand; PDF uses Helvetica for compatibility. Proportional sizes for A4.
TEXT_WHITE = HexColor("#FFFFFF")
TEXT_WHITE_92 = HexColor("#E6E6E6")  # rgba(255,255,255,0.92)
TEXT_WHITE_85 = HexColor("#D9D9D9")  # rgba(255,255,255,0.85)

_BACK_COVER_LOGO_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "drawtopia_frontend" / "src" / "assets" / "white-logo.png",
    Path(__file__).resolve().parent / "assets" / "white-logo.png",
]


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
        (0.0, 0.0, 0.18),
        (0.0, -0.6, 0.20),
        (0.0, 0.6, 0.16),
        (-0.8, 0.0, 0.11),
        (0.8, 0.0, 0.11),
    ]
    for dx, dy, alpha in glow_offsets:
        c.saveState()
        c.setFont(font_name, font_size)
        c.setFillColor(Color(1, 1, 1, alpha=alpha))
        c.drawString(x + dx, y + dy, text)
        c.restoreState()

    # Drop shadow layer
    c.saveState()
    c.setFont(font_name, font_size)
    c.setFillColor(Color(15 / 255, 10 / 255, 59 / 255, alpha=0.5))
    c.drawString(x, y - max(1.0, font_size * 0.09), text)
    c.restoreState()

    # Manual outline pass for broader compatibility and cleaner look.
    # Using multiple offset draws avoids reportlab text-render inconsistencies
    # where thick stroke can overpower the white fill.
    c.saveState()
    c.setFont(font_name, font_size)
    c.setFillColor(stroke_color)
    outline = max(1.2, stroke_width * 0.72)
    outline_offsets = [
        (-outline, 0.0), (outline, 0.0), (0.0, -outline), (0.0, outline),
        (-outline * 0.72, -outline * 0.72), (-outline * 0.72, outline * 0.72),
        (outline * 0.72, -outline * 0.72), (outline * 0.72, outline * 0.72),
        (-outline * 1.35, 0.0), (outline * 1.35, 0.0), (0.0, -outline * 1.35), (0.0, outline * 1.35),
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
    margin_x = width * 0.08
    max_w = width - 2 * margin_x
    cx = width / 2
    y = height * 0.78
    line_height = height * 0.028
    c.setFillColor(TEXT_WHITE_92)
    c.setFont("Helvetica-Bold", 12)
    paras = [
        f"This one-of-a-kind adventure story was created just for {child_name}.",
        "Beyond these pages lies a magical world filled with wonder, mystery, and brave moments. Every scene unfolds a new chapter in the journey.",
        f"Follow {character_name} through lands of shadow and light, where courage is tested and imagination guides the way forward.",
        f"This story celebrates {child_name}'s creativity and courage. Turn the page and begin your adventure into the unknown—where magic awaits.",
    ]
    for p in paras:
        lines = _wrap_lines(c, p, max_w, "Helvetica-Bold", 12)
        for line in lines:
            w = c.stringWidth(line, "Helvetica-Bold", 12)
            c.drawString(cx - w / 2, y, line)
            y -= line_height
        y -= line_height * 0.5
    y -= line_height * 2
    c.setFillColor(TEXT_WHITE_85)
    c.setFont("Helvetica", 10)
    footer = "© 2026 Drawtopia. All rights reserved.\nPublished by Drawtopia | drawtopia.ai"
    for line in footer.split("\n"):
        w = c.stringWidth(line, "Helvetica", 10)
        c.drawString(cx - w / 2, y, line)
        y -= line_height


def _draw_dedication_page_text(
    c: canvas.Canvas, width: float, height: float,
    child_name: str, body: str, signature: str
) -> None:
    """Draw dedication page text overlay (same content and style as preview)."""
    max_w = width * 0.85
    cx = width / 2
    y = height * 0.72
    line_height = height * 0.026
    c.setFillColor(TEXT_WHITE)
    c.setFont("Helvetica", 20)
    title = f"Dear {child_name}"
    w = c.stringWidth(title, "Helvetica", 20)
    c.drawString(cx - w / 2, y, title)
    y -= line_height * 2.5
    if body:
        lines = _wrap_lines(c, body, max_w, "Helvetica", 14)
        c.setFont("Helvetica", 14)
        for line in lines:
            w = c.stringWidth(line, "Helvetica", 14)
            c.drawString(cx - w / 2, y, line)
            y -= line_height
    else:
        default = "In every tiny thing you do each day, never forget that you are loved enormously"
        lines = _wrap_lines(c, default, max_w, "Helvetica", 14)
        c.setFont("Helvetica", 14)
        for line in lines:
            w = c.stringWidth(line, "Helvetica", 14)
            c.drawString(cx - w / 2, y, line)
            y -= line_height
    if signature:
        y -= line_height
        c.setFont("Helvetica", 12)
        w = c.stringWidth(signature, "Helvetica", 12)
        c.drawString(cx - w / 2, y, signature)


def _draw_last_words_page_text(
    c: canvas.Canvas, width: float, height: float, child_name: str
) -> None:
    """Draw last words page text overlay (same content and style as preview)."""
    max_w = width * 0.88
    cx = width / 2
    y = height * 0.68
    line_height = height * 0.026
    c.setFillColor(TEXT_WHITE)
    c.setFont("Helvetica-Bold", 20)
    title = "A Special Thank You"
    w = c.stringWidth(title, "Helvetica-Bold", 20)
    c.drawString(cx - w / 2, y, title)
    y -= line_height * 2
    body = f"This magical adventure wouldn't exist without the incredible imagination of {child_name}. Thank you for sharing your creativity with the world!"
    lines = _wrap_lines(c, body, max_w, "Helvetica", 12)
    c.setFont("Helvetica", 12)
    for line in lines:
        w = c.stringWidth(line, "Helvetica", 12)
        c.drawString(cx - w / 2, y, line)
        y -= line_height
    y -= line_height
    tagline = "Every drawing tells a story. Yours told this one."
    w = c.stringWidth(tagline, "Helvetica", 11)
    c.setFont("Helvetica", 11)
    c.drawString(cx - w / 2, y, tagline)


def _draw_last_admin_page_text(
    c: canvas.Canvas, width: float, height: float
) -> None:
    """Draw last admin page text overlay (same content and style as preview)."""
    max_w = width * 0.88
    cx = width / 2
    y = height * 0.72
    line_height = height * 0.024
    c.setFillColor(TEXT_WHITE)
    c.setFont("Helvetica-Bold", 18)
    title = "Where Every Child Becomes a Storyteller"
    lines = _wrap_lines(c, title, max_w, "Helvetica-Bold", 18)
    for line in lines:
        w = c.stringWidth(line, "Helvetica-Bold", 18)
        c.drawString(cx - w / 2, y, line)
        y -= line_height
    y -= line_height
    c.setFont("Helvetica", 11)
    tagline = "Their imagination. Their characters. Their stories. Enhanced, not replaced."
    lines = _wrap_lines(c, tagline, max_w, "Helvetica", 11)
    for line in lines:
        w = c.stringWidth(line, "Helvetica", 11)
        c.drawString(cx - w / 2, y, line)
        y -= line_height
    y -= line_height
    body = "At Drawtopia, we believe every child's drawing holds a story waiting to be told. We use the magic of AI to enhance - never replace - your child's authentic artwork, turning their imagination into adventures they'll treasure forever."
    lines = _wrap_lines(c, body, max_w, "Helvetica", 11)
    for line in lines:
        w = c.stringWidth(line, "Helvetica", 11)
        c.drawString(cx - w / 2, y, line)
        y -= line_height
    y = height * 0.12
    c.setFont("Helvetica-Bold", 12)
    w = c.stringWidth("Drawtopia.ai", "Helvetica-Bold", 12)
    c.drawString(cx - w / 2, y, "Drawtopia.ai")


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
    c.setFont("Helvetica", 10)
    lw = c.stringWidth(isbn_label, "Helvetica", 10)
    c.drawString(width - margin_right - lw, bottom + box_h + 4, isbn_label)
    # Age label below barcode
    age_text = "[Age 6-12]"
    c.setFillColor(TEXT_WHITE)
    c.setFont("Helvetica", 10)
    aw = c.stringWidth(age_text, "Helvetica", 10)
    c.drawString(width - margin_right - aw, bottom - 12, age_text)


def _draw_back_cover_text(
    c: canvas.Canvas, width: float, height: float
) -> None:
    """Draw back cover text overlay (same content and style as preview)."""
    cx = width / 2
    margin_x = width * 0.06
    max_w = width - 2 * margin_x
    y = height * 0.81
    title_font_size = 35
    title_line_gap = height * 0.06
    stroke_width = max(2.0, title_font_size * 0.13)
    title_stroke = HexColor("#1C596F")
    title_lines = ["Drawtopia Makes", "Every Child a", "Storyteller"]
    for line in title_lines:
        _draw_styled_centered_text_line(
            c=c,
            text=line,
            x_center=cx,
            y=y,
            font_name="Helvetica-Bold",
            font_size=title_font_size,
            fill_color=TEXT_WHITE,
            stroke_color=title_stroke,
            stroke_width=stroke_width,
        )
        y -= title_line_gap
    y -= height * 0.01
    desc_font = 14
    c.setFont("Helvetica", desc_font)
    c.setFillColor(TEXT_WHITE)
    desc = "At Drawtopia, we believe every child's drawing holds a story waiting to be told. We use the magic of AI to enhance - never replace - your child's authentic artwork, turning their imagination into adventures they'll treasure forever."
    lines = _wrap_lines(c, desc, max_w * 0.92, "Helvetica", desc_font)
    line_height = desc_font * 1.5
    for line in lines:
        w = c.stringWidth(line, "Helvetica", desc_font)
        c.drawString(cx - w / 2, y, line)
        y -= line_height

    # Bottom-left block (logo + tagline + website), matching preview placement.
    left_x = width * 0.055
    bottom_margin = height * 0.03
    logo_w = width * 0.23
    logo_h = logo_w * 0.223  # Keep source aspect ratio close to white-logo.png
    logo_drawn = False
    for logo_path in _BACK_COVER_LOGO_CANDIDATES:
        if logo_path.exists():
            try:
                c.drawImage(
                    ImageReader(str(logo_path)),
                    left_x,
                    bottom_margin + 52,
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
    c.setFont("Helvetica-Oblique", 10.5)
    c.drawString(left_x, bottom_margin + 38, "Their imagination. Their characters.")
    c.drawString(left_x, bottom_margin + 24, "Their stories. Enhanced, not replaced.")
    c.setFillColor(TEXT_WHITE)
    c.setFont("Helvetica-Bold", 11)
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

        # 1. Cover (one image = one full page, image covers whole page)
        if story_cover_url:
            logger.info("Adding cover page...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, story_cover_url, width, height, "cover"):
                page_count += 1
            c.showPage()

        # 2. Copyright page (image + text overlay, same style as preview)
        if copyright_image_url:
            logger.info("Adding copyright page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, copyright_image_url, width, height, "copyright"):
                page_count += 1
            _draw_copyright_page_text(c, width, height, child_name, character_name)
            c.showPage()

        # 3. Dedication page (image + text overlay, same style as preview)
        if dedication_image_url:
            logger.info("Adding dedication page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, dedication_image_url, width, height, "dedication"):
                page_count += 1
            _draw_dedication_page_text(c, width, height, child_name, dedication_body or "", dedication_signature or "")
            c.showPage()

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
            c.showPage()

        # 5. Last words page (image + text overlay, same style as preview)
        if last_word_page_image_url:
            logger.info("Adding last words page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, last_word_page_image_url, width, height, "last words"):
                page_count += 1
            _draw_last_words_page_text(c, width, height, child_name)
            c.showPage()

        # 6. Last admin page (image + text overlay, same style as preview)
        if last_admin_page_image_url:
            logger.info("Adding last admin page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, last_admin_page_image_url, width, height, "last admin"):
                page_count += 1
            _draw_last_admin_page_text(c, width, height)
            c.showPage()

        # 7. Back cover (image + text overlay, same style as preview)
        if back_cover_image_url:
            logger.info("Adding back cover page (image + text)...")
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1, stroke=0)
            if _draw_full_page_image(c, back_cover_image_url, width, height, "back cover"):
                page_count += 1
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

