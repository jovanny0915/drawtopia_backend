"""
PDF Generation System
Generates print-ready PDFs for both Interactive Search and Story Adventure formats
"""

import logging
import time
from io import BytesIO
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black
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


def create_book_pdf_with_cover(
    story_title: str,
    story_cover_url: Optional[str],
    scene_urls: List[str],
    output_buffer: BytesIO
) -> bool:
    """
    Create a 6-page PDF with cover and scene images
    
    Format:
    - Page 1: story_cover image (full page)
    - Pages 2-6: scene_images (up to 5 images, full page each)
    - A4 pagesize
    - 1 inch margins
    - preserveAspectRatio=True
    """
    try:
        start_time = time.time()
        logger.info(f"Creating book PDF: {story_title} with cover and {len(scene_urls)} scenes")
        
        # Create PDF canvas with A4 pagesize
        c = canvas.Canvas(output_buffer, pagesize=A4)
        width, height = A4
        
        # Use 1 inch margins as specified
        margin = 1 * inch
        image_width = width - (2 * margin)
        image_height = height - (2 * margin)
        
        # === PAGE 1: COVER IMAGE ===
        if story_cover_url:
            logger.info("Adding cover page...")
            cover_image_data = download_image_from_url(story_cover_url)
            if cover_image_data:
                try:
                    image = PILImage.open(BytesIO(cover_image_data))
                    
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
                    
                    # Draw cover image with 1 inch margins and preserveAspectRatio
                    c.drawImage(
                        img_reader,
                        x=margin,
                        y=margin,
                        width=image_width,
                        height=image_height,
                        preserveAspectRatio=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to process cover image: {e}")
            else:
                logger.warning(f"Failed to download cover image from {story_cover_url}")
        else:
            logger.warning("No cover image URL provided, skipping cover page")
        
        c.showPage()
        
        # === PAGES 2-6: SCENE IMAGES (up to 5 images) ===
        # Limit to 5 scene images to make 6 pages total (1 cover + 5 scenes)
        import ast

        scene_urls_to_use = ast.literal_eval(scene_urls)

        
        for i, scene_url in enumerate(scene_urls_to_use, 1):
            if scene_url:
                logger.info(f"Adding scene {i}/{len(scene_urls_to_use)}...")
                scene_image_data = download_image_from_url(scene_url)
                if scene_image_data:
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
                        
                        # Draw scene image with 1 inch margins and preserveAspectRatio
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
            else:
                logger.warning(f"No scene URL provided for scene {i}, skipping scene page")
            
            c.showPage()
        
        # Save PDF
        c.save()
        
        elapsed = time.time() - start_time
        total_pages = (1 if story_cover_url else 0) + len(scene_urls_to_use)
        logger.info(f"✅ Book PDF created successfully with {total_pages} pages in {elapsed:.2f} seconds")
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

