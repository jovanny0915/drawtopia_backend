"""
PDF Generator using ReportLab (Vercel-compatible, no native deps)
Generates PDFs matching the preview page mobile-image-split structure.
Each mobile-image-split div becomes one PDF page.
"""

import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from pdf_generator import download_image_from_url

logger = logging.getLogger(__name__)

# Letter size
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.5 * inch
TEXT_COLOR = HexColor("#ffffff")
BG_COLOR = HexColor("#1a2f38")


def _clean_url(url: Optional[str]) -> str:
    """Strip query params from URL."""
    if not url or not isinstance(url, str):
        return ""
    return url.split("?")[0].strip()


def _prepare_image(url: str, max_w: float, max_h: float):
    """Download image and return ImageReader, or None."""
    data = download_image_from_url(url)
    if not data:
        return None
    try:
        img = PILImage.open(BytesIO(data))
        if img.mode in ("RGBA", "LA", "P"):
            bg = PILImage.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode in ("RGBA", "LA"):
                bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return ImageReader(img)
    except Exception as e:
        logger.warning(f"Could not prepare image {url}: {e}")
        return None


def _draw_image_fit(
    c: canvas.Canvas,
    img_reader,
    x: float,
    y: float,
    w: float,
    h: float,
    preserve_aspect: bool = True,
):
    """Draw image fitting within rect."""
    if img_reader is None:
        return
    try:
        c.drawImage(img_reader, x, y, width=w, height=h, preserveAspectRatio=preserve_aspect)
    except Exception as e:
        logger.warning(f"Could not draw image: {e}")


def _draw_text_block(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: int = 12,
    leading: float = 14,
):
    """Draw wrapped text block, returns final y."""
    c.setFont(font_name, font_size)
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = f"{current} {w}".strip() if current else w
        if c.stringWidth(test, font_name, font_size) <= width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def _extract_scene_images(story: Dict[str, Any]) -> List[str]:
    """Extract scene image URLs from story_content or scene_images."""
    urls: List[str] = []
    scene_images = story.get("scene_images")
    if scene_images:
        if isinstance(scene_images, str):
            try:
                scene_images = json.loads(scene_images)
            except Exception:
                pass
        if isinstance(scene_images, list):
            for u in scene_images:
                if u:
                    urls.append(str(u).split("?")[0])
            if urls:
                return urls
    content = story.get("story_content")
    if not content:
        return urls
    try:
        if isinstance(content, str):
            content = json.loads(content)
        pages = content.get("pages") if isinstance(content, dict) else content
        if isinstance(pages, list):
            for p in pages:
                if isinstance(p, dict):
                    u = p.get("sceneImage") or p.get("scene") or p.get("imageUrl") or p.get("image_url") or p.get("image")
                    if u:
                        urls.append(str(u).split("?")[0])
    except Exception as e:
        logger.warning(f"Could not extract scene images: {e}")
    return urls


def create_story_pdf_xhtml(
    story: Dict[str, Any],
    output_buffer: BytesIO,
    *,
    logo_url: Optional[str] = None,
    app_url: Optional[str] = None,
) -> bool:
    """
    Create a PDF from story data using ReportLab (no xhtml2pdf - Vercel compatible).
    Matches the preview page mobile-image-split structure.
    """
    try:
        logo = logo_url or (f"{app_url.rstrip('/')}/assets/logo.png" if app_url else "")

        cover_url = _clean_url(story.get("story_cover") or "")
        copyright_image = _clean_url(story.get("copyright_image") or "")
        dedication_image = _clean_url(story.get("dedication_image") or "")
        dedication_text = story.get("dedication_text") or ""
        last_word_image = _clean_url(story.get("last_word_page_image") or "")
        last_admin_image = _clean_url(story.get("last_admin_page_image") or "")
        back_cover_image = _clean_url(story.get("back_cover_image") or "")
        character_name = story.get("character_name") or "[CHARACTER_NAME]"
        child_name = story.get("child_first_name", "[CHILD_NAME]") or "[CHILD_NAME]"

        has_dedication = bool(copyright_image or dedication_image or dedication_text)
        has_last_words = bool(last_word_image or last_admin_image)
        scene_urls = _extract_scene_images(story)

        # Parse dedication
        raw = (dedication_text or "").strip()
        body_text = raw
        signature = ""
        dash_match = re.search(r"\s+[—–-]\s+(.+)$", raw)
        if dash_match:
            body_text = raw[: dash_match.start()].strip()
            sig = (dash_match.group(1) or "").strip()
            signature = f"— {sig}" if sig else ""
        if not body_text:
            body_text = "In every tiny thing you do each day, never forget that you are loved enormously"

        c = canvas.Canvas(output_buffer, pagesize=letter)
        w, h = PAGE_WIDTH, PAGE_HEIGHT
        img_w = w - 2 * MARGIN
        img_h = h - 2 * MARGIN
        half_w = (w - 2 * MARGIN - 10) / 2  # For two-column spreads

        # 1. Cover
        if cover_url:
            ir = _prepare_image(cover_url, img_w, img_h)
            if ir:
                c.setFillColor(white)
                c.rect(0, 0, w, h, fill=1, stroke=0)
                _draw_image_fit(c, ir, MARGIN, MARGIN, img_w, img_h)
            c.showPage()

        # 2. Copyright + Dedication spread
        if has_dedication:
            c.setFillColor(BG_COLOR)
            c.rect(0, 0, w, h, fill=1, stroke=0)
            c.setFillColor(TEXT_COLOR)

            # Left: copyright
            lx, rx = MARGIN, MARGIN + half_w + 10
            ty = h - MARGIN - 20

            if copyright_image:
                ir = _prepare_image(copyright_image, half_w, 200)
                if ir:
                    _draw_image_fit(c, ir, lx, ty - 150, half_w, 150)
                ty -= 160

            copyright_lines = [
                f"This one-of-a-kind adventure story was created just for {child_name}.",
                "Beyond these pages lies a magical world filled with wonder, mystery, and brave moments.",
                f"Follow {character_name} through lands of shadow and light.",
                f"This story celebrates {child_name}'s creativity and courage.",
                "",
                "© 2026 Drawtopia. All rights reserved.",
                "Published by Drawtopia | drawtopia.ai",
            ]
            for line in copyright_lines:
                c.setFont("Helvetica", 10)
                c.drawString(lx, ty, line[:60] + "..." if len(line) > 60 else line)
                ty -= 14

            # Right: dedication
            ty = h - MARGIN - 20
            if dedication_image:
                ir = _prepare_image(dedication_image, half_w, 200)
                if ir:
                    _draw_image_fit(c, ir, rx, ty - 150, half_w, 150)
                ty -= 160

            c.setFont("Helvetica-Bold", 16)
            c.drawString(rx, ty, f"Dear {child_name}")
            ty -= 24
            c.setFont("Helvetica", 12)
            ty = _draw_text_block(c, body_text, rx, ty, half_w - 10, font_size=12, leading=16)
            if signature:
                ty -= 8
                c.drawString(rx, ty, signature)
            c.showPage()

        # 3. Story scene spreads
        for url in scene_urls:
            if not url:
                continue
            c.setFillColor(white)
            c.rect(0, 0, w, h, fill=1, stroke=0)
            ir = _prepare_image(url, img_w, img_h)
            if ir:
                _draw_image_fit(c, ir, MARGIN, MARGIN, img_w, img_h)
            c.showPage()

        # 4. Last words + Admin spread
        if has_last_words:
            c.setFillColor(BG_COLOR)
            c.rect(0, 0, w, h, fill=1, stroke=0)
            c.setFillColor(TEXT_COLOR)

            ty = h - MARGIN - 20
            if last_word_image:
                ir = _prepare_image(last_word_image, half_w, 180)
                if ir:
                    _draw_image_fit(c, ir, MARGIN, ty - 140, half_w, 140)
                ty -= 150

            c.setFont("Helvetica-Bold", 14)
            c.drawString(MARGIN, ty, "A Special Thank You")
            ty -= 20
            c.setFont("Helvetica", 11)
            ty = _draw_text_block(
                c,
                f"This magical adventure wouldn't exist without the incredible imagination of {child_name}. Thank you!",
                MARGIN,
                ty,
                half_w - 10,
                font_size=11,
                leading=14,
            )

            ty = h - MARGIN - 20
            if last_admin_image:
                ir = _prepare_image(last_admin_image, half_w, 180)
                if ir:
                    _draw_image_fit(c, ir, MARGIN + half_w + 10, ty - 140, half_w, 140)
                ty -= 150

            if logo:
                ir = _prepare_image(logo, 80, 80)
                if ir:
                    _draw_image_fit(c, ir, MARGIN + half_w + 10 + (half_w - 80) / 2, ty - 50, 80, 50)
            ty -= 60
            c.setFont("Helvetica-Bold", 12)
            c.drawString(MARGIN + half_w + 10, ty, "Where Every Child Becomes a Storyteller")
            ty -= 18
            c.setFont("Helvetica", 10)
            _draw_text_block(
                c,
                "Their imagination. Their characters. Their stories. Enhanced, not replaced.",
                MARGIN + half_w + 10,
                ty,
                half_w - 10,
                font_size=10,
                leading=12,
            )
            c.showPage()

        # 5. Back cover
        if back_cover_image:
            c.setFillColor(BG_COLOR)
            c.rect(0, 0, w, h, fill=1, stroke=0)
            ir = _prepare_image(back_cover_image, img_w, img_h)
            if ir:
                _draw_image_fit(c, ir, MARGIN, MARGIN, img_w, img_h)
            c.setFillColor(TEXT_COLOR)
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(w / 2, h - MARGIN - 60, "Drawtopia Makes")
            c.drawCentredString(w / 2, h - MARGIN - 80, "Every Child a")
            c.drawCentredString(w / 2, h - MARGIN - 100, "Storyteller")
            c.setFont("Helvetica", 11)
            c.drawCentredString(w / 2, h - MARGIN - 140, "drawtopia.ai")
            c.setFont("Helvetica", 9)
            c.drawCentredString(w / 2, MARGIN + 20, "ISBN placeholder | [Age 6-12]")
            c.showPage()

        c.save()
        logger.info("PDF generated successfully with ReportLab")
        return True
    except Exception as e:
        logger.exception(f"Error creating story PDF: {e}")
        return False
