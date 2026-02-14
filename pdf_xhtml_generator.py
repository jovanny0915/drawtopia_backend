"""
PDF Generator using xhtml2pdf
Generates PDFs matching the preview page mobile-image-split structure.
Each mobile-image-split div becomes one PDF page.
"""

import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from xhtml2pdf import pisa

logger = logging.getLogger(__name__)

# PDF page size (letter) - matches book spread
PAGE_WIDTH_PT = 612
PAGE_HEIGHT_PT = 792

# Base CSS for xhtml2pdf (limited support - avoid height %, complex positioning)
BASE_CSS = """
@page {
    size: letter;
    margin: 0;
}
body {
    margin: 0;
    padding: 0;
    font-family: Helvetica, Arial, sans-serif;
}
.page {
    page-break-after: always;
}
.page:last-child {
    page-break-after: avoid;
}
.cover-page {
    text-align: center;
}
.cover-page img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}
.copyright-page-content,
.dedication-page-content {
    padding: 2rem;
    text-align: center;
    color: #ffffff;
}
.copyright-page-p {
    font-size: 14px;
    line-height: 1.5;
    margin: 0.5em 0;
}
.copyright-page-footer {
    font-size: 12px;
    margin-top: 1em;
}
.dedication-greeting {
    font-size: 24px;
    margin: 0 0 1em 0;
}
.dedication-body {
    font-size: 16px;
    line-height: 1.6;
    margin: 0.5em 0;
}
.dedication-signature {
    font-size: 14px;
    margin-top: 1em;
}
.last-words-page-content,
.last-admin-page-content {
    padding: 1.5rem;
    text-align: center;
    color: #ffffff;
}
.last-words-page-title,
.last-admin-page-title {
    font-size: 20px;
    font-weight: bold;
    margin: 0 0 0.5em 0;
}
.last-words-page-body,
.last-admin-page-body {
    font-size: 14px;
    line-height: 1.5;
    margin: 0.5em 0;
}
.back-cover-content {
    padding: 1.5rem;
    text-align: center;
    color: #ffffff;
}
.back-cover-title {
    font-size: 24px;
    font-weight: bold;
    margin: 0.5em 0;
}
.back-cover-description {
    font-size: 14px;
    line-height: 1.5;
    margin: 0.5em 0;
}
"""


def _clean_url(url: Optional[str]) -> str:
    """Strip query params from URL for consistent fetching."""
    if not url or not isinstance(url, str):
        return ""
    return url.split("?")[0].strip()


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_cover_page(cover_url: str) -> str:
    """Build HTML for cover page (single full image)."""
    url = _clean_url(cover_url)
    if not url:
        return ""
    return f"""
    <div class="page cover-page">
        <img src="{url}" alt="Cover" />
    </div>
    """


def _build_copyright_dedication_spread(
    copyright_image: str,
    dedication_image: str,
    dedication_text: str,
    child_name: str,
    character_name: str,
) -> str:
    """Build HTML for copyright (left) + dedication (right) spread."""
    cpy_img = _clean_url(copyright_image) if copyright_image else ""
    ded_img = _clean_url(dedication_image) if dedication_image else ""
    child = _escape_html(child_name or "[CHILD_NAME]")
    char = _escape_html(character_name or "[CHARACTER_NAME]")

    # Parse dedication: body and signature
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

    body_esc = _escape_html(body_text)
    sig_esc = _escape_html(signature)

    copyright_content = f"""
    <div class="copyright-page-content">
        <div class="copyright-page-text-container">
            <p class="copyright-page-p">This one-of-a-kind adventure story<br />was created just for <b>{child}</b>.</p>
            <p class="copyright-page-p">Beyond these pages lies a magical world<br />filled with wonder, mystery, and brave moments.<br />Every scene unfolds a new chapter in the journey.</p>
            <p class="copyright-page-p">Follow {char} through lands of shadow<br />and light, where courage is tested and imagination<br />guides the way forward.</p>
            <p class="copyright-page-p">This story celebrates <b>{child}</b>'s creativity<br />and courage. Turn the page and begin your adventure<br />into the unknown—where magic awaits.</p>
        </div>
        <p class="copyright-page-footer">© 2026 Drawtopia. All rights reserved.<br />Published by Drawtopia | drawtopia.ai</p>
    </div>
    """

    dedication_content = f"""
    <div class="dedication-page-content">
        <h2 class="dedication-greeting">Dear {child}</h2>
        <p class="dedication-body">{body_esc}</p>
        {f'<p class="dedication-signature">{sig_esc}</p>' if sig_esc else ''}
    </div>
    """

    left_bg = f'<img src="{cpy_img}" alt="" style="width:100%;height:auto;margin-bottom:1em;" />' if cpy_img else ""
    right_bg = f'<img src="{ded_img}" alt="" style="width:100%;height:auto;margin-bottom:1em;" />' if ded_img else ""

    return f"""
    <div class="page">
        <table width="100%" cellpadding="0" cellspacing="2">
            <tr>
                <td width="50%" style="background-color:#1a2f38;padding:1rem;">
                    {left_bg}
                    {copyright_content}
                </td>
                <td width="50%" style="background-color:#1a2f38;padding:1rem;">
                    {right_bg}
                    {dedication_content}
                </td>
            </tr>
        </table>
    </div>
    """


def _build_story_spread(image_url: str) -> str:
    """Build HTML for a story spread (one page per scene image)."""
    url = _clean_url(image_url)
    if not url:
        return ""
    # Full image per page - xhtml2pdf has limited split/crop support.
    # Each mobile-image-split conceptually = one PDF page.
    return f"""
    <div class="page">
        <div class="cover-page">
            <img src="{url}" alt="Scene" style="width:100%;height:auto;" />
        </div>
    </div>
    """


def _build_last_words_admin_spread(
    last_word_image: str,
    last_admin_image: str,
    child_name: str,
    logo_url: str,
) -> str:
    """Build HTML for last words (left) + last admin (right) spread."""
    lw_img = _clean_url(last_word_image) if last_word_image else ""
    la_img = _clean_url(last_admin_image) if last_admin_image else ""
    child = _escape_html(child_name or "[CHILD_NAME]")

    left_bg = f'<img src="{lw_img}" alt="" style="width:100%;height:auto;margin-bottom:1em;" />' if lw_img else ""
    right_bg = f'<img src="{la_img}" alt="" style="width:100%;height:auto;margin-bottom:1em;" />' if la_img else ""
    logo_img = f'<img src="{logo_url}" alt="Drawtopia" style="width:7rem;height:auto;margin-bottom:0.25rem;" />' if logo_url else ""

    return f"""
    <div class="page">
        <table width="100%" cellpadding="0" cellspacing="2">
            <tr>
                <td width="50%" style="background-color:#1a2f38;padding:1rem;">
                    {left_bg}
                    <div class="last-words-page-content">
                        <h2 class="last-words-page-title">A Special Thank You</h2>
                        <p class="last-words-page-body">This magical adventure wouldn't exist without the incredible imagination of {child}. Thank you for sharing your creativity with the world!</p>
                        <p class="last-words-page-body">Every drawing tells a story. Yours told this one.</p>
                    </div>
                </td>
                <td width="50%" style="background-color:#1a2f38;padding:1rem;">
                    {right_bg}
                    <div class="last-admin-page-content">
                        {logo_img}
                        <h2 class="last-admin-page-title">Where Every Child Becomes a Storyteller</h2>
                        <p class="last-admin-page-body">Their imagination. Their characters. Their stories. <u>Enhanced, not replaced.</u></p>
                        <p class="last-admin-page-body">At Drawtopia, we believe every child's drawing holds a story waiting to be told.</p>
                        <a href="https://drawtopia.ai" style="color:#3b82f6;">Drawtopia.ai</a>
                    </div>
                </td>
            </tr>
        </table>
    </div>
    """


def _build_back_cover(back_cover_image: str, logo_url: str) -> str:
    """Build HTML for back cover page."""
    url = _clean_url(back_cover_image) if back_cover_image else ""
    logo_img = f'<img src="{logo_url}" alt="Drawtopia" style="width:6rem;height:auto;" />' if logo_url else ""
    bg_img = f'<img src="{url}" alt="" style="width:100%;height:auto;margin-bottom:1em;" />' if url else ""

    return f"""
    <div class="page">
        <div style="background-color:#1a2f38;padding:1.5rem;">
            {bg_img}
            <div class="back-cover-content">
                <h1 class="back-cover-title">Drawtopia Makes<br />Every Child a<br />Storyteller</h1>
                <p class="back-cover-description">At Drawtopia, we believe every child's drawing holds a story waiting to be told. We use the magic of AI to enhance - never replace - your child's authentic artwork.</p>
                <div style="margin-top:2em;">
                    {logo_img}
                    <p style="font-size:0.9rem;margin:0.25em 0;">Their imagination. Their characters. Their stories. Enhanced, not replaced.</p>
                    <span style="font-size:0.95rem;">drawtopia.ai</span>
                </div>
                <div style="font-size:11px;margin-top:1em;">ISBN placeholder | [Age 6-12]</div>
            </div>
        </div>
    </div>
    """


def _extract_scene_images(story: Dict[str, Any]) -> List[str]:
    """Extract scene image URLs from story_content or scene_images."""
    urls: List[str] = []
    # First try scene_images array (direct column)
    scene_images = story.get("scene_images")
    if scene_images:
        if isinstance(scene_images, str):
            import json
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
    # Fallback: extract from story_content JSON
    content = story.get("story_content")
    if not content:
        return urls
    try:
        if isinstance(content, str):
            import json
            content = json.loads(content)
        if not content:
            return urls
        pages = content.get("pages") if isinstance(content, dict) else content
        if isinstance(pages, list):
            for p in pages:
                if isinstance(p, dict):
                    u = p.get("sceneImage") or p.get("scene") or p.get("imageUrl") or p.get("image_url") or p.get("image")
                    if u:
                        urls.append(str(u).split("?")[0])
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict):
                    u = p.get("sceneImage") or p.get("scene") or p.get("imageUrl") or p.get("image_url") or p.get("image")
                    if u:
                        urls.append(str(u).split("?")[0])
    except Exception as e:
        logger.warning(f"Could not extract scene images from content: {e}")
    return urls


def create_story_pdf_xhtml(
    story: Dict[str, Any],
    output_buffer: BytesIO,
    *,
    logo_url: Optional[str] = None,
    app_url: Optional[str] = None,
) -> bool:
    """
    Create a PDF from story data using xhtml2pdf.
    Matches the preview page mobile-image-split structure: each spread = one PDF page.

    Args:
        story: Story dict from DB (stories table)
        output_buffer: BytesIO to write PDF to
        logo_url: Full URL to Drawtopia logo (e.g. https://app.example.com/assets/logo.png)
        app_url: App base URL for logo if logo_url not provided

    Returns:
        True if successful
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
        child_name = "[CHILD_NAME]"  # Will be filled from child_profiles if available

        # Child name from story - might be in related data
        # For now use placeholder; API can pass child_name if fetched
        if "child_first_name" in story:
            child_name = story["child_first_name"] or child_name

        has_dedication = bool(copyright_image or dedication_image or dedication_text)
        has_last_words = bool(last_word_image or last_admin_image)

        scene_urls = _extract_scene_images(story)

        html_parts: List[str] = []
        html_parts.append(f'<!DOCTYPE html><html><head><meta charset="UTF-8"/><style>{BASE_CSS}</style></head><body>')

        # 1. Cover
        if cover_url:
            html_parts.append(_build_cover_page(cover_url))

        # 2. Copyright + Dedication spread
        if has_dedication:
            html_parts.append(
                _build_copyright_dedication_spread(
                    copyright_image=copyright_image,
                    dedication_image=dedication_image,
                    dedication_text=dedication_text,
                    child_name=child_name,
                    character_name=character_name,
                )
            )

        # 3. Story scene spreads (each scene = one spread)
        for url in scene_urls:
            if url:
                html_parts.append(_build_story_spread(url))

        # 4. Last words + Admin spread
        if has_last_words:
            html_parts.append(
                _build_last_words_admin_spread(
                    last_word_image=last_word_image,
                    last_admin_image=last_admin_image,
                    child_name=child_name,
                    logo_url=logo,
                )
            )

        # 5. Back cover
        if back_cover_image:
            html_parts.append(_build_back_cover(back_cover_image, logo))

        html_parts.append("</body></html>")
        html = "\n".join(html_parts)

        # Remove last page's page-break-after for cleaner output
        html = html.replace('page-break-after: auto;', 'page-break-after: avoid;')

        result = pisa.CreatePDF(
            html,
            dest=output_buffer,
            encoding="utf-8",
        )
        if result.err:
            logger.error(f"xhtml2pdf error: {result.err}")
            return False
        logger.info("PDF generated successfully with xhtml2pdf")
        return True
    except Exception as e:
        logger.exception(f"Error creating story PDF: {e}")
        return False
