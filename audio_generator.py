"""
Google Text-to-Speech Audio Generator using gTTS
Uses Google's free TTS service (no API keys or service accounts required)
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any
from io import BytesIO
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Try to import gTTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("gTTS not installed. Install with: pip install gtts>=2.5.0")

# Age-appropriate language and speed mapping
# gTTS uses language codes and can adjust speed
AGE_VOICE_MAPPING = {
    "3-6": {
        "language": "en",  # English
        "tld": "com",  # Top-level domain (affects accent: 'com' = US, 'co.uk' = UK)
        "slow": True,  # Slower speech for younger children
        "lang_check": True
    },
    "7-10": {
        "language": "en",
        "tld": "com",
        "slow": False,  # Normal speed
        "lang_check": True
    },
    "11-12": {
        "language": "en",
        "tld": "com",
        "slow": False,  # Normal speed
        "lang_check": True
    }
}


class AudioGenerator:
    """Generates audio narration using Google Text-to-Speech (gTTS) - No API keys required"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the audio generator
        
        Args:
            api_key: Not required for gTTS (kept for compatibility, ignored)
        """
        if not TTS_AVAILABLE:
            self.available = False
            logger.error("gTTS not available. Install: pip install gtts>=2.5.0")
            return
        
        self.available = True
        logger.info("✅ Google Text-to-Speech (gTTS) initialized - No API keys required")
    
    def generate_audio_for_page(
        self,
        text: str,
        page_number: int,
        age_group: str,
        timeout_seconds: int = 60
    ) -> Optional[bytes]:
        """
        Generate audio for a single story page
        
        Args:
            text: The text content of the page
            page_number: Page number (1-5)
            age_group: Age group ("3-6", "7-10", "11-12")
            timeout_seconds: Maximum time to generate audio (default: 60s)
        
        Returns:
            Audio data as bytes (MP3 format) or None if failed
        """
        if not self.available:
            logger.error("gTTS not available")
            return None
        
        if not text or not text.strip():
            logger.warning(f"⚠️ Page {page_number} has empty text, skipping audio generation")
            return None
        
        start_time = time.time()
        
        try:
            # Get voice configuration for age group
            voice_config = AGE_VOICE_MAPPING.get(age_group, AGE_VOICE_MAPPING["7-10"])
            
            # Generate speech using gTTS
            logger.info(f"Generating audio for page {page_number} (age group: {age_group})...")
            
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=voice_config["language"],
                tld=voice_config["tld"],  # 'com' for US accent
                slow=voice_config["slow"],
                lang_check=voice_config["lang_check"]
            )
            
            # Save to BytesIO buffer
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_data = audio_buffer.read()
            
            elapsed_time = time.time() - start_time
            
            if audio_data and len(audio_data) > 0:
                audio_size_kb = len(audio_data) / 1024
                logger.info(
                    f"✅ Generated audio for page {page_number}: "
                    f"{audio_size_kb:.2f} KB in {elapsed_time:.2f}s"
                )
                
                # Check timeout
                if elapsed_time > timeout_seconds:
                    logger.warning(
                        f"⚠️ Audio generation took {elapsed_time:.2f}s "
                        f"(exceeded {timeout_seconds}s limit)"
                    )
                
                return audio_data
            else:
                logger.error(f"❌ No audio content returned for page {page_number}")
                return None
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"❌ Error generating audio for page {page_number} "
                f"(took {elapsed_time:.2f}s): {e}"
            )
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def generate_audio_for_story(
        self,
        story_pages: List[str],
        age_group: str,
        timeout_per_page: int = 60
    ) -> List[Optional[bytes]]:
        """
        Generate audio for all story pages
        
        Args:
            story_pages: List of text content for each page (5 pages)
            age_group: Age group ("3-6", "7-10", "11-12")
            timeout_per_page: Maximum time per page (default: 60s)
        
        Returns:
            List of audio data (bytes) for each page, None for failed pages
        """
        if not self.available:
            logger.error("gTTS not available")
            return []
        
        if not story_pages or len(story_pages) != 5:
            logger.error(f"Expected 5 story pages, got {len(story_pages) if story_pages else 0}")
            return []
        
        audio_results = []
        
        for i, page_text in enumerate(story_pages, 1):
            if not page_text or not page_text.strip():
                logger.warning(f"⚠️ Page {i} has empty text, skipping audio generation")
                audio_results.append(None)
                continue
            
            audio_data = self.generate_audio_for_page(
                text=page_text,
                page_number=i,
                age_group=age_group,
                timeout_seconds=timeout_per_page
            )
            audio_results.append(audio_data)
        
        successful = sum(1 for audio in audio_results if audio is not None)
        logger.info(
            f"✅ Generated audio for {successful}/{len(story_pages)} pages"
        )
        
        return audio_results
