"""
Security utilities for data encryption, sanitization, and validation
"""
import os
import re
import bleach
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Encryption key management
def get_encryption_key() -> bytes:
    """
    Get or generate encryption key for sensitive data.
    In production, this should be stored securely (e.g., AWS KMS, HashiCorp Vault)
    """
    key_env = os.getenv("ENCRYPTION_KEY")
    if key_env:
        return key_env.encode()
    
    # Generate key from password and salt
    password = os.getenv("ENCRYPTION_PASSWORD", "default-change-in-production").encode()
    salt = os.getenv("ENCRYPTION_SALT", "default-salt-change-in-production").encode()
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key


def encrypt_data(data: str) -> str:
    """
    Encrypt sensitive data at rest
    
    Args:
        data: Plain text data to encrypt
        
    Returns:
        Encrypted data as base64 string
    """
    try:
        key = get_encryption_key()
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise


def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypt sensitive data
    
    Args:
        encrypted_data: Base64 encoded encrypted data
        
    Returns:
        Decrypted plain text
    """
    try:
        key = get_encryption_key()
        f = Fernet(key)
        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = f.decrypt(decoded)
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise


def sanitize_html(text: str, allowed_tags: Optional[list] = None) -> str:
    """
    Sanitize HTML input to prevent XSS attacks
    
    Args:
        text: Input text that may contain HTML
        allowed_tags: List of allowed HTML tags (default: none)
        
    Returns:
        Sanitized text
    """
    if allowed_tags is None:
        allowed_tags = []
    
    return bleach.clean(
        text,
        tags=allowed_tags,
        strip=True,
        strip_comments=True
    )


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize general text input
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Trim to max length
    text = text[:max_length]
    
    # Remove HTML tags
    text = sanitize_html(text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()


def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    # Check if it's 10-15 digits
    return bool(re.match(r'^\d{10,15}$', cleaned))


def validate_url(url: str) -> bool:
    """
    Validate URL format and protocol
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
    return bool(re.match(pattern, url))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal attacks
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and parent directory references
    filename = os.path.basename(filename)
    filename = filename.replace('..', '')
    
    # Allow only alphanumeric, dots, hyphens, and underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename


def check_sql_injection(text: str) -> bool:
    """
    Basic SQL injection pattern detection
    
    Args:
        text: Input text to check
        
    Returns:
        True if suspicious patterns detected, False otherwise
    """
    suspicious_patterns = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(--)",
        r"(;.*--)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)"
    ]
    
    text_upper = text.upper()
    for pattern in suspicious_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            logger.warning(f"Potential SQL injection detected: {pattern}")
            return True
    
    return False


def validate_age_group(age_group: str) -> bool:
    """
    Validate age group format for COPPA compliance
    
    Args:
        age_group: Age group string
        
    Returns:
        True if valid, False otherwise
    """
    valid_groups = ["3-6", "7-10", "11-12", "13+"]
    return age_group in valid_groups


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging/display
    
    Args:
        data: Sensitive data to mask
        mask_char: Character to use for masking
        visible_chars: Number of characters to leave visible at the end
        
    Returns:
        Masked string
    """
    if len(data) <= visible_chars:
        return mask_char * len(data)
    
    masked_length = len(data) - visible_chars
    return (mask_char * masked_length) + data[-visible_chars:]


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token
    
    Args:
        length: Length of token in bytes
        
    Returns:
        Secure random token as hex string
    """
    return os.urandom(length).hex()

