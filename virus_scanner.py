"""
File upload virus scanning utilities
"""
import os
import logging
from typing import Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)

try:
    import clamd
    CLAMD_AVAILABLE = True
except ImportError:
    CLAMD_AVAILABLE = False
    logger.warning("clamd package not installed. Virus scanning will use basic checks only. Install with: pip install clamd")


class VirusScanner:
    """
    Virus scanner for uploaded files
    """
    
    def __init__(self):
        self.clamd_client = None
        if CLAMD_AVAILABLE:
            try:
                # Try to connect to ClamAV daemon
                self.clamd_client = clamd.ClamdUnixSocket()
                # Test connection
                self.clamd_client.ping()
                logger.info("✅ ClamAV daemon connected successfully")
            except Exception as e:
                logger.warning(f"ClamAV daemon not available: {e}. Using basic checks only.")
                self.clamd_client = None
    
    def scan_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Scan file for viruses and malware
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with scan results
        """
        result = {
            "is_safe": True,
            "threats_found": [],
            "scan_method": "basic",
            "file_hash": self._calculate_hash(file_data),
            "file_size": len(file_data)
        }
        
        # Basic checks first
        basic_check = self._basic_security_check(file_data, filename)
        if not basic_check["is_safe"]:
            result.update(basic_check)
            return result
        
        # ClamAV scan if available
        if self.clamd_client:
            try:
                scan_result = self.clamd_client.instream(file_data)
                result["scan_method"] = "clamav"
                
                # Parse ClamAV result
                if scan_result and 'stream' in scan_result:
                    status = scan_result['stream']
                    if status[0] == 'FOUND':
                        result["is_safe"] = False
                        result["threats_found"].append(status[1])
                        logger.warning(f"Virus detected in {filename}: {status[1]}")
                    elif status[0] == 'OK':
                        result["is_safe"] = True
                        logger.info(f"File {filename} scanned clean")
                    else:
                        logger.warning(f"Unknown ClamAV status: {status}")
                        
            except Exception as e:
                logger.error(f"ClamAV scan error: {e}")
                result["scan_method"] = "basic_fallback"
                result["scan_error"] = str(e)
        
        return result
    
    def _basic_security_check(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Perform basic security checks on file
        
        Args:
            file_data: File content
            filename: Original filename
            
        Returns:
            Dictionary with check results
        """
        result = {
            "is_safe": True,
            "threats_found": [],
            "scan_method": "basic"
        }
        
        # Check file size (prevent zip bombs)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_data) > max_size:
            result["is_safe"] = False
            result["threats_found"].append("File too large (potential zip bomb)")
            return result
        
        # Check for suspicious file extensions
        suspicious_extensions = [
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.msi', '.app', '.deb', '.rpm', '.dmg', '.pkg', '.sh'
        ]
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext in suspicious_extensions:
            result["is_safe"] = False
            result["threats_found"].append(f"Suspicious file extension: {file_ext}")
            return result
        
        # Check for executable signatures in file content
        executable_signatures = [
            b'MZ',  # Windows executable
            b'\x7fELF',  # Linux executable
            b'\xca\xfe\xba\xbe',  # macOS Mach-O
            b'#!/bin/sh',  # Shell script
            b'#!/bin/bash',  # Bash script
        ]
        
        for signature in executable_signatures:
            if file_data.startswith(signature):
                result["is_safe"] = False
                result["threats_found"].append("Executable file signature detected")
                return result
        
        # Check for embedded scripts in images (basic)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            suspicious_patterns = [
                b'<script',
                b'<?php',
                b'eval(',
                b'base64_decode'
            ]
            for pattern in suspicious_patterns:
                if pattern in file_data.lower():
                    result["is_safe"] = False
                    result["threats_found"].append("Suspicious code pattern in image")
                    return result
        
        return result
    
    def _calculate_hash(self, file_data: bytes) -> str:
        """
        Calculate SHA-256 hash of file
        
        Args:
            file_data: File content
            
        Returns:
            SHA-256 hash as hex string
        """
        return hashlib.sha256(file_data).hexdigest()
    
    def is_available(self) -> bool:
        """Check if virus scanning is available"""
        return self.clamd_client is not None


# Global scanner instance
_scanner_instance: Optional[VirusScanner] = None

def get_virus_scanner() -> VirusScanner:
    """Get or create virus scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = VirusScanner()
    return _scanner_instance

