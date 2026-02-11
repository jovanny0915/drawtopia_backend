"""
Image Optimization Module
Handles image optimization before storage upload to save space.
Supports conversion to WebP and JPEG formats with quality optimization.
"""
from PIL import Image
from io import BytesIO
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)


class ImageOptimizer:
    """
    Image optimization utility for compressing and converting images
    before storage upload.
    """
    
    # Default settings
    DEFAULT_FORMAT = "WebP"  # WebP provides better compression than JPEG
    DEFAULT_QUALITY = 85  # Balance between quality and file size
    FALLBACK_FORMAT = "JPEG"  # Fallback if WebP is not supported
    MAX_DIMENSION = 2048  # Maximum width or height in pixels
    
    def __init__(
        self,
        format: str = DEFAULT_FORMAT,
        quality: int = DEFAULT_QUALITY,
        max_dimension: Optional[int] = MAX_DIMENSION
    ):
        """
        Initialize the image optimizer.
        
        Args:
            format: Target image format ('WebP' or 'JPEG')
            quality: Compression quality (1-100, higher is better quality)
            max_dimension: Maximum width or height, None for no limit
        """
        self.format = format.upper()
        self.quality = quality
        self.max_dimension = max_dimension
        
        # Validate format
        if self.format not in ["WEBP", "JPEG", "JPG"]:
            logger.warning(f"Invalid format '{format}', falling back to WebP")
            self.format = "WEBP"
        
        # Normalize JPEG format
        if self.format == "JPG":
            self.format = "JPEG"
    
    def optimize_image(
        self,
        image_data: bytes,
        filename: Optional[str] = None
    ) -> Tuple[bytes, str, str]:
        """
        Optimize an image from bytes.
        
        Args:
            image_data: Original image data as bytes
            filename: Optional original filename (for extension detection)
        
        Returns:
            Tuple of (optimized_bytes, content_type, extension)
            
        Raises:
            ValueError: If image data is invalid or cannot be processed
        """
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_data))
            
            # Log original image info
            original_size = len(image_data)
            logger.info(f"Original image: {img.format} {img.size} {img.mode} ({original_size / 1024:.1f} KB)")
            
            # Convert to RGB if necessary (required for JPEG/WebP)
            img = self._convert_to_rgb(img)
            
            # Resize if exceeds max dimension
            if self.max_dimension:
                img = self._resize_image(img, self.max_dimension)
            
            # Optimize and compress
            optimized_data = self._compress_image(img)
            
            # Determine content type and extension
            content_type = self._get_content_type(self.format)
            extension = self._get_extension(self.format)
            
            # Log optimization results
            optimized_size = len(optimized_data)
            compression_ratio = (1 - optimized_size / original_size) * 100
            logger.info(
                f"Optimized image: {self.format} ({optimized_size / 1024:.1f} KB) "
                f"- Saved {compression_ratio:.1f}%"
            )
            
            return optimized_data, content_type, extension
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            raise ValueError(f"Failed to optimize image: {str(e)}")
    
    def optimize_from_file(self, file_path: str) -> Tuple[bytes, str, str]:
        """
        Optimize an image from a file path.
        
        Args:
            file_path: Path to the image file
        
        Returns:
            Tuple of (optimized_bytes, content_type, extension)
        """
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        return self.optimize_image(image_data, filename=os.path.basename(file_path))
    
    def _convert_to_rgb(self, img: Image.Image) -> Image.Image:
        """
        Convert image to RGB mode if necessary.
        WebP and JPEG require RGB mode.
        """
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            if img.mode == 'RGBA' or img.mode == 'LA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(img, mask=img.split()[-1])
                return background
            else:
                # Convert palette mode to RGB
                return img.convert('RGB')
        elif img.mode != 'RGB':
            return img.convert('RGB')
        
        return img
    
    def _resize_image(self, img: Image.Image, max_dimension: int) -> Image.Image:
        """
        Resize image if it exceeds maximum dimension while maintaining aspect ratio.
        
        Args:
            img: PIL Image object
            max_dimension: Maximum width or height
        
        Returns:
            Resized image (or original if within limits)
        """
        width, height = img.size
        
        if width <= max_dimension and height <= max_dimension:
            return img
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        logger.info(f"Resizing image from {img.size} to ({new_width}, {new_height})")
        
        # Use LANCZOS for high-quality downsampling
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _compress_image(self, img: Image.Image) -> bytes:
        """
        Compress image to target format with quality settings.
        
        Args:
            img: PIL Image object
        
        Returns:
            Compressed image as bytes
        """
        output = BytesIO()
        
        if self.format == "WEBP":
            # WebP compression
            img.save(
                output,
                format="WebP",
                quality=self.quality,
                method=6,  # Maximum compression effort (0-6)
                lossless=False  # Use lossy compression for better size reduction
            )
        elif self.format == "JPEG":
            # JPEG compression
            img.save(
                output,
                format="JPEG",
                quality=self.quality,
                optimize=True,  # Enable JPEG optimization
                progressive=True  # Progressive JPEG for better web loading
            )
        else:
            # Fallback to PNG (shouldn't happen with validation)
            img.save(output, format="PNG", optimize=True)
        
        output.seek(0)
        return output.read()
    
    def _get_content_type(self, format: str) -> str:
        """Get MIME content type for format."""
        content_types = {
            "WEBP": "image/webp",
            "JPEG": "image/jpeg",
            "PNG": "image/png"
        }
        return content_types.get(format, "image/jpeg")
    
    def _get_extension(self, format: str) -> str:
        """Get file extension for format."""
        extensions = {
            "WEBP": "webp",
            "JPEG": "jpg",
            "PNG": "png"
        }
        return extensions.get(format, "jpg")


# Convenience function for quick optimization
def optimize_image(
    image_data: bytes,
    format: str = "WebP",
    quality: int = 85,
    max_dimension: Optional[int] = 2048
) -> Tuple[bytes, str, str]:
    """
    Quick image optimization function.
    
    Args:
        image_data: Original image data as bytes
        format: Target format ('WebP' or 'JPEG')
        quality: Compression quality (1-100)
        max_dimension: Maximum dimension, None for no limit
    
    Returns:
        Tuple of (optimized_bytes, content_type, extension)
    
    Example:
        >>> with open('image.jpg', 'rb') as f:
        ...     data = f.read()
        >>> optimized, content_type, ext = optimize_image(data)
        >>> # optimized is now a WebP image at 85% quality
    """
    optimizer = ImageOptimizer(format=format, quality=quality, max_dimension=max_dimension)
    return optimizer.optimize_image(image_data)


# Profile-specific optimizers for different use cases
class TemplateImageOptimizer(ImageOptimizer):
    """Optimizer specifically for book template images (high quality)."""
    
    def __init__(self):
        super().__init__(
            format="WebP",
            quality=90,  # Higher quality for templates
            max_dimension=2048
        )


class ThumbnailOptimizer(ImageOptimizer):
    """Optimizer for thumbnail images (smaller size, lower quality acceptable)."""
    
    def __init__(self):
        super().__init__(
            format="WebP",
            quality=75,  # Lower quality acceptable for thumbnails
            max_dimension=512  # Smaller dimensions
        )


class CoverImageOptimizer(ImageOptimizer):
    """Optimizer for cover images (balanced quality and size)."""
    
    def __init__(self):
        super().__init__(
            format="WebP",
            quality=88,
            max_dimension=1920
        )


if __name__ == "__main__":
    # Test the optimizer
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        print(f"Optimizing: {file_path}")
        
        # Test with different optimizers
        optimizers = [
            ("Standard", ImageOptimizer()),
            ("Template", TemplateImageOptimizer()),
            ("Thumbnail", ThumbnailOptimizer()),
            ("Cover", CoverImageOptimizer()),
        ]
        
        with open(file_path, 'rb') as f:
            original_data = f.read()
        
        print(f"\nOriginal size: {len(original_data) / 1024:.1f} KB")
        
        for name, optimizer in optimizers:
            optimized, content_type, ext = optimizer.optimize_image(original_data)
            print(f"{name}: {len(optimized) / 1024:.1f} KB ({content_type}, .{ext})")
    else:
        print("Usage: python image_optimizer.py <image_file>")
