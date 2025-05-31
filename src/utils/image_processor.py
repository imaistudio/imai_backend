from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import io
import base64
from .config import settings
from .logger import logger

class ImageProcessor:
    """Utility class for processing images."""
    
    @staticmethod
    def validate_image(image_path: Path) -> bool:
        """
        Validate if the image meets the requirements.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            # Check if file exists
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return False
                
            # Check file extension
            if image_path.suffix.lower()[1:] not in settings.image.supported_formats:
                logger.error(f"Unsupported image format: {image_path.suffix}")
                return False
                
            # Check image size
            img = Image.open(image_path)
            width, height = img.size
            if width > settings.image.max_image_size or height > settings.image.max_image_size:
                logger.error(f"Image dimensions exceed maximum size: {width}x{height}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            return False
    
    @staticmethod
    def load_image(image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Optional[np.ndarray]: Processed image array or None if loading fails
        """
        try:
            if not ImageProcessor.validate_image(image_path):
                return None
                
            # Read image using OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image array
            max_size: Maximum dimension size
            
        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_size / width, max_size / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        return image
    
    @staticmethod
    def encode_image(image: np.ndarray, format: str = "JPEG") -> Optional[str]:
        """
        Encode image to base64 string.
        
        Args:
            image: Input image array
            format: Output image format
            
        Returns:
            Optional[str]: Base64 encoded image string or None if encoding fails
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            
            # Encode to base64
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None
    
    @staticmethod
    def process_images(image_paths: List[Path]) -> List[Tuple[Path, np.ndarray, str]]:
        """
        Process multiple images for API input.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List[Tuple[Path, np.ndarray, str]]: List of tuples containing
                (original path, processed image array, base64 encoded string)
        """
        processed_images = []
        
        for image_path in image_paths:
            # Load and validate image
            image = ImageProcessor.load_image(image_path)
            if image is None:
                continue
                
            # Resize if necessary
            image = ImageProcessor.resize_image(image)
            
            # Encode image
            encoded = ImageProcessor.encode_image(image)
            if encoded is None:
                continue
                
            processed_images.append((image_path, image, encoded))
            
        return processed_images

# Export the class for easy access
__all__ = ['ImageProcessor'] 