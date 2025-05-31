import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import time
from ..utils.logger import logger
import cloudinary
import cloudinary.uploader

class FalClient:
    """Client for interacting with FAL.ai services."""
    
    def __init__(self):
        """Initialize the FAL.ai client with API key from environment."""
        self.api_key = os.getenv("FAL_API_KEY")
        if not self.api_key:
            raise ValueError("FAL_API_KEY environment variable is required")
        
        self.base_url = "https://fal.run/fal-ai"
        self.headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def reframe_image(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image with FAL.ai Ideogram V3 Reframe service.
        Args:
            image_url: URL of the image to process
            options: Optional parameters for reframe (e.g., image_size)
        Returns:
            Dict containing the processed image data
        """
        try:
            logger.info(f"Starting FAL.ai reframe with image URL: {image_url}")
            # Ensure HTTPS
            image_url = image_url.replace("http://", "https://")
            # Default options
            default_options = {
                "image_size": "landscape_16_9" # square_hd
            }
            final_options = {**default_options, **(options or {})}
            # Use the correct endpoint and payload
            url = f"{self.base_url}/ideogram/v3/reframe"
            payload = {
                "image_url": image_url,
                "image_size": final_options["image_size"]
            }
            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"FAL.ai reframe response: {json.dumps(result, indent=2)}")
            # Extract the first image URL from the images list
            images = result.get("images")
            if not images or not isinstance(images, list) or not images[0].get("url"):
                raise Exception("No image URL found in FAL.ai reframe response")
            return {
                "url": images[0]["url"],
                "original_url": image_url,
                "seed": result.get("seed")
            }
        except Exception as e:
            logger.error(f"Error in FAL reframe: {str(e)}")
            raise
    
    async def upscale_image(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image with FAL.ai Aura upscaling service.
        
        Args:
            image_url: URL of the image to process
            options: Optional parameters for upscaling
                - upscaling_factor: Factor by which to upscale (default: 4)
                - overlapping_tiles: Whether to use overlapping tiles (default: true)
                - checkpoint: Model checkpoint to use (default: "v2")
        
        Returns:
            Dict containing the processed image data
        """
        try:
            logger.info(f"Starting FAL.ai Aura upscaling with image URL: {image_url}")
            
            # Default options
            default_options = {
                "upscaling_factor": 4,
                "overlapping_tiles": True,
                "checkpoint": "v2"
            }
            final_options = {**default_options, **(options or {})}
            
            payload = {
                "image_url": image_url,
                "upscaling_factor": final_options["upscaling_factor"],
                "overlapping_tiles": final_options["overlapping_tiles"],
                "checkpoint": final_options["checkpoint"]
            }
            url = f"{self.base_url}/aura-sr"
            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"FAL.ai Aura upscaling response: {json.dumps(result, indent=2)}")
            # Extract image info
            image_info = result.get("image") or (result.get("data", {}).get("image") if "data" in result else None)
            if not image_info:
                raise Exception("No image info found in FAL.ai response")
            output = {
                "url": image_info.get("url"),
                "content_type": image_info.get("content_type"),
                "file_name": image_info.get("file_name"),
                "file_size": image_info.get("file_size"),
                "width": image_info.get("width"),
                "height": image_info.get("height")
            }
            # Optionally add timings if present
            if "timings" in result:
                output["timings"] = result["timings"]
            # Optionally add request_id if present
            if "request_id" in result:
                output["request_id"] = result["request_id"]
            return output
        except Exception as e:
            logger.error(f"Error in FAL upscaling: {str(e)}")
            if hasattr(e, "response") and e.response.status_code == 422:
                logger.error(f"Validation Error Details: {e.response.json().get('detail')}")
            raise

# Create a singleton instance
fal_client = FalClient()

# --- Cloudinary Upload Utility ---
print("CLOUDINARY_CLOUD_NAME:", os.getenv('CLOUDINARY_CLOUD_NAME'))
print("CLOUDINARY_API_KEY:", os.getenv('CLOUDINARY_API_KEY'))
print("CLOUDINARY_API_SECRET:", os.getenv('CLOUDINARY_API_SECRET'))
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

def upload_to_cloudinary(image_path: Path, folder: str = 'imai') -> dict:
    """
    Upload an image to Cloudinary and return the result dict.
    Args:
        image_path: Path to the image file
        folder: Cloudinary folder to upload to
    Returns:
        dict: { 'secure_url': ..., 'public_id': ... }
    """
    try:
        result = cloudinary.uploader.upload(str(image_path), folder=folder, resource_type='auto')
        return {
            'secure_url': result['secure_url'],
            'public_id': result['public_id']
        }
    except Exception as e:
        logger.error(f'Error uploading to Cloudinary: {str(e)}')
        raise 