import os
import time
import threading
import cloudinary
import cloudinary.uploader
import cloudinary.api
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio
from PIL import Image
import pillow_heif
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env_variables() -> bool:
    """Check if all required environment variables are present and valid
    
    Returns:
        bool: True if all required variables are present and valid, False otherwise
    """
    required_vars = {
        'FAL_KEY': 'FAL AI API key',
        'CLOUDINARY_CLOUD_NAME': 'Cloudinary cloud name',
        'CLOUDINARY_API_KEY': 'Cloudinary API key',
        'CLOUDINARY_SECRET_KEY': 'Cloudinary secret key'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        print("\nError: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease add these variables to your .env file")
        return False
    
    # Configure Cloudinary to test credentials
    try:
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_SECRET_KEY')
        )
        # Test Cloudinary configuration
        cloudinary.api.ping()
        print("✓ Cloudinary credentials are valid")
    except Exception as e:
        print(f"\nError: Invalid Cloudinary credentials: {str(e)}")
        return False
    
    print("✓ All environment variables are present and valid")
    return True

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_SECRET_KEY')
)

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'main_images',
        'tail_images',
        'Kling_video',
        'output'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Directory '{directory}' is ready")

def delete_from_cloudinary(public_id: str) -> bool:
    """Delete a file from Cloudinary using its public ID"""
    try:
        result = cloudinary.uploader.destroy(public_id)
        return result.get('result') == 'ok'
    except Exception as e:
        print(f"Error deleting from Cloudinary: {str(e)}")
        return False

def delayed_delete_from_cloudinary(public_id: str, delay_seconds: int = 45) -> None:
    """Delete a file from Cloudinary after a specified delay
    
    Args:
        public_id (str): The public ID of the file to delete
        delay_seconds (int): Number of seconds to wait before deletion
    """
    def delete_after_delay():
        time.sleep(delay_seconds)
        if delete_from_cloudinary(public_id):
            print(f"Successfully deleted file {public_id} from Cloudinary after {delay_seconds} seconds")
        else:
            print(f"Failed to delete file {public_id} from Cloudinary")
    
    # Start deletion in a separate thread
    thread = threading.Thread(target=delete_after_delay)
    thread.daemon = True  # Thread will be terminated when main program exits
    thread.start()

def upload_to_cloudinary(file_path: str) -> dict:
    """Upload a file to Cloudinary
    
    Args:
        file_path (str): Path to the file to upload
    
    Returns:
        dict: Contains publicId and publicUrl of the uploaded file
    """
    try:
        result = cloudinary.uploader.upload(
            file_path,
            resource_type="auto"
        )
        
        return {
            "publicId": result["public_id"],
            "publicUrl": result["secure_url"]
        }
    except Exception as e:
        print(f"Error uploading to Cloudinary: {str(e)}")
        raise 

def convert_to_png(input_path: str, output_dir: str = 'main_images') -> str:
    """
    Convert various image formats to PNG.
    
    Args:
        input_path (str): Path to the input image file
        output_dir (str): Directory to save the converted PNG file (default: 'main_images')
    
    Returns:
        str: Path to the converted PNG file
    
    Supported formats:
    - JPEG/JPG
    - WEBP
    - HEIC/HEIF
    - PNG (will be copied to output directory)
    - Other formats supported by Pillow
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get file extension
        input_path = Path(input_path)
        file_ext = input_path.suffix.lower()
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        output_filename = f"converted_{timestamp}{input_path.stem}.png"
        output_path = Path(output_dir) / output_filename
        
        logger.info(f"Converting {input_path} to PNG format...")
        
        # Handle different input formats
        if file_ext in ['.heic', '.heif']:
            # Handle HEIC/HEIF format
            heif_file = pillow_heif.read_heif(input_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
        else:
            # Handle other formats (JPEG, WEBP, PNG, etc.)
            image = Image.open(input_path)
        
        # Convert to RGB if necessary (for RGBA or other modes)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as PNG
        image.save(output_path, 'PNG', quality=95)
        logger.info(f"Successfully converted to PNG: {output_path}")
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error converting image to PNG: {str(e)}")
        raise 