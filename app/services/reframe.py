import os
import sys
import time
import asyncio
import requests
from pathlib import Path
from dotenv import load_dotenv
import fal_client
from utils import create_directories, upload_to_cloudinary, delayed_delete_from_cloudinary

# Load environment variables
load_dotenv()

def get_latest_image(folder: str) -> str:
    """Get the latest image from the specified folder"""
    try:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")
        
        # Get all image files
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.jpeg')) + list(folder_path.glob('*.png'))
        if not image_files:
            raise FileNotFoundError(f"No images found in {folder}")
        
        # Get the latest image
        latest_image = max(image_files, key=lambda x: x.stat().st_mtime)
        return str(latest_image)
    
    except Exception as e:
        print(f"Error getting latest image: {str(e)}")
        raise

def get_cloudinary_url(image_path: str) -> str:
    """Upload image to Cloudinary and get URL"""
    try:
        # Upload image to Cloudinary
        cloudinary_result = upload_to_cloudinary(image_path)
        
        # Schedule deletion after 45 seconds
        delayed_delete_from_cloudinary(cloudinary_result["publicId"])
        
        return cloudinary_result["publicUrl"]
    
    except Exception as e:
        print(f"Error getting Cloudinary URL: {str(e)}")
        raise

def on_queue_update(update):
    """Handle queue updates from FAL AI"""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

async def reframe_image(
    image_size: str = "square_hd",
    rendering_speed: str = "BALANCED",
    num_images: int = 1
) -> dict:
    print("[DEBUG] Entered reframe_image")
    try:
        # Create directories
        print("[DEBUG] Creating directories...")
        create_directories()
        
        # Get latest image from main_images folder
        print("[DEBUG] Getting latest image from main_images...")
        image_path = get_latest_image('main_images')
        print(f"[DEBUG] Got image path: {image_path}")
        
        # Upload to Cloudinary and get URL
        print("[DEBUG] Uploading to Cloudinary...")
        image_url = get_cloudinary_url(image_path)
        print(f"[DEBUG] Cloudinary URL: {image_url}")
        
        # Prepare input for FAL AI
        input_data = {
            "image_url": image_url,
            "image_size": image_size,
            "rendering_speed": rendering_speed,
            "num_images": num_images
        }
        print(f"[DEBUG] Input data for FAL: {input_data}")
        
        # Submit the request to FAL AI using async submit
        handler = await fal_client.submit_async(
            "fal-ai/ideogram/v3/reframe",
            arguments=input_data
        )
        request_id = handler.request_id
        print(f"[DEBUG] Request submitted. Request ID: {request_id}")
        
        # Monitor the status
        max_retries = 60
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"[DEBUG] Checking status (attempt {retry_count + 1}/{max_retries})...")
                status = fal_client.status(
                    "fal-ai/ideogram/v3/reframe",
                    request_id,
                    with_logs=True
                )
                print(f"[DEBUG] Status: {status}")
                if isinstance(status, fal_client.InProgress):
                    print("[DEBUG] Status: In Progress")
                    if hasattr(status, 'logs') and status.logs:
                        print("[DEBUG] Logs:")
                        for log in status.logs:
                            print(f"- {log}")
                    await asyncio.sleep(10)
                    retry_count += 1
                    continue
                if isinstance(status, fal_client.Completed):
                    print("[DEBUG] Processing completed successfully!")
                    break
                if hasattr(status, 'status'):
                    if status.status == 'COMPLETED':
                        print("[DEBUG] Processing completed successfully!")
                        break
                    elif status.status == 'FAILED':
                        error_msg = "Reframing failed"
                        if hasattr(status, 'error') and status.error:
                            error_msg += f": {status.error}"
                        print(f"[DEBUG] {error_msg}")
                        raise Exception(error_msg)
                await asyncio.sleep(10)
                retry_count += 1
            except Exception as e:
                print(f"[DEBUG] Error checking status: {str(e)}")
                await asyncio.sleep(10)
                retry_count += 1
                continue
        if retry_count >= max_retries:
            print("[DEBUG] Timeout waiting for processing to complete")
            raise Exception("Timeout waiting for processing to complete")
        # Get the result
        try:
            print("[DEBUG] Retrieving result...")
            result = await fal_client.result_async(
                "fal-ai/ideogram/v3/reframe",
                request_id
            )
            print(f"[DEBUG] Raw result: {result}")
            if not result:
                print("[DEBUG] Empty result received from FAL AI")
                raise Exception("Empty result received from FAL AI")
            image_url = None
            if isinstance(result, dict):
                image_url = result.get('image_url')
                if not image_url and 'images' in result:
                    images = result.get('images', [])
                    if images and isinstance(images, list) and len(images) > 0:
                        image_url = images[0].get('url')
            print(f"[DEBUG] Extracted image_url: {image_url}")
            if not image_url:
                print(f"[DEBUG] No image URL found in result. Result structure: {result}")
                raise Exception(f"No image URL found in result. Result structure: {result}")
            reframed_path = Path('output') / f"reframed_{Path(image_url).name}"
            print(f"[DEBUG] Attempting to download reframed image from: {image_url}")
            response = requests.get(image_url, stream=True)
            print(f"[DEBUG] HTTP status code: {response.status_code}")
            response.raise_for_status()
            with open(reframed_path, 'wb') as writer:
                for chunk in response.iter_content(chunk_size=8192):
                    writer.write(chunk)
            print(f"[DEBUG] Image successfully downloaded to: {reframed_path}")
            cloudinary_result = upload_to_cloudinary(str(reframed_path))
            delayed_delete_from_cloudinary(cloudinary_result["publicId"])
            return {
                "imageUrl": image_url,
                "cloudinaryUrl": cloudinary_result["publicUrl"],
                "localPath": str(reframed_path),
                "requestId": request_id
            }
        except Exception as e:
            print(f"[DEBUG] Error getting result: {str(e)}")
            print(f"[DEBUG] Result structure: {result if 'result' in locals() else 'No result available'}")
            raise
    except Exception as e:
        print(f"[DEBUG] Exception in reframe_image: {str(e)}")
        raise

def get_reframe_status(request_id: str) -> dict:
    """Get the status of a reframing request"""
    try:
        fal = Fal(api_key=os.getenv('FAL_KEY'))
        status = fal.queue.status(
            "fal-ai/outpaint",
            request_id=request_id,
            logs=True
        )
        return status
    except Exception as e:
        print(f"Error in get_reframe_status: {str(e)}")
        raise

def get_reframe_result(request_id: str) -> dict:
    """Get the result of a completed reframing"""
    try:
        fal = Fal(api_key=os.getenv('FAL_KEY'))
        result = fal.queue.result(
            "fal-ai/outpaint",
            request_id=request_id
        )
        return result
    except Exception as e:
        print(f"Error in get_reframe_result: {str(e)}")
        raise

def check_env_variables() -> bool:
    """Check if all required environment variables are set"""
    required_vars = ['FAL_KEY']
    for var in required_vars:
        if not os.getenv(var):
            print(f"Error: {var} environment variable is not set")
            return False
    return True

async def main_async():
    """Async main function to run the script"""
    if len(sys.argv) < 2:
        print("Usage: python reframe.py [image_size] [rendering_speed] [num_images]")
        print("Example: python reframe.py square_hd BALANCED 1")
        sys.exit(1)
    
    # Check environment variables
    if not check_env_variables():
        sys.exit(1)
    
    # Optional parameters
    image_size = sys.argv[1] if len(sys.argv) > 1 else "square_hd"
    rendering_speed = sys.argv[2] if len(sys.argv) > 2 else "BALANCED"
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    try:
        print("Starting image reframing...")
        print("Using latest image from main_images folder")
        
        result = await reframe_image(
            image_size=image_size,
            rendering_speed=rendering_speed,
            num_images=num_images
        )
        
        print("\nReframing completed!")
        print(f"Image URL: {result.get('imageUrl')}")
        print(f"Cloudinary URL: {result.get('cloudinaryUrl')}")
        if 'localPath' in result:
            print(f"Image saved at: {result['localPath']}")
        
        print("\nNote: All uploaded files will be automatically deleted from Cloudinary after 45 seconds.")
        print("Please download or save the image URL if needed.")
        
        # Wait for 45 seconds to ensure deletion messages are shown
        await asyncio.sleep(45)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the script"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 