import os
import sys
import asyncio
import requests
from pathlib import Path
from dotenv import load_dotenv
import fal_client
from utils import (
    create_directories,
    upload_to_cloudinary,
    delayed_delete_from_cloudinary,
    convert_to_png
)

# Load environment variables
load_dotenv()

def get_latest_image(folder: str) -> str:
    """Get the latest image from the specified folder"""
    try:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")
        
        # Get all image files
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.jpeg')) + list(folder_path.glob('*.png')) + list(folder_path.glob('*.webp'))
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

async def clarity_upscale_image(
    upscale_factor: int = 2,
    prompt: str = "masterpiece, best quality, highres",
    negative_prompt: str = "(worst quality, low quality, normal quality:2)",
    creativity: float = 0.35,
    resemblance: float = 0.6,
    guidance_scale: float = 4,
    num_inference_steps: int = 18,
    enable_safety_checker: bool = True,
    input_image: str = None
) -> dict:
    """Upscale an image using FAL AI's Clarity Upscaler"""
    try:
        # Create directories
        create_directories()
        
        # Get input image
        if input_image:
            # Convert input image to PNG if needed
            image_path = convert_to_png(input_image)
        else:
            # Get latest image from main_images folder
            image_path = get_latest_image('main_images')
        
        # Upload to Cloudinary and get URL
        image_url = get_cloudinary_url(image_path)
        
        # Prepare input for FAL AI
        input_data = {
            "image_url": image_url,
            "prompt": prompt,
            "upscale_factor": upscale_factor,
            "negative_prompt": negative_prompt,
            "creativity": creativity,
            "resemblance": resemblance,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "enable_safety_checker": enable_safety_checker
        }
        
        print("Submitting request to FAL AI...")
        print(f"Arguments: {input_data}")
        
        # Submit the request to FAL AI using async submit
        handler = await fal_client.submit_async(
            "fal-ai/clarity-upscaler",
            arguments=input_data
        )
        
        request_id = handler.request_id
        print(f"Request submitted. Request ID: {request_id}")
        
        # Monitor the status with improved error handling and logging
        max_retries = 60  # 5 minutes maximum wait time
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"\nChecking status (attempt {retry_count + 1}/{max_retries})...")
                status = await fal_client.status_async(
                    "fal-ai/clarity-upscaler",
                    request_id,
                    with_logs=True
                )
                
                print(f"Status: {status}")
                
                if isinstance(status, fal_client.InProgress):
                    print("Status: In Progress")
                    if hasattr(status, 'logs') and status.logs:
                        print("Logs:")
                        for log in status.logs:
                            print(f"- {log}")
                    await asyncio.sleep(10)
                    retry_count += 1
                    continue
                
                if isinstance(status, fal_client.Completed):
                    print("Processing completed successfully!")
                    break
                
                if hasattr(status, 'status'):
                    if status.status == 'COMPLETED':
                        print("Processing completed successfully!")
                        break
                    elif status.status == 'FAILED':
                        error_msg = "Upscaling failed"
                        if hasattr(status, 'error') and status.error:
                            error_msg += f": {status.error}"
                        raise Exception(error_msg)
                
                await asyncio.sleep(10)
                retry_count += 1
                
            except Exception as e:
                print(f"Error checking status: {str(e)}")
                await asyncio.sleep(10)
                retry_count += 1
                continue
        
        if retry_count >= max_retries:
            raise Exception("Timeout waiting for processing to complete")
        
        # Get the result
        try:
            print("\nRetrieving result...")
            result = await fal_client.result_async(
                "fal-ai/clarity-upscaler",
                request_id
            )
            
            print(f"Raw result: {result}")
            
            if not result:
                raise Exception("Empty result received from FAL AI")
            
            # Check for different possible result structures
            image_url = None
            if isinstance(result, dict):
                if 'image' in result and isinstance(result['image'], dict):
                    image_url = result['image'].get('url')
                elif 'url' in result:
                    image_url = result['url']
            
            print(f"Extracted image_url: {image_url}")
            
            if not image_url:
                raise Exception(f"No image URL found in result. Result structure: {result}")
            
            # Save the upscaled image locally
            upscaled_path = Path('output') / f"clarity_upscaled_{Path(image_url).name}"
            
            print(f"Attempting to download upscaled image from: {image_url}")
            response = requests.get(image_url, stream=True)
            print(f"HTTP status code: {response.status_code}")
            response.raise_for_status()
            
            with open(upscaled_path, 'wb') as writer:
                for chunk in response.iter_content(chunk_size=8192):
                    writer.write(chunk)
            
            print(f"Image successfully downloaded to: {upscaled_path}")
            
            # Return the FAL AI URL directly
            return {
                "status": "success",
                "imageUrl": image_url,
                "localPath": str(upscaled_path)
            }
            
        except Exception as e:
            print(f"Error getting result: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Error in clarity_upscale_image: {str(e)}")
        raise

async def main_async():
    """Async main function to run the script"""
    if len(sys.argv) < 2:
        print("Usage: python clarity_upscaler.py [upscale_factor] [prompt] [negative_prompt] [creativity] [resemblance] [guidance_scale] [num_inference_steps] [enable_safety_checker] [input_image]")
        print("Example: python clarity_upscaler.py 2 'masterpiece, best quality' '(worst quality)' 0.35 0.6 4 18 true path/to/image.jpg")
        sys.exit(1)
    
    # Check environment variables
    if not check_env_variables():
        sys.exit(1)
    
    # Optional parameters with defaults
    upscale_factor = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    prompt = sys.argv[2] if len(sys.argv) > 2 else "masterpiece, best quality, highres"
    negative_prompt = sys.argv[3] if len(sys.argv) > 3 else "(worst quality, low quality, normal quality:2)"
    creativity = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
    resemblance = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    guidance_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 4
    num_inference_steps = int(sys.argv[7]) if len(sys.argv) > 7 else 18
    enable_safety_checker = sys.argv[8].lower() == 'true' if len(sys.argv) > 8 else True
    input_image = sys.argv[9] if len(sys.argv) > 9 else None
    
    try:
        print("Starting image upscaling with Clarity...")
        if input_image:
            print(f"Using input image: {input_image}")
        else:
            print("Using latest image from main_images folder")
        
        result = await clarity_upscale_image(
            upscale_factor=upscale_factor,
            prompt=prompt,
            negative_prompt=negative_prompt,
            creativity=creativity,
            resemblance=resemblance,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            enable_safety_checker=enable_safety_checker,
            input_image=input_image
        )
        
        print("\nUpscaling completed!")
        print(f"Image URL: {result.get('imageUrl')}")
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

def check_env_variables() -> bool:
    """Check if all required environment variables are set"""
    required_vars = ['FAL_KEY']
    for var in required_vars:
        if not os.getenv(var):
            print(f"Error: {var} environment variable is not set")
            return False
    return True

if __name__ == "__main__":
    main() 