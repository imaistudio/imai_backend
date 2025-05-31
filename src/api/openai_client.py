from typing import List, Optional, Dict, Any
import openai
from pathlib import Path
import asyncio
import io # Required for BytesIO for image data
from concurrent.futures import ProcessPoolExecutor

from ..utils.config import settings
from ..utils.logger import logger
# ImageProcessor might not be directly needed here if orchestrator passes file paths or bytes

# This function must be defined at the top level or be a static method of a class
# for it to be picklable by ProcessPoolExecutor.
def _process_openai_edit_call(api_key: str, base_url: Optional[str], request_sdk_params: Dict[str, Any], image_paths_str: List[str], mask_path_str: Optional[str]) -> Any:
    thread_local_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    opened_images_for_process = []
    opened_mask_for_process = None
    try:
        for p_str in image_paths_str:
            p = Path(p_str)
            if not p.exists(): 
                # This error should ideally be caught before even trying to send to process pool
                # but as a safeguard within the process:
                raise FileNotFoundError(f"Process: Required image file {p} not found.")
            opened_images_for_process.append(open(p, "rb"))
        
        final_request_params = request_sdk_params.copy()
        final_request_params["image"] = opened_images_for_process[0] if len(opened_images_for_process) == 1 else opened_images_for_process

        if mask_path_str:
            m_path = Path(mask_path_str)
            if m_path.exists():
                opened_mask_for_process = open(m_path, "rb")
                final_request_params["mask"] = opened_mask_for_process
            else:
                logger.warning(f"Process: Mask file {m_path} not found. Proceeding without mask.") # This logger won't go to main process log

        return thread_local_client.images.edit(**final_request_params)
    finally:
        for f_obj in opened_images_for_process:
            if not f_obj.closed:
                f_obj.close()
        if opened_mask_for_process and not opened_mask_for_process.closed:
            opened_mask_for_process.close()

class OpenAIClient:
    """Client for OpenAI's image generation and editing API."""
    
    def __init__(self):
        """Store API configuration but defer client instantiation."""
        # Defer client instantiation to be thread-safe / fork-safe
        self.api_key = settings.api.openai_api_key
        self.base_url = settings.api.openai_api_base
        # self.client is no longer initialized here
        logger.info("OpenAIClient configured for ProcessPoolExecutor; openai.OpenAI object will be created per-call in a separate process.")
    
    async def edit_images_with_prompt(
        self,
        prompt: str,
        image_paths: List[Path], # List of paths to actual image files
        mask_path: Optional[Path] = None, # Optional mask image path
        size: str = "1024x1024", # e.g., "1024x1024", "1536x1024", "1024x1536", "auto"
        quality: str = "high", # e.g., "auto", "high", "medium", "low"
        n: int = 1, # Number of images to generate (1-10 for gpt-image-1)
    ) -> Optional[List[str]]: # Will return list of base64 strings
        """
        Edit or combine images using OpenAI's gpt-image-1 model via /images/edits.
        """
        if not image_paths:
            logger.error("No image paths provided for editing.")
            return None
        
        # Ensure all paths are valid before attempting to send to another process
        for p in image_paths:
            if not p.exists():
                logger.error(f"Pre-check failed: Required image file {p} not found.")
                return None
        if mask_path and not mask_path.exists():
            logger.warning(f"Pre-check: Mask file {mask_path} not found. Proceeding without mask, but it won't be used.")
            mask_path = None

        try:
            logger.info(f"Requesting image edit/combination via asyncio.to_thread. Prompt: '{prompt[:100]}...' and {len(image_paths)} image(s). Model: gpt-image-1.")
            
            # Parameters for the SDK call
            request_params = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "n": n,
                "size": size,
                "quality": quality,
            }

            def _sync_edit_call():
                # Create a new client instance in the thread
                thread_local_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                
                # Open and prepare image files
                opened_images = []
                opened_mask = None
                try:
                    for p in image_paths:
                        opened_images.append(open(p, "rb"))
                    
                    request_params["image"] = opened_images[0] if len(opened_images) == 1 else opened_images
                    
                    if mask_path:
                        opened_mask = open(mask_path, "rb")
                        request_params["mask"] = opened_mask
                    
                    return thread_local_client.images.edit(**request_params)
                finally:
                    # Ensure all files are closed
                    for f in opened_images:
                        if not f.closed:
                            f.close()
                    if opened_mask and not opened_mask.closed:
                        opened_mask.close()

            # Use asyncio.to_thread instead of ProcessPoolExecutor
            response = await asyncio.to_thread(_sync_edit_call)
            
            logger.info(f"!!! OpenAIClient.edit_images_with_prompt - RAW RESPONSE TYPE: {type(response)}, RAW RESPONSE REPR: {repr(response)} !!!")

            base64_images = [img.b64_json for img in response.data if img.b64_json]
            if not base64_images:
                raise ValueError("OpenAI client did not return any image data from /edits endpoint.")
                
            logger.info(f"Successfully edited/combined images ({len(base64_images)} received).")
            return base64_images
            
        except FileNotFoundError as fnfe:
            logger.error(f"FileNotFoundError in edit_images_with_prompt: {str(fnfe)}", exc_info=True)
            return None
        except openai.APIError as e:
            logger.error(f"OpenAI API Error during image edit/combination: {type(e).__name__} - {e.status_code} - {e.message}", exc_info=True)
            return None
        except ValueError as ve:
            logger.error(f"ValueError in edit_images_with_prompt: {str(ve)}", exc_info=True)
            raise  # Re-raise ValueError as it's a critical error
        except BaseException as e:
            logger.critical(f"CRITICAL: OpenAIClient.edit_images_with_prompt BASE_EXCEPTION: {type(e).__name__} - {str(e)}", exc_info=True)
            return None
    
    async def analyze_images(
        self,
        images: List[str], # Expects base64 encoded images
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze images using OpenAI's Vision model.
        
        Args:
            images: List of base64 encoded images
            prompt: Optional prompt for specific analysis
            
        Returns:
            Optional[Dict[str, Any]]: Analysis results or None if analysis fails
        """
        if not images:
            logger.warning("No images provided for analysis.")
            return None
        try:
            # Prepare the messages
            messages_content = []
            if prompt:
                messages_content.append({"type": "text", "text": prompt})
            else:
                messages_content.append({"type": "text", "text": "Analyze these images in detail."})
            
            for img_b64 in images: # Assuming images are already base64 strings
                # Ensure it has the correct prefix for the API
                # The Vision API expects "data:[<mediatype>];base64,<data>"
                # We need to know the media type if not already prefixed.
                # For now, let's assume JPEG if no prefix, but this might need adjustment
                # based on what ImageProcessor.encode_image produces or what gpt-image-1 returns (if analyzing its output)
                image_url_content = f"data:image/png;base64,{img_b64}" # Assuming PNG from gpt-image-1 output_format
                if ";base64," in img_b64: # If already has a prefix
                    image_url_content = img_b64 
                
                messages_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_content
                    }
                })
            
            messages = [
                {
                    "role": "user",
                    "content": messages_content
                }
            ]
            
            def _sync_call():
                # Instantiate client inside the thread
                thread_local_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                return thread_local_client.chat.completions.create(
                    model="gpt-4.1", # Updated model to gpt-4-turbo
                    messages=messages,
                    max_tokens=1500 # Increased max_tokens for potentially detailed analysis
                )

            response = await asyncio.to_thread(_sync_call)
            logger.info(f"!!! OpenAIClient.analyze_images - RAW RESPONSE TYPE: {type(response)}, RAW RESPONSE REPR: {repr(response)} !!!")
            
            # Extract analysis
            analysis = response.choices[0].message.content
            logger.info("Successfully analyzed images with gpt-4-turbo")
            
            usage_dict = None
            if response.usage:
                try:
                    # Try _asdict() first for compatibility if it's a Pydantic model or similar
                    usage_dict = response.usage._asdict()
                except AttributeError:
                    # Fallback for standard objects or if _asdict is not available
                    usage_dict = {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

            return {
                "analysis": analysis,
                "model": response.model,
                "usage": usage_dict
            }
            
        except Exception as e:
            logger.error(f"Error analyzing images with OpenAI (gpt-4-turbo): {str(e)}", exc_info=True)
            return None

    async def analyze_images_with_image_gen(
        self,
        images: List[str], # Expects base64 encoded images
        prompt: Optional[str] = None,
        model: str = "gpt-image-1"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze images using OpenAI's gpt-image-1 model.
        
        Args:
            images: List of base64 encoded images
            prompt: Optional prompt for specific analysis
            model: Model to use (defaults to gpt-image-1)
            
        Returns:
            Optional[Dict[str, Any]]: Analysis results or None if analysis fails
        """
        if not images:
            logger.warning("No images provided for analysis.")
            return None
        try:
            # Prepare the messages for gpt-image-1
            messages_content = []
            if prompt:
                messages_content.append({"type": "text", "text": prompt})
            else:
                messages_content.append({"type": "text", "text": "Analyze these images in detail."})
            
            for img_b64 in images:
                # Ensure it has the correct prefix for the API
                image_url_content = f"data:image/png;base64,{img_b64}" # Assuming PNG from gpt-image-1 output_format
                if ";base64," in img_b64: # If already has a prefix
                    image_url_content = img_b64 
                
                messages_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_content
                    }
                })
            
            messages = [
                {
                    "role": "user",
                    "content": messages_content
                }
            ]
            
            def _sync_call():
                # Instantiate client inside the thread
                thread_local_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                return thread_local_client.chat.completions.create(
                    model=model,  # Use gpt-image-1
                    messages=messages,
                    max_tokens=1500 # Increased max_tokens for potentially detailed analysis
                )

            response = await asyncio.to_thread(_sync_call)
            logger.info(f"!!! OpenAIClient.analyze_images_with_image_gen - RAW RESPONSE TYPE: {type(response)}, RAW RESPONSE REPR: {repr(response)} !!!")
            
            # Extract analysis
            analysis = response.choices[0].message.content
            logger.info(f"Successfully analyzed images with {model}")
            
            usage_dict = None
            if response.usage:
                try:
                    # Try _asdict() first for compatibility if it's a Pydantic model or similar
                    usage_dict = response.usage._asdict()
                except AttributeError:
                    # Fallback for standard objects or if _asdict is not available
                    usage_dict = {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

            return {
                "analysis": analysis,
                "model": response.model,
                "usage": usage_dict
            }
            
        except Exception as e:
            logger.error(f"Error analyzing images with OpenAI ({model}): {str(e)}", exc_info=True)
            return None

    async def generate_image_from_text(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        style: Optional[str] = None,
        output_image_format: str = "b64_json"
    ) -> Optional[List[str]]:
        """
        Generates an image from a text prompt using OpenAI's gpt-image-1 model.
        This targets the /images/generations endpoint.
        """
        try:
            # Validate input parameters
            if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
                raise ValueError("Prompt must be a non-empty string")
                
            if n < 1 or n > 4:
                raise ValueError("Number of images (n) must be between 1 and 4")
                
            if size not in ["1024x1024", "1536x1024", "1024x1536"]:
                raise ValueError("Invalid size. Must be one of: 1024x1024, 1536x1024, 1024x1536")
                
            if quality not in ["high", "medium", "low", "standard"]:
                raise ValueError("Invalid quality. Must be one of: high, medium, low, standard")

            logger.info(f"Requesting image generation with prompt: '{prompt[:100]}...'")
            
            model_to_use = "gpt-image-1"
            request_sdk_params = {
                "model": model_to_use,
                "prompt": prompt,
                "n": n,
                "size": size
            }
            
            # Quality parameter handling
            if quality == "standard":
                request_sdk_params["quality"] = "standard"
            else:
                request_sdk_params["quality"] = quality
            
            # Note: The OpenAI API now automatically returns base64 data for gpt-image-1
            # No need to specify response_format parameter
            
            logger.debug(f"OpenAI /images/generations request params: {request_sdk_params}")

            def _sync_call():
                try:
                    # Instantiate client inside the thread
                    thread_local_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                    return thread_local_client.images.generate(**request_sdk_params)
                except openai.AuthenticationError as auth_err:
                    logger.error(f"OpenAI Authentication Error: {str(auth_err)}")
                    raise ValueError("OpenAI API authentication failed. Please check your API key.")
                except openai.RateLimitError as rate_err:
                    logger.error(f"OpenAI Rate Limit Error: {str(rate_err)}")
                    raise ValueError("OpenAI API rate limit exceeded. Please try again later.")
                except openai.APIError as api_err:
                    logger.error(f"OpenAI API Error: {str(api_err)}")
                    raise ValueError(f"OpenAI API error: {str(api_err)}")
                except Exception as e:
                    logger.error(f"Unexpected error in OpenAI API call: {str(e)}")
                    raise ValueError(f"Unexpected error during image generation: {str(e)}")

            response = await asyncio.to_thread(_sync_call)
            
            if not response or not hasattr(response, 'data'):
                raise ValueError("Invalid response from OpenAI API: no data received")
                
            if not response.data:
                raise ValueError("OpenAI API returned empty data")
                
            base64_images = [img.b64_json for img in response.data if hasattr(img, 'b64_json') and img.b64_json]
            
            if not base64_images:
                raise ValueError("No base64 image data in OpenAI API response")
                
            logger.info(f"Successfully generated {len(base64_images)} image(s) from text")
            return base64_images
            
        except ValueError as ve:
            # Re-raise ValueError with the specific error message
            logger.error(f"Validation error in generate_image_from_text: {str(ve)}")
            raise
        except Exception as e:
            # Log unexpected errors
            error_msg = f"Unexpected error in generate_image_from_text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

# Export the class for easy access
__all__ = ['OpenAIClient'] 