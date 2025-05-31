import google.generativeai as genai
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path
from PIL import Image # For image loading and format checking
import io # For BytesIO

from ..utils.config import settings
from ..utils.logger import logger
from ..utils.image_processor import ImageProcessor # To use its encoding potentially

class GeminiClient:
    """Client for Google Gemini API."""

    def __init__(self):
        """Initialize the Gemini client with API key."""
        try:
            genai.configure(api_key=settings.api.gemini_api_key)
            # Ensure the model used supports multimodal input if we plan to send image data
            # You updated to gemini-2.5-flash-preview-04-17. Let's assume it's multimodal.
            # Otherwise, models like 'gemini-1.5-pro-latest' or 'gemini-1.5-flash-latest' are good choices.
            self.model_name = getattr(settings.api, 'gemini_model_name', 'gemini-1.5-flash-latest') # Allow override via .env
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized successfully with {self.model_name}.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}", exc_info=True)
            self.model = None

    async def _get_image_part(self, image_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        """Convert a single image path to a Gemini API Part object."""
        if not image_path or not image_path.exists():
            if image_path: # Only log if a path was actually given but not found
                 logger.warning(f"Image not found at {image_path}, skipping for Gemini.")
            return None
        try:
            img = Image.open(image_path)
            img_format = img.format or "JPEG"
            mime_type = f"image/{img_format.lower()}"
            byte_io = io.BytesIO()
            img.save(byte_io, format=img_format)
            img_bytes = byte_io.getvalue()
            logger.info(f"Prepared image {image_path} ({mime_type}) for Gemini.")
            return {"mime_type": mime_type, "data": img_bytes}
        except Exception as e:
            logger.error(f"Failed to process image {image_path} for Gemini: {e}", exc_info=True)
            return None

    async def generate_structured_edit_prompt(
        self,
        user_core_prompt: str,
        product_image_path: Path, # Mandatory
        color_image_path: Optional[Path],
        design_image_path: Optional[Path],
        max_length: int = 4000 
    ) -> str: # Returns the final prompt for OpenAI, or original user_core_prompt on failure
        """
        Generates a structured prompt for OpenAI image editing by:
        1. Describing each provided image (product, color source, design source) using Gemini.
        2. Combining these descriptions with the user's core prompt/goal.
        3. Formulating a detailed instruction for OpenAI's /images/edits endpoint.
        """
        if not self.model:
            logger.error("Gemini model not initialized. Falling back to user's core prompt.")
            return user_core_prompt

        product_img_part = await self._get_image_part(product_image_path)
        if not product_img_part: # Product image is essential
            logger.error(f"Failed to process mandatory product image {product_image_path}. Falling back to user's core prompt.")
            return user_core_prompt

        color_img_part = await self._get_image_part(color_image_path)
        design_img_part = await self._get_image_part(design_image_path)

        descriptions = {"product": "Product could not be described.", "color": "Color source not provided or could not be described.", "design": "Design source not provided or could not be described."}
        gemini_tasks = []

        # Task for Product Image Description
        gemini_tasks.append(self.model.generate_content_async(["Describe the main product in this image, paying attention to its form, material, and distinct surfaces that could receive a new design. Example: 'Red leather handbag with a large front flap.' Focus on aspects relevant for applying a pattern.", product_img_part]))
        
        # Task for Color Image Description
        if color_img_part:
            gemini_tasks.append(self.model.generate_content_async(["Extract and list the dominant colors and describe the overall color palette of this image. Example: 'Dominant colors: deep blue, vibrant orange, gold. Palette: warm and contrasting.'", color_img_part]))
        else:
            gemini_tasks.append(None) # Placeholder for indexing

        # Task for Design Image Description
        if design_img_part:
            gemini_tasks.append(self.model.generate_content_async(["Analyze key design elements, motifs, patterns, and artistic style of this image. Example: 'Style: Art Deco. Motifs: Geometric shapes, chevrons, metallic accents.'", design_img_part]))
        else:
            gemini_tasks.append(None) # Placeholder for indexing
        
        try:
            task_responses = await asyncio.gather(*[task for task in gemini_tasks if task is not None], return_exceptions=True)
            
            current_response_idx = 0
            if not isinstance(task_responses[current_response_idx], Exception):
                descriptions["product"] = task_responses[current_response_idx].text.strip()
                logger.info(f"Gemini description for Product Image: {descriptions['product'][:100]}...")
            else:
                logger.error(f"Gemini failed to describe Product Image: {task_responses[current_response_idx]}")
            current_response_idx += 1
            
            if color_img_part:
                if current_response_idx < len(task_responses) and not isinstance(task_responses[current_response_idx], Exception):
                    descriptions["color"] = task_responses[current_response_idx].text.strip()
                    logger.info(f"Gemini description for Color Image: {descriptions['color'][:100]}...")
                else:
                    logger.error(f"Gemini failed to describe Color Image: {task_responses[current_response_idx] if current_response_idx < len(task_responses) else 'No response gathered'}")
                current_response_idx += 1
            
            if design_img_part:
                if current_response_idx < len(task_responses) and not isinstance(task_responses[current_response_idx], Exception):
                    descriptions["design"] = task_responses[current_response_idx].text.strip()
                    logger.info(f"Gemini description for Design Image: {descriptions['design'][:100]}...")
                else:
                    logger.error(f"Gemini failed to describe Design Image: {task_responses[current_response_idx] if current_response_idx < len(task_responses) else 'No response gathered'}")

        except Exception as e:
            logger.error(f"Error during Gemini image description phase: {e}", exc_info=True)

        master_prompt_system_instruction = (
            f"You are an expert prompt engineer for an AI image editing model (OpenAI gpt-image-1 using an /edits endpoint). "
            f"Your goal is to create a precise, actionable prompt for this model based on image analyses and a user's core request.\n\n"
            f"ANALYSIS RESULTS:\n"
            f"1. Product Image Description (This is the target image to be modified):\n{descriptions['product']}\n\n"
            f"2. Color Palette Image Description (Source for colors for the new design. If not provided, be creative or use colors from product/design images if suitable or user prompt hints at it.):\n{descriptions['color']}\n\n"
            f"3. Design Inspiration Image Description (Source for style/motifs for the new design. If not provided, create a novel design based on user prompt and product type.):\n{descriptions['design']}\n\n"
            f"USER'S CORE REQUEST: \"{user_core_prompt}\"\n\n"
            f"TASK: Generate a single, consolidated prompt for the AI image editing model. This prompt MUST instruct the model to:\n"
            f"  a. Use the Product Image as the primary base for editing.\n"
            f"  b. Create a NEW design pattern/style. This new pattern should be INSPIRED by the elements/style described in the 'Design Inspiration Image Description'. If no design image was provided, the new design should be creatively conceived based on the user's core request and the nature of the product.\n"
            f"  c. The new design pattern MUST primarily use colors sourced from the 'Color Palette Image Description'. If no color image was provided, choose colors that are harmonious with the product, design inspiration, or as hinted by the user's core request.\n"
            f"  d. This new design MUST be applied ONLY to the relevant surfaces of the product, as identified in the 'Product Image Description'.\n"
            f"  e. CRITICALLY IMPORTANT: The product's original SHAPE, FORM, overall structure, and its original BACKGROUND (unless the user's core request explicitly demands a background change) MUST BE PRESERVED. The edit should be focused on the product's surface design.\n"
            f"  f. The final prompt should be highly descriptive, focusing on visual details, materials, artistic style, and desired transformations for the image editing model. Do not chat or explain; just provide the direct prompt.\n"
            f"Example of an excellent final prompt: 'Apply a new Art Deco-inspired geometric pattern, featuring repeating gold chevrons and stylized sunburst motifs (inspired by the Design Image analysis), onto the main front panel of the described leather handbag. The pattern should utilize a color palette of deep teal, cream, and rose gold (derived from the Color Image analysis). Ensure the handbag's original silhouette, handles, and its existing plain background are perfectly preserved, with the changes limited to the new surface design.'"
            f"\nFINAL PROMPT FOR IMAGE EDITING MODEL:"
        )
        
        try:
            logger.info("Sending final prompt generation request to Gemini.")
            final_response = await asyncio.to_thread(
                self.model.generate_content,
                [master_prompt_system_instruction],
                generation_config=genai.types.GenerationConfig(candidate_count=1, temperature=0.5)
            )
            enhanced_prompt = final_response.text.strip()
            
            if len(enhanced_prompt) > max_length:
                logger.warning(f"Gemini enhanced prompt exceeded max_length ({len(enhanced_prompt)} > {max_length}). Truncating.")
                enhanced_prompt = enhanced_prompt[:max_length - 3] + "..."
            
            logger.info(f"User Core Prompt: {user_core_prompt}")
            logger.info(f"Gemini Generated Final Edit Prompt: {enhanced_prompt}")
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error generating final edit prompt with Gemini: {e}", exc_info=True)
            logger.warning("Falling back to user's core prompt due to Gemini API error in final stage.")
            return user_core_prompt

    async def generate_design_creation_prompt(
        self,
        image_path1: Optional[Path],
        image_path2: Optional[Path],
        image_path3: Optional[Path],
        user_directive: str,
        max_length: int = 1000 # DALL-E prompts are shorter
    ) -> str:
        """
        Generates a prompt for creating a new design image based on up to three inspirational images
        and a user's textual directive. This prompt is intended for an image *generation* model (e.g., DALL-E 3).
        """
        if not self.model:
            logger.error("Gemini model not initialized. Falling back to user's directive.")
            return user_directive

        img_parts_tasks = []
        if image_path1:
            img_parts_tasks.append(self._get_image_part(image_path1))
        if image_path2:
            img_parts_tasks.append(self._get_image_part(image_path2))
        if image_path3:
            img_parts_tasks.append(self._get_image_part(image_path3))
        
        image_parts_results = await asyncio.gather(*img_parts_tasks)
        
        valid_image_parts = [part for part in image_parts_results if part is not None]

        if not valid_image_parts and not user_directive:
            logger.error("No valid images and no user directive provided for design creation.")
            return "abstract colorful pattern" # Fallback generic prompt

        image_descriptions_text = ""
        if valid_image_parts:
            description_prompts = []
            for i, part in enumerate(valid_image_parts):
                description_prompts.append(
                    self.model.generate_content_async(
                        [f"Describe the key visual elements, colors, textures, and style of this image ({i+1}). Focus on aspects that could inspire a new design. Be concise.", part]
                    )
                )
            
            description_responses = await asyncio.gather(*description_prompts, return_exceptions=True)
            
            for i, res in enumerate(description_responses):
                if not isinstance(res, Exception) and hasattr(res, 'text'):
                    image_descriptions_text += f"Image {i+1} Description: {res.text.strip()}\\n"
                    logger.info(f"Gemini description for Design Input Image {i+1}: {res.text.strip()[:100]}...")
                else:
                    logger.error(f"Gemini failed to describe Design Input Image {i+1}: {res}")
                    image_descriptions_text += f"Image {i+1}: Could not be described.\\n"
            image_descriptions_text += "\\n"


        master_prompt_system_instruction = (
            f"You are an expert prompt writer for an AI image *generation* model (like DALL-E 3 or gpt-image-1 via /images/generations endpoint). "
            f"Your goal is to create a single, detailed, and highly descriptive prompt to generate a *new standalone design image*. "
            f"This design should be inspired by the provided image analyses and a user's directive.\\n\\n"
            f"ANALYSIS OF INSPIRATIONAL IMAGES (if any provided):\\n"
            f"{image_descriptions_text if image_descriptions_text else 'No images were provided or described.'}\\n"
            f"USER'S DIRECTIVE FOR THE NEW DESIGN: \\\"{user_directive}\\\"\\n\\n"
            f"TASK: Generate a single, consolidated prompt for the AI image generation model. This prompt MUST:\\n"
            f"  a. Describe a new, coherent design (e.g., a pattern, texture, artwork, abstract concept, a scene if applicable for a design). \\n"
            f"  b. Synthesize elements, styles, colors, or concepts from the 'ANALYSIS OF INSPIRATIONAL IMAGES' and the 'USER'S DIRECTIVE'.\\n"
            f"  c. The generated image should be a self-contained design, suitable for use as a texture, pattern, or piece of art. It is NOT an edit of an existing product.\\n"
            f"  d. The prompt should be rich in visual details: specify colors, shapes, textures, artistic style (e.g., 'vector art', 'photorealistic', 'watercolor', 'abstract oil painting'), mood, and composition.\\n"
            f"  e. Do NOT refer to 'the user' or 'the images provided'. The prompt should be a direct instruction to the image generation model as if you are describing the desired final image from scratch.\\n"
            f"  f. Ensure the prompt is concise enough for models like DALL-E (around {max_length} characters or less is ideal) but still very descriptive.\\n"
            f"Example of an excellent final prompt: 'Seamless tileable pattern of intricate psychedelic paisley swirls, vibrant blues, deep purples, and electric pinks, with a glossy, almost liquid texture, reminiscent of 70s art nouveau, vector illustration.'\\n"
            f"Another example: 'Abstract digital artwork depicting a swirling nebula of cosmic dust in shades of teal, gold, and magenta, with glowing star-like particles, a sense of vastness and ethereal beauty, photorealistic.'\\n"
            f"\\nFINAL PROMPT FOR IMAGE GENERATION MODEL (Max {max_length} chars):"
        )

        try:
            logger.info("Sending design creation prompt generation request to Gemini.")
            
            # Using a different model for this if needed, e.g. one better for creative text generation
            # For now, using the same as configured for the client.
            final_response = await asyncio.to_thread(
                self.model.generate_content,
                [master_prompt_system_instruction],
                generation_config=genai.types.GenerationConfig(candidate_count=1, temperature=0.7) # Slightly higher temp for creativity
            )
            generated_prompt = final_response.text.strip()
            
            if len(generated_prompt) > max_length:
                logger.warning(f"Gemini generated design prompt exceeded max_length ({len(generated_prompt)} > {max_length}). Truncating.")
                # Simple truncation, could be smarter (e.g. sentence aware)
                generated_prompt = generated_prompt[:max_length-3] + "..." 
            
            logger.info(f"User Directive for Design: {user_directive}")
            logger.info(f"Gemini Generated Design Creation Prompt: {generated_prompt}")
            return generated_prompt
        except Exception as e:
            logger.error(f"Error generating design creation prompt with Gemini: {e}", exc_info=True)
            logger.warning("Falling back to user's directive due to Gemini API error in design prompt generation.")
            return user_directive if user_directive else "abstract colorful pattern"

    # Old enhance_prompt might be deprecated or adapted if pure text enhancement is still needed elsewhere
    # async def enhance_prompt( ... ) -> Optional[str]: ... 

__all__ = ['GeminiClient'] 