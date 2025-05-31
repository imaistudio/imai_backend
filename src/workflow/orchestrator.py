from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
import json
import aiofiles
import base64 # For decoding base64
from PIL import Image # Keep for other uses if any, or ImageProcessor handles it.
import subprocess
import requests
import os

from ..api.openai_client import OpenAIClient
from ..api.model_selector_client import ModelSelectorClient
from ..utils.image_processor import ImageProcessor
from ..utils.config import settings
from ..utils.logger import logger
from ..utils.fal_client import fal_client, upload_to_cloudinary

class ImageWorkflow:
    """Main workflow orchestrator for structured image editing."""
    
    def __init__(self):
        """Initialize the workflow with required clients and processors."""
        self.openai_client = OpenAIClient()
        self.model_selector = ModelSelectorClient()
        self.image_processor = ImageProcessor()
        logger.info("ImageWorkflow initialized with OpenAI client and ModelSelector client")
    
    async def _save_b64_image(self, b64_data: str, output_path: Path) -> bool:
        """
        Decode base64 image data and save it to a file.
        
        Args:
            b64_data: Base64 encoded image string
            output_path: Path to save the image
            
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_bytes = base64.b64decode(b64_data)
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(img_bytes)
            logger.info(f"Successfully saved base64 image to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving base64 image {output_path}: {str(e)}", exc_info=True)
            return False

    async def _save_metadata(
        self,
        metadata: Dict[str, Any],
        output_dir: Path
    ) -> Optional[Path]:
        """
        Save workflow metadata to a JSON file.
        
        Args:
            metadata: Workflow metadata
            output_dir: Output directory
            
        Returns:
            Optional[Path]: Path to metadata file or None if saving fails
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for more uniqueness
            metadata_path = output_dir / f"metadata_{timestamp}.json"
            
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            logger.info(f"Successfully saved metadata to {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}", exc_info=True)
            return None
    
    async def generate_or_edit_image(
        self,
        product_image_path: Path, # Mandatory
        color_image_path: Optional[Path],
        design_image_path: Optional[Path],
        prompt: str, # User's high-level goal / core request
        output_dir: Optional[Path] = None,
        size: str = "1024x1024",
        quality: str = "high", 
        n: int = 1,
        enhance_prompt_max_length: int = 4000,
        output_image_format_for_saving: str = "png", 
        mask_path: Optional[Path] = None,
        skip_analysis: bool = False  # New parameter to skip initial analysis
    ) -> Dict[str, Any]:
        """
        Orchestrates structured image editing.
        """
        # Initialize result with default error state
        result = {
            "status": "error",
            "output_dir": str(output_dir or settings.output.output_dir),
            "original_prompt": prompt,
            "enhanced_prompt": prompt,  # Default to original prompt
            "error": None,  # Will be set if an error occurs
            "model_used": None  # Add field to track which model was used
        }

        try:
            output_dir = output_dir or settings.output.output_dir
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = output_dir / run_timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Update output_dir in result
            result["output_dir"] = str(run_dir)

            # Validate mandatory product image
            if not self.image_processor.validate_image(product_image_path):
                error_msg = f"Product Image {product_image_path} is invalid or not found."
                logger.error(error_msg)
                result["error"] = error_msg
                return result
            
            # Validate optional images if provided
            valid_color_image_path = color_image_path if color_image_path and self.image_processor.validate_image(color_image_path) else None
            if color_image_path and not valid_color_image_path:
                logger.warning(f"Color Image at {color_image_path} failed validation or not found; will proceed without it.")
            
            valid_design_image_path = design_image_path if design_image_path and self.image_processor.validate_image(design_image_path) else None
            if design_image_path and not valid_design_image_path:
                logger.warning(f"Design Image at {design_image_path} failed validation or not found; will proceed without it.")

            # Prepare list of images for OpenAI, product image first
            openai_image_inputs = [product_image_path]
            if valid_color_image_path: openai_image_inputs.append(valid_color_image_path)
            if valid_design_image_path: openai_image_inputs.append(valid_design_image_path)

            # Use ModelSelectorClient for prompt enhancement
            enhanced_prompt = await self.model_selector.generate_structured_edit_prompt(
                user_core_prompt=prompt,
                product_image_path=product_image_path,
                color_image_path=valid_color_image_path,
                design_image_path=valid_design_image_path,
                max_length=enhance_prompt_max_length
            )
            
            # Store which model was used (for developer reference)
            result["model_used"] = self.model_selector.last_used_model

            generated_b64_images = await self.openai_client.edit_images_with_prompt(
                prompt=enhanced_prompt,  # Use the enhanced prompt
                image_paths=openai_image_inputs, 
                mask_path=mask_path,
                size=size,
                quality=quality,
                n=n
            )
            
            if not generated_b64_images:
                raise ValueError("OpenAI client did not return any image data from /edits endpoint.")
            
            saved_image_paths: List[Path] = []
            for i, b64_img_data in enumerate(generated_b64_images):
                file_extension = output_image_format_for_saving.lower()
                if file_extension == "jpeg": file_extension = "jpg"
                img_output_path = run_dir / f"edited_product_{i+1}.{file_extension}"
                if await self._save_b64_image(b64_img_data, img_output_path):
                    saved_image_paths.append(img_output_path)
            
            if not saved_image_paths:
                raise ValueError("Failed to save any images from OpenAI output.")

            # Only perform analysis if not skipped
            analysis_result = None
            if not skip_analysis:
                analysis_input_b64_images = []
                for p in saved_image_paths:
                    img_array = self.image_processor.load_image(p)
                    if img_array is not None:
                        encoded_str = self.image_processor.encode_image(img_array, format=output_image_format_for_saving.upper())
                        if encoded_str:
                            analysis_input_b64_images.append(encoded_str)
                
                if analysis_input_b64_images:
                    analysis_result = await self.openai_client.analyze_images(
                        images=analysis_input_b64_images,
                        prompt="Analyze the edited product image(s). Describe the new design applied, its adherence to the requested style and colors, and how well the original product form and background were preserved."
                    )
                else:
                    logger.warning("No saved images could be prepared for analysis.")
            
            metadata = {
                "run_timestamp": run_timestamp,
                "user_core_prompt": prompt,
                "product_image_used": str(product_image_path),
                "color_image_used": str(valid_color_image_path) if valid_color_image_path else None,
                "design_image_used": str(valid_design_image_path) if valid_design_image_path else None,
                "mask_image_provided": str(mask_path) if mask_path else None,
                "edited_product_images_saved": [str(p) for p in saved_image_paths],
                "parameters_to_openai": {"size": size, "quality": quality, "n": n},
                "analysis_of_output": analysis_result
            }
            metadata_path = await self._save_metadata(metadata, run_dir)
            
            # Update result with success data
            result.update({
                "status": "success",
                "generated_images": [str(p) for p in saved_image_paths],
                "metadata_path": str(metadata_path) if metadata_path else None,
                "analysis": analysis_result,
                "enhanced_prompt": enhanced_prompt  # Use the enhanced prompt
            })
            return result
            
        except Exception as e:
            error_msg = f"Error during image processing: {str(e)}"
            logger.error(f"!!! ORCHESTRATOR (generate_or_edit_image) CAUGHT EXCEPTION - {error_msg}", exc_info=True)
            result.update({
                "error": error_msg,
                "enhanced_prompt": prompt
            })
            return result

    async def elemental_design_creation_from_images(
        self,
        image_path1: Path,
        image_path2: Path,
        image_path3: Path,
        user_directive: str,
        output_dir: Optional[Path] = None,
        size: str = "1024x1024", # Default for DALL-E 3, gpt-image-1
        quality: str = "high", # Changed default from 'standard' to 'high' to match OpenAI's supported values
        style: Optional[str] = None, # 'vivid' or 'natural' for DALL-E 3
        n: int = 1, # Number of images to generate
        output_image_format_for_saving: str = "png", # For saving the b64_json output
        max_prompt_length: int = 4000 # Max length for Gemini prompt
    ) -> Dict[str, Any]:
        """
        Orchestrates the creation of a new elemental design based on three input images (elements/products) and a user directive.
        """
        # Map quality values to OpenAI supported values
        quality_mapping = {
            "standard": "high",  # Map 'standard' to 'high'
            "hd": "high",       # Map 'hd' to 'high'
            "low": "low",
            "medium": "medium",
            "high": "high",
            "auto": "auto"
        }
        
        # Use mapped quality value or default to 'high'
        mapped_quality = quality_mapping.get(quality.lower(), "high")

        # Initialize result dictionary
        result = {
            "status": "error",
            "output_dir": str(output_dir or settings.output.output_dir),
            "user_directive": user_directive,
            "error": None,
            "model_used": None
        }

        output_dir = output_dir or settings.output.output_dir
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Create a subdirectory for this specific design creation run
        run_dir_name = f"design_creation_{run_timestamp}"
        run_dir = output_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting new design creation workflow. Output will be in: {run_dir}")
        logger.info(f"User directive: {user_directive}")
        logger.info(f"Input images: {image_path1}, {image_path2}, {image_path3}")

        # Validate input images
        validated_image_paths: List[Path] = []
        for i, img_path in enumerate([image_path1, image_path2, image_path3]):
            if img_path and self.image_processor.validate_image(img_path):
                validated_image_paths.append(img_path)
                logger.info(f"Image {i+1} ({img_path}) validated successfully.")
            else:
                logger.error(f"Input image {i+1} ({img_path}) is invalid or not found. This image is mandatory for design creation.")
                result["error"] = f"Input image {i+1} ({img_path}) is invalid or not found."
                result["output_dir"] = str(run_dir)
                return result
        
        if len(validated_image_paths) != 3:
            # This case should ideally be caught by the individual checks above,
            # but as a safeguard:
            logger.error("Not all three mandatory input images were provided or validated.")
            result["error"] = "All three input images are mandatory and must be valid."
            result["output_dir"] = str(run_dir)
            return result

        # Use ModelSelectorClient for design creation prompt
        generation_prompt = await self.model_selector.generate_design_creation_prompt(
            user_directive=user_directive,
            image_path1=validated_image_paths[0],
            image_path2=validated_image_paths[1],
            image_path3=validated_image_paths[2],
            max_length=max_prompt_length
        )
        
        # Store which model was used (for developer reference)
        result["model_used"] = self.model_selector.last_used_model

        try:
            logger.info(f"Final prompt for OpenAI /images/generations: {generation_prompt[:300]}...")

            # Call OpenAI to generate the image using gpt-image-1 (or as configured in openai_client)
            generated_b64_images = await self.openai_client.generate_image_from_text(
                prompt=generation_prompt,
                size=size,
                quality=mapped_quality,  # Use the mapped quality value
                style=style,
                n=n
            )

            if not generated_b64_images:
                raise ValueError("OpenAI client did not return any image data from /images/generations endpoint.")

            saved_image_paths: List[Path] = []
            for i, b64_img_data in enumerate(generated_b64_images):
                file_extension = output_image_format_for_saving.lower()
                if file_extension == "jpeg": file_extension = "jpg" # common practice
                # Save with a clear name indicating it's a created design
                img_output_path = run_dir / f"created_design_{run_timestamp}_{i+1}.{file_extension}"
                if await self._save_b64_image(b64_img_data, img_output_path):
                    saved_image_paths.append(img_output_path)
                else:
                    logger.error(f"Failed to save generated design image to {img_output_path}")
            
            if not saved_image_paths:
                raise ValueError("Failed to save any images from OpenAI generation output.")

            # Metadata saving
            metadata = {
                "run_timestamp": run_timestamp,
                "workflow_type": "Design Creation",
                "user_directive": user_directive,
                "model_used": self.model_selector.last_used_model,
                "gemini_generated_prompt": generation_prompt,
                "input_image_paths": [str(p) for p in validated_image_paths],
                "openai_parameters": {"size": size, "quality": quality, "style": style, "n": n},
                "generated_design_paths": [str(p) for p in saved_image_paths],
                "output_directory": str(run_dir)
            }
            metadata_path = await self._save_metadata(metadata, run_dir)
            
            logger.info(f"Design creation workflow completed successfully. Output(s) at: {saved_image_paths}")
            return {
                "status": "success",
                "output_dir": str(run_dir),
                "created_designs": [str(p) for p in saved_image_paths], # Key for Gradio to pick up
                "metadata_path": str(metadata_path) if metadata_path else None,
                "user_directive": user_directive,
                "final_prompt_to_openai": generation_prompt
            }

        except Exception as e:
            logger.error(f"!!! ORCHESTRATOR (elemental_design_creation_from_images) CAUGHT EXCEPTION - repr(e): {repr(e)}, str(e): {str(e)} !!!", exc_info=True)
            current_final_prompt = generation_prompt if 'generation_prompt' in locals() and generation_prompt else user_directive
            return {
                "status": "error", 
                "error": str(e), 
                "output_dir": str(run_dir),
                "user_directive": user_directive,
                "final_prompt_to_openai": current_final_prompt
            }

    async def generate_image_from_text_only(
        self,
        prompt: str,
        output_dir: Optional[Path] = None,
        size: str = "1024x1024",
        quality: str = "high",
        n: int = 1,
        output_image_format_for_saving: str = "png",
        skip_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Generate images directly from text prompt without any input images.
        Uses OpenAI's gpt-image-1 model for generation.
        """
        # Initialize result with default error state
        result = {
            "status": "error",
            "output_dir": str(output_dir or settings.output.output_dir),
            "original_prompt": prompt,
            "error": None
        }

        try:
            # Create output directory
            output_dir = output_dir or settings.output.output_dir
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = output_dir / f"text_to_image_{run_timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Update output_dir in result
            result["output_dir"] = str(run_dir)

            # Validate prompt
            if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
                raise ValueError("Prompt must be a non-empty string")

            # Generate images using OpenAI
            try:
                logger.info(f"Requesting image generation with prompt: '{prompt[:100]}...'")
                generated_b64_images = await self.openai_client.generate_image_from_text(
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=n
                )
                
                if not generated_b64_images or not isinstance(generated_b64_images, list):
                    raise ValueError("Invalid response from OpenAI API: no images generated")
                
                logger.info(f"Successfully received {len(generated_b64_images)} images from OpenAI")
                
            except Exception as openai_error:
                logger.error(f"OpenAI API error during image generation: {str(openai_error)}", exc_info=True)
                raise ValueError(f"Failed to generate images: {str(openai_error)}")
            
            # Save generated images
            saved_image_paths: List[Path] = []
            for i, b64_img_data in enumerate(generated_b64_images):
                if not isinstance(b64_img_data, str):
                    logger.warning(f"Skipping invalid image data for image {i+1}")
                    continue
                    
                try:
                    file_extension = output_image_format_for_saving.lower()
                    if file_extension == "jpeg": file_extension = "jpg"
                    img_output_path = run_dir / f"generated_image_{i+1}.{file_extension}"
                    if await self._save_b64_image(b64_img_data, img_output_path):
                        saved_image_paths.append(img_output_path)
                        logger.info(f"Successfully saved image {i+1} to {img_output_path}")
                    else:
                        logger.warning(f"Failed to save image {i+1}")
                except Exception as save_error:
                    logger.error(f"Error saving image {i+1}: {str(save_error)}", exc_info=True)
            
            if not saved_image_paths:
                raise ValueError("Failed to save any of the generated images")

            # Optional post-generation analysis
            analysis_result = None
            if not skip_analysis and saved_image_paths:
                try:
                    analysis_input_b64_images = []
                    for p in saved_image_paths:
                        try:
                            img_array = self.image_processor.load_image(p)
                            if img_array is not None:
                                encoded_str = self.image_processor.encode_image(img_array, format=output_image_format_for_saving.upper())
                                if encoded_str:
                                    analysis_input_b64_images.append(encoded_str)
                        except Exception as img_error:
                            logger.error(f"Error processing image {p} for analysis: {str(img_error)}", exc_info=True)
                    
                    if analysis_input_b64_images:
                        analysis_result = await self.openai_client.analyze_images(
                            images=analysis_input_b64_images,
                            prompt="Analyze the generated image(s). Describe how well they match the original prompt, their visual quality, and any notable features or elements."
                        )
                        logger.info("Successfully completed image analysis")
                    else:
                        logger.warning("No images could be prepared for analysis")
                except Exception as analysis_error:
                    logger.error(f"Error during image analysis: {str(analysis_error)}", exc_info=True)
                    analysis_result = {"error": "Analysis failed", "details": str(analysis_error)}

            # Save metadata
            try:
                metadata = {
                    "run_timestamp": run_timestamp,
                    "workflow_type": "Text-to-Image Generation",
                    "original_prompt": prompt,
                    "generated_images": [str(p) for p in saved_image_paths],
                    "parameters": {
                        "size": size,
                        "quality": quality,
                        "n": n,
                        "output_format": output_image_format_for_saving
                    },
                    "analysis": analysis_result
                }
                metadata_path = await self._save_metadata(metadata, run_dir)
                logger.info(f"Successfully saved metadata to {metadata_path}")
            except Exception as metadata_error:
                logger.error(f"Error saving metadata: {str(metadata_error)}", exc_info=True)
                metadata_path = None
            
            # Update result with success data
            result.update({
                "status": "success",
                "generated_images": [str(p) for p in saved_image_paths],
                "metadata_path": str(metadata_path) if metadata_path else None,
                "analysis": analysis_result
            })
            logger.info("Text-to-image generation completed successfully")
            return result
            
        except ValueError as ve:
            # Handle validation and expected errors
            error_msg = str(ve)
            logger.error(f"Validation error in text-to-image generation: {error_msg}")
            result.update({
                "error": error_msg,
                "status": "error"
            })
            return result
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during text-to-image generation: {str(e)}"
            logger.error(f"!!! ORCHESTRATOR (generate_image_from_text_only) CAUGHT EXCEPTION - {error_msg}", exc_info=True)
            result.update({
                "error": error_msg,
                "status": "error"
            })
            return result

    async def analyze_image(
        self,
        image_path: Path,
        analysis_prompt: str = "Provide a detailed analysis of this image, including composition, colors, style, and any notable elements.",
        output_dir: Optional[Path] = None,
        detail_level: str = "high"
    ) -> Dict[str, Any]:
        """
        Analyze a single image using image-gen-1 to provide detailed insights.
        
        Args:
            image_path: Path to the image to analyze
            analysis_prompt: Custom prompt for the analysis
            output_dir: Optional output directory
            detail_level: Level of detail for analysis (high, medium, low)
            
        Returns:
            Dict containing analysis results and metadata
        """
        # Initialize result with default error state
        result = {
            "status": "error",
            "output_dir": str(output_dir or settings.output.output_dir),
            "image_path": str(image_path),
            "analysis_prompt": analysis_prompt,
            "error": None
        }

        try:
            # Create output directory
            output_dir = output_dir or settings.output.output_dir
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = output_dir / f"image_analysis_{run_timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Update output_dir in result
            result["output_dir"] = str(run_dir)

            # Validate input image
            if not self.image_processor.validate_image(image_path):
                raise ValueError(f"Invalid or missing image at {image_path}")

            # Load and encode image for analysis
            img_array = self.image_processor.load_image(image_path)
            if img_array is None:
                raise ValueError(f"Failed to load image at {image_path}")

            encoded_image = self.image_processor.encode_image(img_array)
            if not encoded_image:
                raise ValueError(f"Failed to encode image at {image_path}")

            # Get analysis from image-gen-1
            analysis_result = await self.openai_client.analyze_images_with_image_gen(
                images=[encoded_image],
                prompt=analysis_prompt,
                model="image-gen-1"  # Specify image-gen-1 model
            )

            # Save metadata
            metadata = {
                "run_timestamp": run_timestamp,
                "workflow_type": "Image Analysis",
                "model_used": "image-gen-1",
                "image_path": str(image_path),
                "analysis_prompt": analysis_prompt,
                "detail_level": detail_level,
                "analysis_result": analysis_result
            }
            metadata_path = await self._save_metadata(metadata, run_dir)

            # Update result with success data
            result.update({
                "status": "success",
                "analysis": analysis_result,
                "metadata_path": str(metadata_path) if metadata_path else None,
                "model_used": "image-gen-1"
            })
            return result

        except Exception as e:
            error_msg = f"Error during image analysis: {str(e)}"
            logger.error(f"!!! ORCHESTRATOR (analyze_image) CAUGHT EXCEPTION - {error_msg}", exc_info=True)
            result.update({
                "error": error_msg,
                "status": "error"
            })
            return result

    async def compare_images(
        self,
        image_paths: List[Path],
        comparison_prompt: str = "Compare these images, highlighting similarities and differences in style, composition, colors, and key elements.",
        output_dir: Optional[Path] = None,
        detail_level: str = "high"
    ) -> Dict[str, Any]:
        """
        Compare multiple images using image-gen-1 to identify similarities and differences.
        
        Args:
            image_paths: List of paths to images to compare (2-3 images)
            comparison_prompt: Custom prompt for the comparison
            output_dir: Optional output directory
            detail_level: Level of detail for comparison (high, medium, low)
            
        Returns:
            Dict containing comparison results and metadata
        """
        # Initialize result with default error state
        result = {
            "status": "error",
            "output_dir": str(output_dir or settings.output.output_dir),
            "image_paths": [str(p) for p in image_paths],
            "comparison_prompt": comparison_prompt,
            "error": None
        }

        try:
            # Validate number of images
            if len(image_paths) < 2 or len(image_paths) > 3:
                raise ValueError("Image comparison requires 2-3 images")

            # Create output directory
            output_dir = output_dir or settings.output.output_dir
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = output_dir / f"image_comparison_{run_timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Update output_dir in result
            result["output_dir"] = str(run_dir)

            # Validate and encode all images
            encoded_images = []
            for img_path in image_paths:
                if not self.image_processor.validate_image(img_path):
                    raise ValueError(f"Invalid or missing image at {img_path}")

                img_array = self.image_processor.load_image(img_path)
                if img_array is None:
                    raise ValueError(f"Failed to load image at {img_path}")

                encoded = self.image_processor.encode_image(img_array)
                if not encoded:
                    raise ValueError(f"Failed to encode image at {img_path}")

                encoded_images.append(encoded)

            # Get comparison from image-gen-1
            comparison_result = await self.openai_client.analyze_images_with_image_gen(
                images=encoded_images,
                prompt=comparison_prompt,
                model="image-gen-1"  # Specify image-gen-1 model
            )

            # Save metadata
            metadata = {
                "run_timestamp": run_timestamp,
                "workflow_type": "Image Comparison",
                "model_used": "image-gen-1",
                "image_paths": [str(p) for p in image_paths],
                "comparison_prompt": comparison_prompt,
                "detail_level": detail_level,
                "comparison_result": comparison_result
            }
            metadata_path = await self._save_metadata(metadata, run_dir)

            # Update result with success data
            result.update({
                "status": "success",
                "comparison": comparison_result,
                "metadata_path": str(metadata_path) if metadata_path else None,
                "model_used": "image-gen-1"
            })
            return result

        except Exception as e:
            error_msg = f"Error during image comparison: {str(e)}"
            logger.error(f"!!! ORCHESTRATOR (compare_images) CAUGHT EXCEPTION - {error_msg}", exc_info=True)
            result.update({
                "error": error_msg,
                "status": "error"
            })
            return result

    async def reframe_image(self, image_path: Path, image_size: str = None) -> Dict[str, Any]:
        """
        Reframe an image using FAL.ai's reframe service.

        image_size parameter documentation:
        - Controls the output aspect ratio and resolution for the FAL Ideogram V3 Reframe API.
        - Default: If not specified, uses 'square_hd' (square, high-definition output).
        - User override: If provided (via API context or parameters), that value is used for the FAL request.
        - Supported values (as per FAL docs):
            'square_hd', 'square', 'portrait_4_3', 'portrait_16_9', 'landscape_4_3', 'landscape_16_9',
            or a custom object: {"width": 1280, "height": 720}
        - The value is passed through: API request → intent analysis → workflow → FAL API.

        Args:
            image_path: Path to the image file to reframe
            image_size: Optional image size (e.g., 'square_hd', 'landscape_16_9')
        Returns:
            Dict containing the reframed image data
        """
        try:
            # Upload image to Cloudinary first
            cloudinary_result = upload_to_cloudinary(image_path)
            image_url = cloudinary_result["secure_url"]
            # Use provided image_size or default to 'square_hd'
            # image_size can be: 'square_hd', 'square', 'portrait_4_3', 'portrait_16_9', 'landscape_4_3', 'landscape_16_9', or a custom object
            options = {"image_size": image_size or "square_hd"}
            # Process with FAL.ai reframe
            result = await fal_client.reframe_image(image_url, options=options)
            # Download the result
            output_path = self.download_image(
                result["url"],
                f"reframed_{image_path.stem}.png"
            )
            out = {
                "status": "success",
                "message": "Image successfully reframed",
                "original_image": str(image_path),
                "reframed_image": str(output_path),
                "reframed_url": result["url"]
            }
            if "request_id" in result:
                out["request_id"] = result["request_id"]
            return out
        except Exception as e:
            logger.error(f"Error in reframe workflow: {str(e)}")
            raise
    
    async def aurasr_upscale(self, image_path: Path) -> Dict[str, Any]:
        """
        Upscale an image using FAL.ai's AuraSR service.
        
        Args:
            image_path: Path to the image file to upscale
            
        Returns:
            Dict containing the upscaled image data
        """
        try:
            # Upload image to Cloudinary first
            cloudinary_result = upload_to_cloudinary(image_path)
            image_url = cloudinary_result["secure_url"]
            
            # Process with FAL.ai upscaling
            result = await fal_client.upscale_image(image_url)
            
            # Download the result
            output_path = self.download_image(
                result["url"],
                f"upscaled_{image_path.stem}.png"
            )
            
            out = {
                "status": "success",
                "message": "Image successfully upscaled",
                "original_image": str(image_path),
                "upscaled_image": str(output_path),
                "upscaled_url": result["url"],
                "dimensions": {
                    "width": result["width"],
                    "height": result["height"]
                },
                "file_size": result["file_size"]
            }
            if "request_id" in result:
                out["request_id"] = result["request_id"]
            return out
            
        except Exception as e:
            logger.error(f"Error in upscale workflow: {str(e)}")
            raise

    def download_image(self, url: str, output_path: str = None) -> str:
        """
        Download an image from a URL and save it to output_path (or a temp file if not provided).
        Returns the path to the downloaded file.
        """
        if output_path is None:
            import tempfile
            fd, output_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path

# Export the class for easy access
__all__ = ['ImageWorkflow'] 