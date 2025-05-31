from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import asyncio
from ..utils.logger import logger
from ..utils.config import settings
from .openai_client import OpenAIClient

class ModelSelectorClient:
    """Client that uses only OpenAI (GPT-4.1/ChatGPT) for vision and prompt enhancement tasks."""
    def __init__(self):
        self.openai_client = OpenAIClient()
        self._last_used_model: Optional[str] = "gpt-4.1"
        logger.info("ModelSelectorClient initialized with OpenAI (GPT-4.1) client only")

    @property
    def last_used_model(self) -> Optional[str]:
        return self._last_used_model

    async def generate_structured_edit_prompt(
        self,
        user_core_prompt: str,
        product_image_path: Path,
        color_image_path: Optional[Path],
        design_image_path: Optional[Path],
        max_length: int = 4000
    ) -> str:
        """Generate a structured edit prompt using OpenAI (GPT-4.1)."""
        try:
            # Analyze each image with a role-specific prompt
            descriptions = {"product": None, "color": None, "design": None}
            # Product image: form/structure
            if product_image_path and product_image_path.exists():
                try:
                    descriptions["product"] = (await self.openai_client.analyze_images(
                        images=[product_image_path.read_bytes()],
                        prompt="Describe only the shape, form, and structure of the product in this image. Do not mention color or design patterns."
                    ))["analysis"]
                except Exception as e:
                    descriptions["product"] = "Could not describe product form."
            # Color image: color palette
            if color_image_path and color_image_path.exists():
                try:
                    descriptions["color"] = (await self.openai_client.analyze_images(
                        images=[color_image_path.read_bytes()],
                        prompt="Extract and describe only the color palette from this image. List the main colors and their relationships. Do not mention objects or patterns."
                    ))["analysis"]
                except Exception as e:
                    descriptions["color"] = "Could not describe color palette."
            # Design image: pattern/design
            if design_image_path and design_image_path.exists():
                try:
                    descriptions["design"] = (await self.openai_client.analyze_images(
                        images=[design_image_path.read_bytes()],
                        prompt="Extract and describe only the pattern or design elements from this image. Do not mention color or object form."
                    ))["analysis"]
                except Exception as e:
                    descriptions["design"] = "Could not describe design pattern."
            # Compose the final prompt
            prompt_parts = []
            if descriptions["product"]:
                prompt_parts.append(f"Product form/structure: {descriptions['product']}")
            if descriptions["color"]:
                prompt_parts.append(f"Color palette: {descriptions['color']}")
            if descriptions["design"]:
                prompt_parts.append(f"Design/pattern: {descriptions['design']}")
            prompt_parts.append(f"User instruction: {user_core_prompt}")
            final_prompt = "\n".join(prompt_parts)
            if len(final_prompt) > max_length:
                final_prompt = final_prompt[:max_length-3] + "..."
            return final_prompt
        except Exception as e:
            logger.error(f"Error in OpenAI prompt generation: {str(e)}", exc_info=True)
            return user_core_prompt

    async def generate_design_creation_prompt(
        self,
        image_path1: Optional[Path],
        image_path2: Optional[Path],
        image_path3: Optional[Path],
        user_directive: str,
        max_length: int = 1000
    ) -> str:
        """Generate a design creation prompt using OpenAI (GPT-4.1)."""
        try:
            valid_images = []
            for img_path in [image_path1, image_path2, image_path3]:
                if img_path and img_path.exists():
                    with open(img_path, "rb") as f:
                        valid_images.append(f.read())
            if not valid_images and not user_directive:
                return "abstract colorful pattern"
            analysis_prompt = (
                "Analyze these design inspiration images and create a detailed prompt "
                "for generating a new design. Focus on:\n"
                "1. Visual elements and patterns\n"
                "2. Color schemes and palettes\n"
                "3. Artistic style and mood\n"
                f"User's directive: {user_directive}"
            )
            analysis_result = await self.openai_client.analyze_images(
                images=valid_images,
                prompt=analysis_prompt
            )
            if analysis_result and "analysis" in analysis_result:
                final_prompt = (
                    f"Create a new design inspired by: {analysis_result['analysis']}\n"
                    f"Following the directive: {user_directive}\n"
                    f"The design should be a standalone piece suitable for use as a pattern or texture."
                )
                if len(final_prompt) > max_length:
                    final_prompt = final_prompt[:max_length - 3] + "..."
                return final_prompt
            return user_directive if user_directive else "abstract colorful pattern"
        except Exception as e:
            logger.error(f"Error in OpenAI design prompt generation: {str(e)}", exc_info=True)
            return user_directive if user_directive else "abstract colorful pattern"

__all__ = ['ModelSelectorClient'] 