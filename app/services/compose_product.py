import asyncio
from pathlib import Path
from typing import Optional
from src.workflow.orchestrator import ImageWorkflow
from src.utils.logger import logger

workflow = ImageWorkflow()

async def compose_product_workflow(
    product_image_path: Optional[Path],
    design_image_path: Optional[Path],
    color_image_path: Optional[Path],
    prompt: Optional[str] = None,
    workflow_type: str = "full_composition"
):
    """
    Compose a new product design using different combinations of images and an optional prompt.
    Returns the result dict from the orchestrator.
    
    Workflow types:
    - full_composition: Uses all three images (product, design, color)
    - product_color: Uses product and color images, keeps original design
    - product_design: Uses product and design images, design image used for both color and design
    - color_design: Uses color and design images, requires prompt for product description
    """
    try:
        # Validate inputs based on workflow type
        if workflow_type == "full_composition":
            if not all([product_image_path, design_image_path, color_image_path]):
                raise ValueError("Full composition requires all three images")
        elif workflow_type == "product_color":
            if not all([product_image_path, color_image_path]):
                raise ValueError("Product+Color workflow requires both product and color images")
        elif workflow_type == "product_design":
            if not all([product_image_path, design_image_path]):
                raise ValueError("Product+Design workflow requires both product and design images")
        elif workflow_type == "color_design":
            if not all([color_image_path, design_image_path, prompt]):
                raise ValueError("Color+Design workflow requires both images and a prompt")
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Generate appropriate prompt based on workflow type
        if not prompt:
            if workflow_type == "full_composition":
                prompt = "Compose a new design using the provided product, design, and color inspirations. Maintain the original product form and structure."
            elif workflow_type == "product_color":
                prompt = "Apply the color palette to the product while maintaining its original design and structure."
            elif workflow_type == "product_design":
                prompt = "Apply the design's colors and patterns to the product while maintaining its form and structure."
            else:
                raise ValueError("Prompt is required for color+design workflow")

        # Call the workflow with appropriate parameters
        result = await workflow.generate_or_edit_image(
            product_image_path=product_image_path,
            color_image_path=color_image_path,
            design_image_path=design_image_path,
            prompt=prompt
        )
        return result
    except Exception as e:
        logger.error(f"Error in compose_product_workflow: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)} 