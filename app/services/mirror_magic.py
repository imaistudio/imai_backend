import os
from pathlib import Path
from src.api.openai_client import OpenAIClient
from src.api.claude_client import ClaudeClient
from src.workflow.orchestrator import ImageWorkflow
import logging
from src.api.model_selector_client import ModelSelectorClient
from datetime import datetime
from src.utils.image_processor import ImageProcessor
from src.utils.config import settings

async def mirror_magic_image(input_image: str, prompt: str = None) -> dict:
    # (1) Pass the input image into a GPT-4 vision model (via OpenAIClient.analyze_images) to "caption" it.
    vision_client = OpenAIClient()
    analysis = await vision_client.analyze_images(input_image)
    # (You can log or use "analysis" to extract a caption.)

    # (2) Pass the caption (analysis) into Claude (via ClaudeClient.generate_prompt) for "enhancement" (i.e. to produce an "enhanced prompt").
    claude_client = ClaudeClient()
    enhanced_prompt = claude_client.generate_prompt("generate_image", {"prompt": analysis})

    # (3) (If a prompt is provided, you can use it as an extra "guide" for the remix.)
    remix_prompt = prompt or enhanced_prompt

    # (4) Then pass the input image and the enhanced prompt (using orchestrator.generate_image_from_text_only) into an image-gen-1 model (or equivalent) for generation.
    remix_result = await generate_image_from_text_only(prompt=remix_prompt, input_image=input_image)

    # (Later, you can decide to return a URL or the image file directly.)
    return {"output_image": remix_result["output_image"], "note": "Mirror magic applied (remixed)."}

async def mirror_magic_remix(input_image: str, prompt: str = None) -> dict:
    """
    Analyze the input image with GPT-4.1, enhance the caption with Claude Sonnet 4,
    and remix the image with gpt-image-1 using the enhanced prompt and the original image.
    """
    vision_client = OpenAIClient()
    analysis = await vision_client.analyze_images(input_image)
    claude_client = ClaudeClient()
    enhanced_prompt = claude_client.generate_prompt("generate_image", {"prompt": analysis})
    remix_prompt = prompt or enhanced_prompt
    workflow = ImageWorkflow()
    result = await workflow.generate_or_edit_image(
        product_image_path=Path(input_image),
        color_image_path=None,
        design_image_path=None,
        prompt=remix_prompt
    )
    # Log the full result for debugging
    logging.getLogger("mirror_magic").info(f"Mirror magic remix result: {result}")
    # Improved error handling
    if result.get("status") != "success":
        return {
            "error": result.get("error", "Unknown error in mirror magic remix."),
            "note": "Mirror magic remix failed.",
            "raw_result": result
        }
    return {
        "output_image": result["generated_images"][0] if result.get("generated_images") else None,
        "metadata_path": result.get("metadata_path"),
        "note": "Mirror magic remix: GPT-4.1 vision → Claude Sonnet 4 → gpt-image-1 (image+prompt)"
    }

async def black_mirror_workflow(input_image: str, prompt: str = None) -> dict:
    """
    Black Mirror workflow: takes a single image and a prompt, enhances the prompt, and remixes the image using OpenAI's image editing API.
    """
    result = {
        "status": "error",
        "input_image": input_image,
        "prompt": prompt,
        "error": None
    }
    try:
        # Validate image
        image_path = Path(input_image)
        image_processor = ImageProcessor()
        if not image_processor.validate_image(image_path):
            result["error"] = f"Input image {input_image} is invalid or not found."
            return result

        # Enhance the prompt using ModelSelectorClient (Claude/Gemini/ChatGPT)
        model_selector = ModelSelectorClient()
        enhanced_prompt = await model_selector.generate_structured_edit_prompt(
            user_core_prompt=prompt or "Remix this image in a unique, creative way.",
            product_image_path=image_path,
            color_image_path=None,
            design_image_path=None,
            max_length=4000
        )
        result["enhanced_prompt"] = enhanced_prompt

        # Call OpenAI to edit/remix the image
        openai_client = OpenAIClient()
        generated_b64_images = await openai_client.edit_images_with_prompt(
            prompt=enhanced_prompt,
            image_paths=[image_path],
            mask_path=None,
            size="1024x1024",
            quality="high",
            n=1
        )
        if not generated_b64_images:
            result["error"] = "OpenAI client did not return any image data."
            return result

        # Save the generated image
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = settings.output.output_dir / f"black_mirror_{run_timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_extension = "png"
        img_output_path = output_dir / f"black_mirror_{run_timestamp}.png"
        workflow = ImageWorkflow()
        await workflow._save_b64_image(generated_b64_images[0], img_output_path)
        result.update({
            "status": "success",
            "output_image": str(img_output_path),
            "output_dir": str(output_dir)
        })
        return result
    except Exception as e:
        result["error"] = str(e)
        return result 