from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel
import asyncio
from pathlib import Path
import tempfile
import shutil
import json
import inspect
import subprocess
import requests as pyrequests
import os
from datetime import datetime

from src.workflow.orchestrator import ImageWorkflow
from src.api.claude_client import ClaudeClient
from src.utils.logger import logger
from src.utils.config import settings
from app.services.upscale import upscale_image
from app.services.reframe import reframe_image
from app.services.clarity_upscaler import clarity_upscale_image
from app.services.mirror_magic import mirror_magic_image, mirror_magic_remix, black_mirror_workflow
from app.services.image_processor import ImageProcessor
from app.services.ai_analyzer import AIAnalyzer
from app.services.image_generator import ImageGenerator
from app.config import Settings
from app.services.compose_product import compose_product_workflow

# Initialize the main workflow and Claude client
workflow = ImageWorkflow()
claude_client = ClaudeClient()

app = FastAPI(
    title="AI Intent Router API",
    description="Natural language intent recognition and task routing system powered by Claude 3.7 Sonnet",
    version="1.0.0"
)

class IntentRequest(BaseModel):
    """Base model for intent recognition requests."""
    user_input: str
    context: Optional[Dict[str, Any]] = None
    files: Optional[List[str]] = None  # List of file paths if any

class IntentResponse(BaseModel):
    """Base model for intent recognition responses."""
    intent: str
    confidence: float
    parameters: Dict[str, Any]
    suggested_actions: List[str]
    required_inputs: Optional[List[str]] = None

class PromptRequest(BaseModel):
    """Request model for the test endpoint."""
    prompt: str

@app.post("/api/v1/process-intent", response_model=IntentResponse)
async def process_intent(
    user_input: str = Form(""),  # Default to empty string
    context: Optional[str] = Form(None),
    files: List[UploadFile] = File(None)
) -> JSONResponse:
    """
    Process any user input and determine the appropriate action.
    This is the main entry point for all user interactions.
    """
    try:
        # Create temporary directory for any file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save any uploaded files
            saved_files = {}
            if files:
                for file in files:
                    file_path = temp_path / file.filename
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    saved_files[file.filename] = str(file_path)

            # Parse context if provided
            context_dict = None
            if context:
                try:
                    context_dict = json.loads(context)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON context provided: {context}")

            # If no user_input and only images, route to compose_product
            if not user_input.strip() and files and len(files) >= 2:
                # Try to map files to product, design, color by filename heuristics
                product_path = None
                design_path = None
                color_path = None
                for fname, fpath in saved_files.items():
                    lower = fname.lower()
                    if "product" in lower:
                        product_path = Path(fpath)
                    elif "design" in lower:
                        design_path = Path(fpath)
                    elif "color" in lower:
                        color_path = Path(fpath)
                # Fallback: assign by order if not matched
                paths = list(saved_files.values())
                if not product_path and len(paths) > 0:
                    product_path = Path(paths[0])
                if not design_path and len(paths) > 1:
                    design_path = Path(paths[1])
                if not color_path and len(paths) > 2:
                    color_path = Path(paths[2])
                # Compose-product default prompt
                workflow_prompt = "Compose a new design using the provided images. Maintain the original product form and structure."
                result = await compose_product_workflow(
                    product_image_path=product_path,
                    design_image_path=design_path,
                    color_image_path=color_path,
                    prompt=workflow_prompt
                )
                return JSONResponse(content={"status": "success", "result": result})

            # Analyze intent using the user input directly (no await)
            intent_analysis = analyze_intent(user_input, saved_files)
            
            # Route to appropriate workflow based on intent
            result = await route_to_workflow(intent_analysis, saved_files)
            
            return JSONResponse(content={
                "status": "success",
                "intent_analysis": intent_analysis,
                "workflow_result": result
            })
            
    except Exception as e:
        logger.error(f"Error processing intent: {str(e)}", exc_info=True)
        # Don't include the full exception in the response
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

def analyze_intent(user_input: str, files: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze user input using Claude 3.7 Sonnet to determine intent and parameters.
    """
    try:
        # If user input contains 'create design', force create_design intent
        if 'create design' in user_input.lower():
            logger.info("[INTENT] 'create design' detected in user input - using create_design intent (early return)")
            return {
                "intent": "create_design",
                "confidence": 1.0,
                "parameters": {
                    "user_directive": user_input,
                    "quality": "high"
                },
                "suggested_actions": ["specify_style", "set_quality"],
                "required_inputs": [f"image{i+1}" for i in range(len(files))] if files else []
            }
        # If three images are provided, use generate_image intent for product, color, and design images
        if files and len(files) == 3:
            logger.info(f"[INTENT] 3 images provided - using generate_image intent for product, color, and design images")
            return {
                "intent": "generate_image",
                "confidence": 1.0,
                "parameters": {
                    "description": user_input if user_input else "Generate a new design using the product, color, and design images provided.",
                    "quality": "high"
                },
                "suggested_actions": ["specify_details", "set_quality"],
                "required_inputs": []
            }
        # If two images are provided, use generate_image intent for product and color/design images
        elif files and len(files) == 2:
            logger.info(f"[INTENT] 2 images provided - using generate_image intent for product and color/design images")
            return {
                "intent": "generate_image",
                "confidence": 1.0,
                "parameters": {
                    "description": user_input if user_input else "Generate a new design using the product and color/design images provided.",
                    "quality": "high"
                },
                "suggested_actions": ["specify_details", "set_quality"],
                "required_inputs": []
            }
        # If one image is provided, use generate_image intent for product image only
        elif files and len(files) == 1:
            logger.info(f"[INTENT] 1 image provided - using generate_image intent for product image only")
            return {
                "intent": "generate_image",
                "confidence": 1.0,
                "parameters": {
                    "description": user_input if user_input else "Generate a new design using the product image provided.",
                    "quality": "high"
                },
                "suggested_actions": ["specify_details", "set_quality"],
                "required_inputs": []
            }
            
        # Add file context if available
        context = None
        if files:
            context = {
                "available_files": list(files.keys()),
                "file_count": len(files)
            }
        
        # Merge image_size from request context if present (and later into parameters)
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        request_obj = None
        for f in outer_frames:
            local_vars = f.frame.f_locals
            if "request" in local_vars and hasattr(local_vars["request"], "context"):
                request_obj = local_vars["request"]
                break
        image_size_from_context = None
        if request_obj and hasattr(request_obj, "context") and request_obj.context:
            if "image_size" in request_obj.context:
                image_size_from_context = request_obj.context["image_size"]
                if context is None:
                    context = {}
                context["image_size"] = image_size_from_context
        
        # Get intent analysis from Claude
        raw_response = claude_client.analyze_intent(user_input, context)
        
        # Parse the JSON response from Claude
        try:
            # The response might be wrapped in ```json ... ```, so clean it up
            if isinstance(raw_response, str):
                if "```json" in raw_response:
                    json_str = raw_response.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = raw_response.strip()
                intent_analysis = json.loads(json_str)
            else:
                intent_analysis = raw_response
                
            logger.info(f"Successfully parsed Claude's intent analysis: {json.dumps(intent_analysis, indent=2)}")
            
            # PATCH: Merge image_size from context into parameters if present
            if image_size_from_context:
                intent_analysis.setdefault("parameters", {})["image_size"] = image_size_from_context
            
            # Validate the intent analysis has required fields
            if not isinstance(intent_analysis, dict) or "intent" not in intent_analysis:
                raise ValueError("Invalid intent analysis format from Claude")

            # --- PATCH: Remap edit_image upscaling/enhancement to aurasr ---
            if (
                intent_analysis.get("intent") == "edit_image"
                and (
                    "upscale" in str(intent_analysis.get("parameters", {}).get("operation", "")).lower()
                    or "upscale" in str(intent_analysis.get("parameters", {}).get("enhancement_type", "")).lower()
                    or "upscale" in str(intent_analysis.get("parameters", {}).get("target", "")).lower()
                    or "upscale" in str(intent_analysis.get("parameters", {}).get("edit_type", "")).lower()
                    or "enhance" in str(intent_analysis.get("parameters", {}).get("operation", "")).lower()
                    or "enhance" in str(intent_analysis.get("parameters", {}).get("enhancement_type", "")).lower()
                    or "enhance" in str(intent_analysis.get("parameters", {}).get("target", "")).lower()
                    or "enhance" in str(intent_analysis.get("parameters", {}).get("edit_type", "")).lower()
                )
            ):
                intent_analysis["intent"] = "aurasr"
                intent_analysis["parameters"].update({
                    "user_directive": user_input,
                    "upscaling_factor": 4,
                    "overlapping_tiles": True,
                    "checkpoint": "v2"
                })
                intent_analysis["suggested_actions"] = ["upload_image", "specify_upscale_factor", "set_quality"]
                intent_analysis["required_inputs"] = ["image"]
            # --- END PATCH ---

            # --- PATCH: Remap edit_image reframing/outpainting to reframe ---
            reframe_keywords = [
                "reframe", "outpaint", "expand", "extend", "widen",
                "make wider", "make larger", "enlarge", "expand image",
                "extend image", "widen image", "make square", "square crop",
                "change aspect", "change ratio", "adjust frame", "adjust crop",
                "recompose", "recomposition", "expand canvas", "extend canvas",
                "widen canvas", "make landscape", "make portrait", "change orientation"
            ]
            if (
                intent_analysis.get("intent") == "edit_image"
                and (
                    any(kw in user_input.lower() for kw in reframe_keywords)
                    or any(kw in str(intent_analysis.get("parameters", {})).lower() for kw in reframe_keywords)
                )
            ):
                intent_analysis["intent"] = "reframe"
                params = intent_analysis["parameters"]
                params["user_directive"] = user_input
                if "image_size" not in params:
                    params["image_size"] = "square_hd"
                intent_analysis["suggested_actions"] = ["upload_image", "specify_reframe_goal", "set_output_size"]
                intent_analysis["required_inputs"] = ["image"]
            # --- END PATCH ---

            # --- PATCH: Remap video-related requests to kling ---
            kling_keywords = [
                "make a video", "generate a video", "create a video", "video from image",
                "turn into video", "animate this image", "kling", "video workflow",
                "video", "animation", "movie", "clip"
            ]
            if (
                any(kw in user_input.lower() for kw in kling_keywords)
                or any(kw in str(intent_analysis.get("parameters", {})).lower() for kw in kling_keywords)
            ):
                intent_analysis["intent"] = "kling"
                params = intent_analysis["parameters"]
                params["user_directive"] = user_input
                if "duration" not in params:
                    params["duration"] = 5
                if "aspect_ratio" not in params:
                    params["aspect_ratio"] = "1:1"
                intent_analysis["suggested_actions"] = ["upload_image", "specify_duration", "set_aspect_ratio"]
                intent_analysis["required_inputs"] = ["image"]
            # --- END PATCH ---

            # --- PATCH: Only trigger elemental_design_creation intent on exact phrase ---
            if "elemental design creation" in user_input.lower():
                intent_analysis["intent"] = "elemental_design_creation"
                params = intent_analysis["parameters"]
                params["user_directive"] = user_input
                intent_analysis["suggested_actions"] = ["upload_images", "specify_design_goal"]
                intent_analysis["required_inputs"] = ["images"]

            # If we have files and the intent requires them, validate
            if intent_analysis["intent"] == "create_design" and len(files) < 3:
                intent_analysis["suggested_actions"].append("upload_more_images")
                intent_analysis["confidence"] *= 0.8  # Reduce confidence if missing required files
            
            elif intent_analysis["intent"] == "edit_image" and not files:
                intent_analysis["suggested_actions"].append("upload_image")
                intent_analysis["confidence"] *= 0.8
            
            return intent_analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Claude's response: {str(e)}")
            logger.error(f"Raw response was: {raw_response}")
            raise ValueError(f"Failed to parse Claude's response: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in intent analysis: {str(e)}", exc_info=True)
        # Fallback to rule-based analysis
        return fallback_intent_analysis(user_input, files)

def fallback_intent_analysis(user_input: str, files: Dict[str, str]) -> Dict[str, Any]:
    """
    Fallback intent analysis using simple rule-based system.
    Used when Claude analysis fails.
    """
    input_lower = user_input.lower()
    
    # Keywords for upscaling/enhancement
    upscale_keywords = [
        "upscale", "upscaling", "enhance resolution", "enhance quality",
        "increase resolution", "improve quality", "super resolution",
        "super-resolve", "high resolution", "hi-res", "hires",
        "enhance", "enhancement", "sharpen", "sharper", "clearer",
        "higher quality", "better quality", "improve sharpness",
        "increase quality", "boost resolution", "resolution boost",
        "quality boost", "make sharper", "make clearer"
    ]
    
    # Keywords for reframing/outpainting
    reframe_keywords = [
        "reframe", "outpaint", "expand", "extend", "widen",
        "make wider", "make larger", "enlarge", "expand image",
        "extend image", "widen image", "make square", "square crop",
        "change aspect", "change ratio", "adjust frame", "adjust crop",
        "recompose", "recomposition", "expand canvas", "extend canvas",
        "widen canvas", "make landscape", "make portrait", "change orientation"
    ]
    
    # Check for upscaling intent
    if any(keyword in input_lower for keyword in upscale_keywords):
        return {
            "intent": "aurasr",
            "confidence": 0.85,  # Higher confidence for clear upscaling intent
            "parameters": {
                "user_directive": user_input,
                "upscaling_factor": 4,  # Default to 4x upscaling
                "overlapping_tiles": True,
                "checkpoint": "v2"
            },
            "suggested_actions": ["upload_image", "specify_upscale_factor", "set_quality"],
            "required_inputs": ["image"]
        }
    
    # Check for reframing intent
    elif any(keyword in input_lower for keyword in reframe_keywords):
        return {
            "intent": "reframe",
            "confidence": 0.85,  # Higher confidence for clear reframing intent
            "parameters": {
                "user_directive": user_input,
                "image_size": "square_hd"  # Default to square HD output
            },
            "suggested_actions": ["upload_image", "specify_reframe_goal", "set_output_size"],
            "required_inputs": ["image"]
        }
    
    # --- Kling keywords ---
    kling_keywords = [
        "make a video", "generate a video", "create a video", "video from image",
        "turn into video", "animate this image", "kling", "video workflow"
    ]
    if any(keyword in input_lower for keyword in kling_keywords):
        return {
            "intent": "kling",
            "confidence": 0.9,
            "parameters": {
                "user_directive": user_input,
                "prompt": user_input,  # or extract a more specific prompt if needed
                "duration": 5,         # default, or parse from user_input
                "aspect_ratio": "1:1"  # default, or parse from user_input
            },
            "suggested_actions": ["upload_image", "specify_duration", "set_aspect_ratio"],
            "required_inputs": ["image"]
        }
    
    # Only trigger elemental_design_creation on exact phrase
    elif "elemental design creation" in input_lower:
        return {
            "intent": "elemental_design_creation",
            "confidence": 0.9,
            "parameters": {
                "user_directive": user_input,
                "quality": "high"
            },
            "suggested_actions": ["upload_images", "specify_design_goal"],
            "required_inputs": ["images"]
        }
    elif "edit" in input_lower and "image" in input_lower:
        return {
            "intent": "edit_image",
            "confidence": 0.65,
            "parameters": {
                "prompt": user_input,
                "quality": "high"
            },
            "suggested_actions": ["upload_image", "specify_edits", "set_quality"],
            "required_inputs": ["image"]
        }
    elif "generate" in input_lower and "image" in input_lower:
        return {
            "intent": "generate_image",
            "confidence": 0.75,
            "parameters": {
                "prompt": user_input,
                "quality": "high"
            },
            "suggested_actions": ["specify_details", "set_quality", "set_size"],
            "required_inputs": []
        }
    elif "analyze" in input_lower and "image" in input_lower:
        return {
            "intent": "analyze_image",
            "confidence": 0.7,
            "parameters": {
                "analysis_focus": user_input,
                "detail_level": "high"
            },
            "suggested_actions": ["upload_image", "specify_analysis_focus", "set_detail_level"],
            "required_inputs": ["image"]
        }
    elif "compare" in input_lower and "image" in input_lower:
        return {
            "intent": "compare_images",
            "confidence": 0.7,
            "parameters": {
                "comparison_focus": user_input,
                "detail_level": "high"
            },
            "suggested_actions": ["upload_images", "specify_comparison_focus", "set_detail_level"],
            "required_inputs": ["image1", "image2"]
        }
    else:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "parameters": {},
            "suggested_actions": ["clarify_intent", "provide_examples"],
            "required_inputs": []
        }

async def route_to_workflow(intent_analysis: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
    """
    Route the intent to the appropriate workflow handler.
    """
    # Add debug logging to see what intent we're getting
    logger.info(f"[ROUTER] Routing workflow with intent analysis: {json.dumps(intent_analysis, indent=2)}")
    
    intent = intent_analysis.get("intent", "unknown")
    params = intent_analysis.get("parameters", {})
    
    logger.info(f"[ROUTER] Extracted intent: {intent}, parameters: {json.dumps(params, indent=2)}")
    
    try:
        # Handle unknown intent with a friendly response
        if intent == "unknown":
            logger.info("[ROUTER] Entered 'unknown' intent route.")
            # Use Claude to generate a friendly response that guides toward image generation
            system_prompt = """You are a helpful AI assistant NAMED IRIS of IMAI focused on image generation and editing. 
When users send messages that aren't related to image tasks, respond naturally but guide them toward using the image generation features.
Your response should:
1. Be friendly and conversational
2. Acknowledge their message
3. Briefly explain what this app can do (generate/edit images)
4. Ask if they'd like to try any image-related tasks
5. Provide 2-3 example prompts they could try
6. MENTION THAT YOU ARE FROM IMAI AND ARE HERE TO HELP. ALSO DO NOT MENTION THE FACT THAT YOU ARE AN AI ASSISTANT. 
Keep your response concise and engaging."""

            response = claude_client.client.messages.create(
                model=claude_client.model,
                max_tokens=500,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"User message: {params.get('user_input', '')}"
                    }
                ]
            )
            
            return {
                "status": "conversation",
                "message": response.content[0].text,
                "suggested_actions": ["generate_image", "edit_image", "create_design"]
            }
            
        # Generate a detailed prompt using Claude if needed
        if intent in ["edit_image", "generate_image", "create_design"]:
            logger.info(f"[ROUTER] Entered '{intent}' intent route.")
            # Note: generate_prompt is not async, so we don't await it
            detailed_prompt = claude_client.generate_prompt(intent, params)
            params["prompt"] = detailed_prompt
            
        if intent == "create_design":
            logger.info("[ROUTER] Executing enhanced create_design route (analyze-and-generate logic)")
            if len(files) < 1:
                raise HTTPException(
                    status_code=400,
                    detail="Design creation requires at least one inspiration image"
                )
            image_paths = list(files.values())
            try:
                settings = Settings()
                image_processor = ImageProcessor()
                ai_analyzer = AIAnalyzer()
                image_generator = ImageGenerator()

                # Ensure output directory exists
                os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                concatenated_path = os.path.join(settings.OUTPUT_DIR, f"concatenated_{timestamp}.jpg")
                image_processor.concatenate_images(image_paths, concatenated_path, direction='horizontal')

                design_elements = ai_analyzer.analyze_image(concatenated_path)

                generated_path = os.path.join(settings.OUTPUT_DIR, f"generated_{timestamp}.png")
                image_generator.generate_image(design_elements, generated_path)

                return {
                    "status": "success",
                    "design_elements": design_elements,
                    "concatenated_image": concatenated_path,
                    "generated_image": generated_path
                }
            except Exception as e:
                logger.error(f"Error in design creation: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
            
        elif intent == "edit_image":
            logger.info("[ROUTER] Executing edit_image route.")
            # Ensure params is set correctly
            params = intent_analysis.get("parameters", {})
            operation = params.get("operation") if params else None
            if operation in ("clarify", "clarity"):
                if not files or len(files) != 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Clarity upscaling requires exactly one image"
                    )
                image_path = list(files.values())[0]
                result = await clarity_upscale_image(input_image=image_path)
                return result
            else:
                raise HTTPException(status_code=400, detail="The 'edit_image' intent is currently disabled. Please use upscaling, reframing, or clarification instead.")
            
        elif intent == "generate_image":
            logger.info("[ROUTER] Executing generate_image route.")
            # For generate_image, use the description from parameters if available
            prompt = params.get("description", params.get("prompt", ""))
            if not prompt:
                raise HTTPException(
                    status_code=400,
                    detail="No description or prompt provided for image generation"
                )
            logger.info(f"Generating image with prompt: {prompt}")
            
            # REMOVE: generate_or_edit_image is now disabled
            raise HTTPException(status_code=403, detail="The generate_or_edit_image workflow is currently disabled and inaccessible.")
            
        elif intent == "analyze_image":
            logger.info("[ROUTER] Executing analyze_image route.")
            # Analyze image requires exactly one image
            if not files or len(files) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="Image analysis requires exactly one image"
                )
            
            # Get the image path
            image_path = Path(list(files.values())[0])
            
            # Use image-gen-1 to analyze the image
            analysis_prompt = params.get("analysis_focus", "Provide a detailed analysis of this image, including composition, colors, style, and any notable elements.")
            
            return await workflow.analyze_image(
                image_path=image_path,
                analysis_prompt=analysis_prompt,
                detail_level=params.get("detail_level", "high")
            )
            
        elif intent == "compare_images":
            logger.info("[ROUTER] Executing compare_images route.")
            # Compare images requires 2-3 images
            if not files or len(files) < 2 or len(files) > 3:
                raise HTTPException(
                    status_code=400,
                    detail="Image comparison requires 2-3 images"
                )
            
            # Get the image paths
            image_paths = [Path(path) for path in files.values()]
            
            # Use image-gen-1 to compare the images
            comparison_prompt = params.get("comparison_focus", "Compare these images, highlighting similarities and differences in style, composition, colors, and key elements.")
            
            return await workflow.compare_images(
                image_paths=image_paths,
                comparison_prompt=comparison_prompt,
                detail_level=params.get("detail_level", "high")
            )
            
        elif intent == "reframe":
            logger.info("[ROUTER] Executing reframe route.")
            # Reframe requires exactly one image
            if not files or len(files) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="Reframe requires exactly one image"
                )
            image_path = list(files.values())[0]
            result = await reframe_image(image_size=params.get("image_size", "square_hd"))
            return result
            
        elif intent == "aurasr":
            logger.info("[ROUTER] Executing aurasr route.")
            # AuraSR upscaling requires exactly one image
            if not files or len(files) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="AuraSR upscaling requires exactly one image"
                )
            image_path = list(files.values())[0]
            result = await upscale_image(input_image=image_path)
            return result
            
        elif intent == "kling":
            logger.info("[ROUTER] Executing kling route.")
            # Kling requires exactly one image
            if not files or len(files) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="Kling requires exactly one image"
                )
            image_path = list(files.values())[0]
            duration = str(params.get("duration", 5))
            aspect_ratio = params.get("aspect_ratio", "1:1")

            # Copy image to main_images/
            main_images_dir = Path("main_images")
            main_images_dir.mkdir(exist_ok=True)
            dest_image_path = main_images_dir / Path(image_path).name
            shutil.copy(image_path, dest_image_path)

            # Call kling.py with positional arguments
            result = subprocess.run(
                ["python", "kling.py", duration, aspect_ratio],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Kling workflow failed: {result.stderr}"
                )

            # Parse the output for video path or Cloudinary URL
            output_lines = result.stdout.strip().split("\n")
            video_path = None
            cloudinary_url = None
            for line in output_lines:
                if "Video saved at:" in line:
                    video_path = line.split("Video saved at:")[1].strip()
                if "Cloudinary URL:" in line:
                    cloudinary_url = line.split("Cloudinary URL:")[1].strip()

            if not (video_path or cloudinary_url):
                raise HTTPException(
                    status_code=500,
                    detail="Could not find video path or Cloudinary URL in Kling output"
                )

            return {
                "status": "success",
                "message": "Video generated successfully",
                "video_path": video_path,
                "cloudinary_url": cloudinary_url
            }
            
        elif intent == "elemental_design_creation":
            logger.info("[ROUTER] Executing elemental_design_creation route.")
            # Elemental design creation requires exactly 3 images
            if len(files) != 3:
                raise HTTPException(
                    status_code=400,
                    detail="Elemental design creation requires exactly three inspiration images"
                )
            # Get the three image paths
            image_paths = list(files.values())[:3]
            return await workflow.elemental_design_creation_from_images(
                image_path1=Path(image_paths[0]),
                image_path2=Path(image_paths[1]),
                image_path3=Path(image_paths[2]),
                user_directive=params.get("user_directive", params.get("prompt", "")),
                quality=params.get("quality", "high"),
                style=params.get("style")
            )
            
        else:
            logger.error(f"[ROUTER] Unknown intent received: {intent}")
            raise HTTPException(
                status_code=400,
                detail=f"Unknown intent: {intent}"
            )
            
    except Exception as e:
        logger.error(f"[ROUTER] Error in workflow routing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/v1/test-claude")
async def test_claude(request: PromptRequest) -> JSONResponse:
    """
    Simple test endpoint that sends a prompt to Claude and returns the response.
    """
    try:
        # Make a simple call to Claude - note that messages.create() is not async
        response = claude_client.client.messages.create(
            model=claude_client.model,
            max_tokens=1024,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        )
        
        # Return just the text response
        return JSONResponse(content={
            "response": response.content[0].text
        })
            
    except Exception as e:
        logger.error(f"Error in Claude test: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/black-mirror")
async def black_mirror(
    image: UploadFile = File(...),  # image is mandatory
    prompt: Optional[str] = Form(None)  # prompt is optional
):
    if not image:
        raise HTTPException(status_code=400, detail="Image is required.")
    try:
        # Save the uploaded image (for example, in a "main_images" folder)
        image_path = f"main_images/{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        # Call the new black_mirror_workflow service
        result = await black_mirror_workflow(input_image=image_path, prompt=prompt)
        logger.info(f"Black mirror result: {result}")
        if result.get("error"):
            return JSONResponse(content={"status": "error", "result": result}, status_code=500)
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as e:
        logger.error(f"Exception in black_mirror: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)

@app.post("/api/v1/compose-product")
async def compose_product(
    product: Optional[UploadFile] = File(None),
    design: Optional[UploadFile] = File(None),
    color: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form("")
):
    """
    Compose a new product design using different combinations of images:
    1. All three images (product, design, color) - Full composition
    2. Product + Color - Keep original design
    3. Product + Design - Use design for both color and design
    4. Color + Design - Requires prompt for product description
    """
    try:
        # Validate input combinations
        if not product and not design and not color:
            raise HTTPException(status_code=400, detail="At least two images must be provided")
        
        if not product and not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required when product image is not provided")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Save provided images
            product_path = None
            design_path = None
            color_path = None
            
            if product:
                product_path = temp_path / f"product_{product.filename}"
                with open(product_path, "wb") as f:
                    f.write(await product.read())
            
            if design:
                design_path = temp_path / f"design_{design.filename}"
                with open(design_path, "wb") as f:
                    f.write(await design.read())
            
            if color:
                color_path = temp_path / f"color_{color.filename}"
                with open(color_path, "wb") as f:
                    f.write(await color.read())

            # Determine the workflow type and adjust parameters
            if product and design and color:
                # Full composition - use all images
                workflow_type = "full_composition"
                workflow_prompt = prompt or "Compose a new design using the provided product, design, and color inspirations. Maintain the original product form and structure."
            elif product and color:
                # Product + Color - keep original design
                workflow_type = "product_color"
                workflow_prompt = prompt or "Apply the color palette to the product while maintaining its original design and structure."
                design_path = None  # Don't use design image
            elif product and design:
                # Product + Design - use design for both
                workflow_type = "product_design"
                workflow_prompt = prompt or "Apply the design's colors and patterns to the product while maintaining its form and structure."
                color_path = design_path  # Use design image for both color and design
            elif color and design:
                # Color + Design - requires prompt
                workflow_type = "color_design"
                if not prompt:
                    raise HTTPException(status_code=400, detail="Prompt is required when only color and design images are provided")
                workflow_prompt = prompt

            # Call the workflow service
            result = await compose_product_workflow(
                product_image_path=product_path,
                design_image_path=design_path,
                color_image_path=color_path,
                prompt=workflow_prompt,
                workflow_type=workflow_type
            )
            return JSONResponse(content={"status": "success", "result": result})
    except Exception as e:
        logger.error(f"Error in compose_product: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Intent Router API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 