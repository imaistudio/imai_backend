import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
from datetime import datetime
import json
from pathlib import Path

from .services.image_processor import ImageProcessor
from .services.ai_analyzer import AIAnalyzer
from .services.image_generator import ImageGenerator
from .config import Settings

app = FastAPI(title="Design Creation API", 
             description="API for analyzing images and generating new designs")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings
settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

@app.post("/analyze-and-generate/")
async def analyze_and_generate(images: List[UploadFile] = File(...)):
    """
    Analyze multiple images and generate a new design based on their elements.
    
    Args:
        images: List of image files to analyze
        
    Returns:
        dict: Contains paths to the concatenated and generated images
    """
    try:
        # Save uploaded files
        saved_paths = []
        for image in images:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(settings.UPLOAD_DIR, f"{timestamp}_{image.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            saved_paths.append(file_path)

        # Initialize services
        image_processor = ImageProcessor()
        ai_analyzer = AIAnalyzer()
        image_generator = ImageGenerator()

        # Process images
        concatenated_path = os.path.join(settings.OUTPUT_DIR, "concatenated_horizontal.jpg")
        image_processor.concatenate_images(saved_paths, concatenated_path, direction='horizontal')

        # Analyze design elements
        design_elements = ai_analyzer.analyze_image(concatenated_path)

        # Generate new design
        generated_path = os.path.join(settings.OUTPUT_DIR, "generated_image.png")
        image_generator.generate_image(design_elements, generated_path)

        # Clean up uploaded files
        for path in saved_paths:
            os.remove(path)

        return {
            "status": "success",
            "design_elements": design_elements,
            "concatenated_image": concatenated_path,
            "generated_image": generated_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_type}/{filename}")
async def get_image(image_type: str, filename: str):
    """
    Retrieve a generated or concatenated image.
    
    Args:
        image_type: Either 'concatenated' or 'generated'
        filename: Name of the image file
        
    Returns:
        FileResponse: The requested image file
    """
    if image_type not in ["concatenated", "generated"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    file_path = os.path.join(settings.OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 