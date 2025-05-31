import requests
from ..config import get_settings
from typing import Dict, Any
import json

class ImageGenerator:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.AIML_API_KEY
        self.model = settings.AIML_MODEL
        self.api_url = "https://api.aimlapi.com/v1/images/generations"

    def _create_prompt(self, design_elements: Dict[str, Any]) -> str:
        """
        Create a prompt for image generation based on design elements.
        
        Args:
            design_elements (Dict[str, Any]): Dictionary of design elements from analysis
            
        Returns:
            str: Formatted prompt for image generation
        """
        # Convert design elements to a structured prompt
        if isinstance(design_elements, dict) and "raw_analysis" in design_elements:
            analysis = design_elements["raw_analysis"]
        else:
            analysis = json.dumps(design_elements, indent=2)

        prompt = (
            "I am a professional designer, specializing in pattern designs for products. "
            f"Based on the following descriptions and inspirations: {analysis}, create a completely new and unique design. the design should be seamless and blended and inspiring. "
            "Do not copy the existing designs; instead, use them as inspiration to innovate. "
            "Integrate the provided color palette created using all given elements into the new design."
        )
        
        return prompt

    def generate_image(self, design_elements: Dict[str, Any], output_path: str) -> None:
        """
        Generate a new image based on design elements.
        
        Args:
            design_elements (Dict[str, Any]): Dictionary of design elements from analysis
            output_path (str): Path where the generated image will be saved
        """
        try:
            # Create the prompt
            prompt = self._create_prompt(design_elements)

            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "model": self.model
            }

            # Make the API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            # Parse the response
            data = response.json()
            
            if 'images' not in data or not data['images']:
                raise ValueError("No image generated in the response")

            # Get the image URL
            image_url = data['images'][0]['url']
            
            # Download and save the image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(image_response.content)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error generating image: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in image generation: {str(e)}") 