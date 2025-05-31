from openai import OpenAI
from ..config import get_settings
import base64
from typing import Dict, Any

class AIAnalyzer:
    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image using OpenAI's vision model to extract design elements.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict[str, Any]: Dictionary containing the analyzed design elements
        """
        try:
            # Encode the image
            base64_image = self._encode_image(image_path)
            
            # Create the API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze the design elements in this image. Focus on: colors, shapes, patterns, textures, and composition. Do not mention specific objects or types. Format the response as a JSON object with these categories."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            # Extract and parse the response
            analysis = response.choices[0].message.content
            
            # Convert the response to a dictionary
            try:
                design_elements = eval(analysis)  # Note: In production, use json.loads() with proper error handling
            except:
                # If parsing fails, return the raw analysis
                design_elements = {"raw_analysis": analysis}

            return design_elements

        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}") 