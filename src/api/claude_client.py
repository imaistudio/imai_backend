from typing import Dict, Any, Optional
import anthropic
from src.utils.logger import logger
from src.utils.config import settings
import json

class ClaudeClient:
    """Client for interacting with Claude 3.7 Sonnet API."""
    
    def __init__(self):
        """Initialize the Claude client with API key from settings."""
        self.client = anthropic.Anthropic()  # Will use ANTHROPIC_API_KEY from environment    claude-3-7-sonnet-20250219
        self.model = "claude-sonnet-4-20250514"  # Use the general Claude 3.7 Sonnet model name
        
    def analyze_intent(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user input to determine intent and extract parameters.
        
        Args:
            user_input: The user's natural language input
            context: Optional context information (e.g., previous interactions)
            
        Returns:
            Dict containing intent analysis results
        """
        try:
            # Construct the prompt for Claude
            system_prompt = """You are an AI intent recognition system. Your task is to analyze user input and determine:
1. The user's intent (create_design, edit_image, generate_image, analyze_image, compare_images, or unknown)
2. Confidence score (0.0 to 1.0)
3. Relevant parameters extracted from the input
4. Suggested next actions
5. Required inputs

Available intents:
- create_design: User wants to create a new design based on exactly three inspiration images
- edit_image: User wants to edit a single existing image (requires exactly one input image)
- generate_image: User wants to generate a new image, which can be either:
  * Text-only generation (no input images)
  * Image-based generation (1-3 input images: product image + optional color/design inspiration) and a prompt
- analyze_image: User wants to analyze a single image in detail (requires exactly one input image)
- compare_images: User wants to compare 2-3 images (requires 2-3 input images)
- unknown: Intent cannot be determined

Important rules for intent classification:
1. If the user provides multiple images and wants to create something new, use generate_image
2. If the user provides exactly one image and wants to modify it, use edit_image
3. If the user provides exactly three images and wants to create a new design combining them, use create_design
4. If the user provides no images and wants to generate something from text, use generate_image
5. If the user provides exactly one image and wants to analyze it, use analyze_image
6. If the user provides 2-3 images and wants to compare them, use compare_images

Respond in JSON format with the following structure:
{
    "intent": "string",
    "confidence": float,
    "parameters": {
        "key": "value"
    },
    "suggested_actions": ["action1", "action2"],
    "required_inputs": ["input1", "input2"]
}"""

            # Add context if provided
            if context:
                context_str = f"\nContext: {context}"
            else:
                context_str = ""
                
            # Make the API call using the new format
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.1,  # Low temperature for more deterministic results
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"User input: {user_input}{context_str}"
                    }
                ]
            )
            
            # Get the response text
            content = response.content[0].text
            logger.info(f"Claude response: {content}")
            
            # Parse the JSON response
            try:
                # The response might be wrapped in ```json ... ```, so clean it up
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()
                    
                intent_analysis = json.loads(json_str)
                
                # Validate required fields
                if not isinstance(intent_analysis, dict):
                    raise ValueError("Response is not a JSON object")
                    
                required_fields = ["intent", "confidence", "parameters", "suggested_actions", "required_inputs"]
                for field in required_fields:
                    if field not in intent_analysis:
                        raise ValueError(f"Missing required field: {field}")
                        
                return intent_analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing Claude's JSON response: {str(e)}")
                logger.error(f"Raw response was: {content}")
                raise ValueError(f"Failed to parse Claude's response: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in Claude intent analysis: {str(e)}", exc_info=True)
            # Return a safe fallback response
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "suggested_actions": ["clarify_intent"],
                "required_inputs": []
            }
            
    def generate_prompt(self, intent: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a detailed prompt for the workflow based on intent and parameters.
        
        Args:
            intent: The recognized intent
            parameters: Extracted parameters from user input
            
        Returns:
            A detailed prompt for the workflow
        """
        try:
            system_prompt = f"""You are an AI prompt generation system. Generate a detailed prompt for {intent} based on these parameters:
{parameters}

The prompt should be specific, detailed, and suitable for image generation/editing."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": "Generate a detailed prompt."
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}", exc_info=True)
            return str(parameters.get("user_directive", "")) 