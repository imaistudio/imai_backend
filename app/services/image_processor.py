from PIL import Image
import os
from typing import List

class ImageProcessor:
    def concatenate_images(self, image_paths: List[str], output_path: str, direction: str = 'horizontal') -> None:
        """
        Concatenates multiple images into a single image.

        Args:
            image_paths (List[str]): List of paths to input images
            output_path (str): Path where the concatenated image will be saved
            direction (str): 'horizontal' or 'vertical' concatenation
        """
        if not image_paths:
            raise ValueError("No image paths provided")

        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found at {path}")
            try:
                images.append(Image.open(path))
            except IOError as e:
                raise IOError(f"Could not open image at {path}: {str(e)}")

        if not images:
            raise ValueError("No valid images found to concatenate")

        # Determine dimensions for new image
        if direction == 'horizontal':
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            new_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                new_image.paste(img, (x_offset, 0))
                x_offset += img.width
        elif direction == 'vertical':
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            new_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in images:
                new_image.paste(img, (0, y_offset))
                y_offset += img.height
        else:
            raise ValueError("Invalid direction. Choose 'horizontal' or 'vertical'")

        # Save the concatenated image
        try:
            new_image.save(output_path)
        except Exception as e:
            raise IOError(f"Error saving concatenated image: {str(e)}")

        # Close all images
        for img in images:
            img.close() 