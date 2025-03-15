"""
Utility functions for multi-modal data.
- Handle the conversion between multi-modal data and the required format.
- TODO: Display multi-modal data.
"""

import PIL
from PIL import Image
from io import BytesIO
from typing import List, Dict, Union
import base64
import re

class ImageManager:
    """
    Convert between any multi-modal data (image, audio, video, etc.) and base64 string.
    """

    @classmethod
    def decode_base64_to_image(cls, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image object and ensure consistent format"""
        try:
            # Remove data URI prefix if present
            if base64_string.startswith("data:image/"):
                base64_data = base64_string.split(",", 1)[1]
            else:
                base64_data = base64_string
        
            # Decode base64 to bytes and convert to Image
            image = Image.open(BytesIO(base64.b64decode(base64_data)))
            
            # Convert to RGB/RGBA based on whether transparency is needed
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                return image.convert('RGBA')
            return image.convert('RGB')
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    @classmethod
    def encode_image_to_base64(cls, image: Union[str, "PIL.Image"]) -> str:
        """Convert image to base64 string with consistent format"""
        try:
            if isinstance(image, str):
                image_input = Image.open(image)
            else:
                image_input = image
            
            # Determine if image needs transparency
            if image_input.mode == 'RGBA' and cls._has_transparency(image_input):
                buffer = BytesIO()
                image_input.save(buffer, format="PNG")  # Use PNG for images with transparency
            else:
                image_input = image_input.convert("RGB")
                buffer = BytesIO()
                image_input.save(buffer, format="JPEG", quality=95)
            
            img_bytes = buffer.getvalue()
            base64_data = base64.b64encode(img_bytes).decode("utf-8")
            return base64_data
            
        except Exception as e:
            raise ValueError(f"Failed to encode image to base64: {str(e)}")

    @classmethod
    def _has_transparency(cls, image: Image.Image) -> bool:
        """Check if image has any transparent pixels"""
        if image.mode == 'RGBA':
            extrema = image.getextrema()
            if len(extrema) >= 4:  # Make sure we have alpha channel
                alpha_min, alpha_max = extrema[3]
                return alpha_min < 255
        return False

    @classmethod
    def extract_images_from_conversation(cls, conversation: List[Dict]) -> List[Image.Image]:
        """Extract all images from the conversation with consistent format"""
        images = []
    
        for message in conversation:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    try:
                        if 'image' in item:
                            image = cls.decode_base64_to_image(item['image'])
                            images.append(image)
                        elif 'image_url' in item:
                            image = cls.decode_base64_to_image(item['image_url'])
                            images.append(image)
                    except Exception as e:
                        print(f"Warning: Failed to process image in conversation: {str(e)}")
                        continue
        return images

    @classmethod
    def prompt_to_conversation(
            cls,  
            user_prompt: str, 
            system_prompt: Union[str, None] = None, 
            images: Union[List[PIL.Image], List[str]] = []
        ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format
        
        Args:
            user_prompt (str): Input user_prompt with image placeholders in <image n> format
            system_prompt (str): Input system_prompt (if exists)
            images (list): List of PIL.Image objects to be encoded and inserted into the conversation
            
        Returns:
            list: Conversation object in the specified format
        """
        image_pattern = re.compile(r'<image (\d+)>')
        matches = list(image_pattern.finditer(user_prompt))
        assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}"
        
        content_parts = []
        
        if not matches:
            if user_prompt:
                content_parts.append({
                    "type": "text",
                    "text": user_prompt
                })
        else:
            if matches[0].start() > 0:
                content_parts.append({
                    "type": "text",
                    "text": user_prompt[:matches[0].start()]
                })
            
            for i, match in enumerate(matches):
                content_parts.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{cls.encode_image_to_base64(images[i])}"  # Use cls
                })
                
                text_start = match.end()
                text_end = matches[i+1].start() if i+1 < len(matches) else len(user_prompt)
                
                if text_end > text_start:
                    content_parts.append({
                        "type": "text",
                        "text": user_prompt[text_start:text_end]
                    })

        conversation = [
            {
                "role": "user",
                "content": content_parts
            }
        ]

        if system_prompt is not None:
            conversation.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            })
        
        return conversation
    
