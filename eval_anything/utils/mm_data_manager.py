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
    def __init__(self):
        pass

    def decode_base64_to_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image object"""
        # Remove data URI prefix if present
        if base64_string.startswith("data:image/"):
            base64_data = base64_string.split(",", 1)[1]
        else:
            base64_data = base64_string
    
        # Decode base64 to bytes and convert to Image
        return Image.open(BytesIO(base64.b64decode(base64_data)))


    def encode_image_to_base64(image: Union[str, "PIL.Image"]) -> str:
        """Get base64 from image"""
        if isinstance(image, str):
            image_input = Image.open(image)
        else:
            image_input = image
        
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_data = base64.b64encode(img_bytes).decode("utf-8")
        return base64_data
    
    def extract_images_from_conversation(self, conversation: List[Dict]) -> List[Image.Image]:
        """Extract all images from the conversation"""
        images = []
    
        for message in conversation:
            content = message.get('content', '')
            if isinstance(content, str):
                pass

            elif isinstance(content, list):
                for item in content:
                    if 'image' in item:
                        images.append(self.decode_base64_to_image(item['image']))
                    elif 'image_url' in item:
                        images.append(self.decode_base64_to_image(item['image_url']))
        return images

    def prompt_to_conversation(
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
                    "image": f"data:image/jpeg;base64,{ImageManager.encode_image_to_base64(images[i])}"
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
    
