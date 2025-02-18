"""
(multi-modal)支持transformers+accelerate推理
"""

import json
import os
from typing import Any, Dict, List

from transformers import (
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)
from transformers import AutoConfig, AutoProcessor
from accelerate import Accelerator
import torch
from PIL import Image
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.utils import UUIDGenerator
from eval_anything.models.base_model import BaseModel


def generate_message(key, prompt):
    mappings = {
        "ti2t": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ],
    }
    return mappings.get(key, [])

class AccelerateMultimodalModel(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], infer_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs
        
        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.model_max_length = self.infer_cfgs.model_max_length
        self.max_new_tokens = self.infer_cfgs.max_new_tokens

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'
        
        self.init_model()

    def init_model(self) -> None:
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.ModelForConditionalGeneration = self.model_config.architectures[0]
        self.model_class = globals().get(self.ModelForConditionalGeneration)
        
        if self.model_class:
            self.model = getattr(self.model_class, 'from_pretrained')(self.model_name_or_path)
        else:
            raise ValueError(f"Model class {self.ModelForConditionalGeneration} not found.")

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)

    def generation(self, inputs: Dict[str, List[InferenceInput]]) -> Dict[str, List[InferenceOutput]]:
        return self._generation(inputs)

    def _generation(self, input_list: List[InferenceInput]) -> Dict[str, List[InferenceOutput]]:
        prompts = [input.text for input in input_list]
        if input_list and input_list[0].mm_data and input_list[0].mm_data[0].url:
            image_files = [input.mm_data[0].url for input in input_list]
            self.modality = input_list[0].mm_data[0].modality
        elif input_list and input_list[0].mm_data and input_list[0].mm_data[0].file:
            image_files = [input.mm_data[0].file for input in input_list]
            self.modality = input_list[0].mm_data[0].modality
        else:
            raise ValueError("Each input item must have either 'urls' or 'data_files'.")

        outputs = []
        for prompt, image_file in zip(prompts, image_files):
            if isinstance(image_file, Image.Image):
                image = image_file
            elif isinstance(image_file, str):
                image = Image.open(image_file).convert("RGB")
            else:
                raise ValueError("image_file is neither a PIL Image nor a string.")

            messages = generate_message(self.modality, prompt)
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=image, text=text, return_tensors="pt")
            inputs = inputs.to(self.accelerator.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs.append(output)

        inference_outputs = [
            InferenceOutput.from_vllm_output(task=input.task, uuid=input.uuid, vllm_output=output, store_raw=True)
            for input, output in zip(input_list, outputs)
        ]

        return inference_outputs

    # TODO
    def shutdown_model(self):
        pass
