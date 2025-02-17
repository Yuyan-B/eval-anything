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

class AccelerateMultimodalModel(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], accelerate_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.accelerate_cfgs = accelerate_cfgs
        
        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.model_max_length = self.model_cfgs.model_max_length
        self.max_new_tokens = self.model_cfgs.max_new_tokens
        self.model_device = self.accelerate_cfgs.device

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

    def _generation(self, inputs: Dict[str, List[InferenceInput]]) -> Dict[str, List[InferenceOutput]]:
        input_list = []
        for task, data_list in inputs.items():
            for data in data_list:
                data.uuid = UUIDGenerator()(data)
                data.task = task
                input_list.append(data)

        prompts = [input.text for input in input_list]
        if input_list and input_list[0].urls:
            image_paths = [input.urls for input in input_list]
        elif input_list and input_list[0].data_files:
            image_paths = [input.data_files for input in input_list]
        else:
            raise ValueError("Each input item must have either 'urls' or 'data_files'.")



        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        # 编码文本
        text_inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.model_max_length, return_tensors="pt")

        # 加载图像
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        image_inputs = self.image_processor(images=images, return_tensors="pt")

        # 使用 CLIP 模型对图像和文本进行编码
        text_features = self.text_model(**text_inputs).logits
        image_features = self.image_model.get_image_features(**image_inputs)

        # 假设我们通过特定的方式融合图像和文本特征进行生成（你可以根据需求修改）
        multimodal_features = torch.cat((text_features, image_features), dim=-1)

        # 根据融合后的特征生成输出
        outputs = self.text_model.generate(
            input_ids=text_inputs["input_ids"].to(self.accelerator.device),
            attention_mask=text_inputs["attention_mask"].to(self.accelerator.device),
            max_length=self.model_max_length,
            num_return_sequences=1,  # 生成一个输出序列
        )

        # 解析输出并返回
        InferenceOutputs = [
            InferenceOutput.from_transformers_output(task=input.task, uuid=input.uuid, transformers_output=output, store_raw=True)
            for input, output in zip(input_list, outputs)
        ]

        outputs = {task: [] for task in inputs.keys()}
        for output in InferenceOutputs:
            outputs[output.task].append(output)

        return outputs

    # TODO: 添加模型的关闭或清理操作
    def shutdown_model(self):
        pass
