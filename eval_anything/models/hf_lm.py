"""
(t2t)支持transformers+accelerate推理
"""

import json
import os
import torch
from typing import Any, Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.utils import UUIDGenerator
from eval_anything.models.base_model import BaseModel

class AccelerateModel(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], accelerate_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.accelerate_cfgs = accelerate_cfgs
        
        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.model_max_length = self.model_cfgs.model_max_length
        self.model_device = self.accelerate_cfgs.device

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'
        
        self.init_model()

    def init_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        
        encoded_inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.model_max_length, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoded_inputs["input_ids"].to(self.accelerator.device),
                attention_mask=encoded_inputs["attention_mask"].to(self.accelerator.device),
                max_length=self.model_max_length,
                num_return_sequences=1,
            )

        InferenceOutputs = [
            InferenceOutput.from_transformers_output(task=input.task, uuid=input.uuid, transformers_output=output, store_raw=True)
            for input, output in zip(input_list, outputs)
        ]
        
        outputs = {task: [] for task in inputs.keys()}
        for output in InferenceOutputs:
            outputs[output.task].append(output)

        return outputs

    # TODO
    def shutdown_model(self):
        pass
