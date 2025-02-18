"""
(t2t)支持transformers+accelerate推理
"""

import torch
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.register import TemplateRegistry as get_template
from eval_anything.models.base_model import BaseModel

class AccelerateModel(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], infer_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs
        
        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.chat_template = self.model_cfgs.chat_template
        self.template = get_template(self.chat_template)

        self.model_max_length = self.infer_cfgs.model_max_length

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

    def _generation(self, input_list: List[InferenceInput]) -> Dict[str, List[InferenceOutput]]:
        prompts = [
            self.template.system_prompt
            + self.template.user_prompt.format(input=input.text)
            + self.template.assistant_prompt.format(output='')
            for input in input_list
        ]
        encoded_inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.model_max_length, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoded_inputs["input_ids"].to(self.accelerator.device),
                attention_mask=encoded_inputs["attention_mask"].to(self.accelerator.device),
                max_length=self.model_max_length,
                num_return_sequences=1,
            )
            responses = [
                self.tokenizer.decode(output[encoded_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                for output in outputs
            ]

        inference_outputs = [
            InferenceOutput.from_data(question_id=input.uuid, prompt=input.text, response=response, store_raw=True)
            for input, response in zip(input_list, responses)
        ]

        return inference_outputs

    # TODO
    def shutdown_model(self):
        pass
