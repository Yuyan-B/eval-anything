
import json
import os
import re
from PIL import Image
from typing import Any, Dict, List

from vllm import LLM, SamplingParams
from vllm.utils import cuda_device_count_stateless

from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.utils import UUIDGenerator
from eval_anything.utils.register import TemplateRegistry as get_template
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.utils import get_messages
from transformers import AutoProcessor

class vllmMM(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], infer_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs
        self.sp_n = self.infer_cfgs.num_output
        self.sp_top_k = self.infer_cfgs.top_k
        self.sp_top_p = self.infer_cfgs.top_p
        self.sp_temperature = self.infer_cfgs.temperature
        self.sp_max_tokens = self.infer_cfgs.model_max_length
        self.sp_prompt_logprobs = self.infer_cfgs.prompt_logprobs
        self.sp_logprobs = self.infer_cfgs.logprobs

        self.llm_trust_remote_code = self.infer_cfgs_llm.trust_remote_code
        self.llm_gpu_memory_utilization = self.infer_cfgs.gpu_memory_utilization
        tensor_ps = self.infer_cfgs.num_gpu
        self.llm_tensor_parallel_size = tensor_ps if tensor_ps else cuda_device_count_stateless()

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.chat_template = self.model_cfgs.chat_template
        self.template = get_template(self.chat_template)
        
        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'
        self.init_model()

    def init_model(self) -> None:
        """
        Initialize the model with sampling parameters and load the LLM.
        """
        # Create sampling parameters for generation (e.g., top-k, top-p, temperature, etc.)
        self.samplingparams = SamplingParams(
            n=self.sp_n,
            top_k=self.sp_top_k,
            top_p=self.sp_top_p,
            temperature=self.sp_temperature,
            max_tokens=self.sp_max_tokens,
            prompt_logprobs=self.sp_prompt_logprobs,
            logprobs=self.sp_logprobs,
        )

        self.model = LLM(
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            trust_remote_code=self.llm_trust_remote_code,
            tensor_parallel_size=self.llm_tensor_parallel_size,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)


    def generation(self, inputs: Dict[str, List[InferenceInput]]) -> Dict[str, List[InferenceOutput]]:
        """
        Generate outputs for a batch of inputs.
        """
        return self._generation(inputs)

    def _generation(self, input_list: List[InferenceInput]) -> Dict[str, List[InferenceOutput]]:
        """
        Internal method to handle generation logic using the model.
        Processes input list and returns inference outputs.
        """
        try:
            prompts = [
                self.processor.apply_chat_template(get_messages(self.modality, input.text), add_generation_prompt=True)
                for input in input_list
            ]
        except Exception as e:
            if self.chat_template:
                prompts = [
                    self.template.system_prompt
                    + self.template.user_prompt.format(input=input.text)
                    + self.template.assistant_prompt.format(output='')
                    for input in input_list
                ]
            else:
                prompts = [input.text for input in input_list]
            

        if input_list and input_list[0].mm_data and input_list[0].mm_data[0].url:
            image_files = [input.mm_data[0].url for input in input_list]
            self.modality = input_list[0].mm_data[0].modality
        elif input_list and input_list[0].mm_data and input_list[0].mm_data[0].file:
            image_files = [input.mm_data[0].file for input in input_list]
            self.modality = input_list[0].mm_data[0].modality
        else:
            raise ValueError("Each input item must have either 'url' or 'file'.")
        
        vllm_inputs = []
        for prompt, image_file in zip(prompts, image_files):
            if isinstance(image_file, Image.Image):
                image = image_file
            elif isinstance(image_file, str):
                image = Image.open(image_file).convert("RGB")
            else:
                raise ValueError("image_file is neither a PIL Image nor a string.")
            
            vllm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {self.modality: image},
            })
            
        outputs = self.model.generate(
            prompts=vllm_inputs, sampling_params=self.samplingparams
        )
        inference_outputs = [
            InferenceOutput.from_vllm_output(task=input.task, uuid=input.uuid, vllm_output=output, store_raw=True)
            for input, output in zip(input_list, outputs)
        ]

        return inference_outputs
    
    # TODO
    def shutdown_model(self):
        pass
