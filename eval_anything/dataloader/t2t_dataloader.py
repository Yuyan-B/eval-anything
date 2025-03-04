"""
t2t dataloader基类
输入：
    - 数据集路径
    - split
    - size
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型路径（如果需要pre-tokenize）
    - shuffle
    - num_workers
    - chat_template
    - ...
输出：
    - InferenceInput类
"""

from typing import Any, Dict, List
from collections import namedtuple

from eval_anything.dataloader.base_dataloader import BaseDataLoader


# from eval_anything.utils.registry import TemplateRegistry as get_template_class
from eval_anything.utils.utils import MultiChoicePromptBuilder, DialoguePromptBuilder, CodesGenerationPromptBuilder
from eval_anything.utils.data_type import InferenceInput


class T2TDataLoader(BaseDataLoader):
    
    def build_multi_choice_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        # template = get_template_class(self.chat_template)
        prompt_builder = MultiChoicePromptBuilder(
            candidate_labels=task.candidate_labels,
            few_shot_examples=few_shot_examples,
            cot=self.enable_cot
        )
        prompts = []
        question_key = task.question_key
        answer_key = task.answer_key
        ground_truth_key = task.ground_truth_key
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[answer_key])
            prompts.append(InferenceInput(task=task.name, text=prompt, ref_answer=item[ground_truth_key]))
        
        return prompts

    def build_dialogue_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        prompt_builder = DialoguePromptBuilder(
            few_shot_examples=few_shot_examples, 
            cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        prompts = [InferenceInput(task=task.name, text=prompt_builder.build_prompt(item[question_key]), ref_answer=item[ground_truth_key]) for item in data]
        
        return prompts
    
    def build_codes_generation_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        prompt_builder = CodesGenerationPromptBuilder(
            few_shot_examples=few_shot_examples,
            cot=self.enable_cot
        )
        prompts = []
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[ground_truth_key])
            prompts.append(InferenceInput(task=task.name, text=prompt, ref_answer=item[ground_truth_key]))
        
        return prompts
