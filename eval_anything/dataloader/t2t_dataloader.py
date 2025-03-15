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
from eval_anything.utils.utils import MultiChoicePromptBuilder, MultiChoiceAutoLabelPromptBuilder, MultiChoicePromptChineseBuilder, DialoguePromptBuilder, DialoguePromptChineseBuilder, CodesGenerationPromptBuilder
from eval_anything.utils.data_type import InferenceInput


class T2TDataLoader(BaseDataLoader):

    def build_conversation_from_prompt(self, prompt: str):
        """
        Build a default conversation from a text prompt.
        """
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        return conversation
    
    def build_multi_choice_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        
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
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]))
        
        return prompts
    
    # for mc1_targets of truthfulqa multiple_choice subset currently
    def build_multi_choice_auto_label_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []

        prompt_builder = MultiChoiceAutoLabelPromptBuilder(
            few_shot_examples=few_shot_examples,
            cot=self.enable_cot
        )
        prompts = []
        question_key = task.question_key
        answer_key = task.answer_key
        ground_truth_key = task.ground_truth_key
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[answer_key]['choices'])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]['labels']))
        
        return prompts

    def build_multi_choice_prompt_chinese(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = MultiChoicePromptChineseBuilder(
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
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]))
        
        return prompts

    def build_dialogue_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = DialoguePromptBuilder(
            few_shot_examples=few_shot_examples, 
            cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        prompts = []
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]))
        
        return prompts
    
    def build_dialogue_with_answer_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = DialoguePromptBuilder(
            few_shot_examples=few_shot_examples, 
            cot=self.enable_cot
        )
        question_key = task.question_key
        best_ground_truth_key = task.best_ground_truth_key
        ground_truth_key = task.ground_truth_key
        anti_ground_truth_key = task.anti_ground_truth_key
        
        prompts = []
        idx = 0
        for item in data:
            best_answer_index = item[ground_truth_key].index(item[best_ground_truth_key])
            for correct_answer in item[ground_truth_key]:
                prompt = prompt_builder.build_prompt(item[question_key], correct_answer)
                conversation = self.build_conversation_from_prompt(prompt)
                prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=best_answer_index, text_id=idx))
            idx += 1
            for incorrect_answer in item[anti_ground_truth_key]:    
                prompt = prompt_builder.build_prompt(item[question_key], incorrect_answer)
                conversation = self.build_conversation_from_prompt(prompt)
                prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=best_answer_index, text_id=idx))        
        return prompts
    
    # for bleurt metric of truthfulqa currently
    def build_dialogue_list_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = DialoguePromptBuilder(
            few_shot_examples=few_shot_examples, 
            cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        anti_ground_truth_key = task.anti_ground_truth_key
        
        for item in data:
            if "I have no comment." not in item[ground_truth_key]:
                item[ground_truth_key].append("I have no comment.")

        prompts = []
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer={'correct_answers': item[ground_truth_key], 'incorrect_answers': item[anti_ground_truth_key]}))
        
        return prompts
    
    def build_dialogue_chinese_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = DialoguePromptChineseBuilder(
            few_shot_examples=few_shot_examples, 
            cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        prompts = []
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]))
        
        return prompts

    def build_codes_generation_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = CodesGenerationPromptBuilder(
            few_shot_examples=few_shot_examples,
            cot=self.enable_cot,
            language=task.language
        )
        prompts = []
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[ground_truth_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(InferenceInput(task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]))
        
        return prompts
