import os
import ast
from eval_anything.utils.register import MMDatasetRegistry
from eval_anything.utils.data_type import InferenceInput
import eval_anything.utils.utils as utils
from eval_anything.utils.utils import MultiChoicePromptBuilder, DialoguePromptBuilder
from eval_anything.dataloader.base_dataloader import TASK_TYPE_MAP
from datasets import Dataset, load_dataset
from collections import namedtuple

class BaseMMDataset:
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        self.bench_cfgs = bench_cfgs
        self.task = task
        self.enable_cot = enable_cot
        self.num_shot = num_shot
        self.few_shot_examples = []
        self.few_shot_mm_examples = []

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        for item in few_shot_dataset[:self.num_shot]:
            self.few_shot_examples.append({
                "question": item[self.task.question_key],
                "candidate_answers": item[self.task.answer_key],
                "ground_truth": item[self.task.ground_truth_key]
            })        

    def build_multi_choice_prompt(self, item: dict):
        self.prompt_builder = MultiChoicePromptBuilder(
            candidate_labels=self.task.candidate_labels,
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], item[self.task.answer_key])
        return prompt

    def build_dialogue_prompt(self, item: dict):
        self.prompt_builder = DialoguePromptBuilder(
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], item[self.task.answer_key])
        return prompt
    
    def _to_InferenceInput(self, dataset: Dataset):
        pass

    def __call__(self, dataset: Dataset):
        return self._to_InferenceInput(dataset)

@MMDatasetRegistry.register("mmmu")
class MMMUDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def build_multi_choice_prompt(self, item: dict):
        self.prompt_builder = MultiChoicePromptBuilder(
            candidate_labels=self.task.candidate_labels,
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], ast.literal_eval(item[self.task.answer_key]))
        return prompt

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        for item in list(few_shot_dataset)[:self.num_shot]:
            images = []
            for i in range(1, 8):
                if item[f"image_{i}"]:
                    images.append(item[f"image_{i}"])
            self.few_shot_mm_examples.append(images)

            self.few_shot_examples.append({
                "question": item[self.task.question_key],
                "candidate_answers": ast.literal_eval(item[self.task.answer_key]),
                "ground_truth": item[self.task.ground_truth_key]
            })

    def _to_InferenceInput(self, dataset: Dataset):
        inference_inputs = []
        
        for item in dataset:
            images = []
            for i in range(1, 8):
                if item[f"image_{i}"]:
                    images.append(item[f"image_{i}"])
            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    text=self.build_multi_choice_prompt(item),
                    data_files=self.few_shot_mm_examples + images,
                    ref_answer=item[self.task.ground_truth_key],
                )
            )
        return inference_inputs
    