"""
dataloader基类，不直接使用，而是继承后实现具体的数据加载逻辑
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

from abc import ABC, abstractmethod
from datasets import load_dataset
import os
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import get_project_root
from collections import defaultdict
from typing import List, Dict

TASK_TYPE_MAP = {
        'Dialogue': 'build_dialogue_prompt',
        'DialogueWithAnswer': 'build_dialogue_with_answer_prompt',
        'DialogueList': 'build_dialogue_list_prompt',
        'DialogueChinese': 'build_dialogue_chinese_prompt',
        'MultiChoice': 'build_multi_choice_prompt',
        'MultiChoiceAutoLabel': 'build_multi_choice_auto_label_prompt',
        'MultiChoiceChinese': 'build_multi_choice_prompt_chinese',
        'CodeGeneration': 'build_codes_generation_prompt'
    }

class BaseDataLoader:

    def __init__(self, eval_cfgs, bench_cfgs, logger):
        self.eval_cfgs, self.bench_cfgs = (
            eval_cfgs,
            bench_cfgs,
        )
        self.benchmark_name = self.bench_cfgs.dataset.name
        self.num_shot = getattr(self.eval_cfgs.n_shot, self.benchmark_name, 0)
        self.enable_cot = getattr(self.eval_cfgs.cot, self.benchmark_name, False)
        self.split = self.bench_cfgs.dataset.split
        self.data_dir = self.bench_cfgs.dataset.path
        self.task_info = self.get_task_info()
        self.few_shot_data = defaultdict(list)
        self.logger = logger
        
    def get_task_info(self):
        if isinstance(self.bench_cfgs.task, list):
            return self.bench_cfgs.task
        else:
            tasks = [self.bench_cfgs.task]
            return tasks

    def load_dataset(self, task_list: list[str]) -> Dict[str, List[InferenceInput]]:
        prompts = {}
        if task_list == []:
            task_list = list(self.bench_cfgs.dataset.default_task_list)
        for task in self.task_info:
            if task.name not in task_list:
                continue
            if task.data_files:
                dataset = load_dataset(self.data_dir, data_files=task.data_files, split="train")
            else:
                dataset = load_dataset(self.data_dir, task.name, split=self.bench_cfgs.dataset.split)
            self.few_shot_data[task.name] = self.set_fewshot_dataset(task.name) if self.num_shot != 0 else None

            # Build prompts
            prompt_builder = getattr(self, TASK_TYPE_MAP[task.type])
            prompts[task.name] = prompt_builder(task, dataset)

            # Save metadata if needed
            if task.has_metadata:
                for i, item in enumerate(prompts[task.name]):
                    item.metadata = dataset[i]

        return prompts

    def set_fewshot_dataset(self, task: str):
        if self.enable_cot:
            cot_fewshot_data_split = self.bench_cfgs.dataset.cot_fewshot_data_split if self.bench_cfgs.dataset.cot_fewshot_data_split else 'train'
            try:
                data_path = os.path.join(get_project_root(), "eval_anything", self.bench_cfgs.dataset.cot_fewshot_data_path)
                if os.path.exists(data_path):
                    few_shot_data = load_dataset(data_path, data_files=self.bench_cfgs.dataset.cot_fewshot_data_file, name=task, split=cot_fewshot_data_split)
                    return few_shot_data
                else:
                    few_shot_data = load_dataset(self.bench_cfgs.dataset.cot_fewshot_data_path, name=task, data_files=self.bench_cfgs.dataset.cot_fewshot_data_file, split=cot_fewshot_data_split)
                    return few_shot_data
            except:
                self.logger.log('error', f"Chain of thought fewshot is not supported for task {self.bench_cfgs.dataset.name}: {task}")
                raise
        else:
            fewshot_data_split = self.bench_cfgs.dataset.fewshot_data_split if self.bench_cfgs.dataset.fewshot_data_split else 'train'
            try:
                data_path = os.path.join(get_project_root(), "eval_anything", self.bench_cfgs.dataset.fewshot_data_path)
                if os.path.exists(data_path):
                    few_shot_data = load_dataset(data_path, data_files=self.bench_cfgs.dataset.fewshot_data_file, name=task, split=fewshot_data_split)
                    return few_shot_data
                else:
                    few_shot_data = load_dataset(self.bench_cfgs.dataset.fewshot_data_path, name=task, data_files=self.bench_cfgs.dataset.fewshot_data_file, split=fewshot_data_split)
                    return few_shot_data
            except:
                self.logger.log('error', f"Fewshot is not supported for task {self.bench_cfgs.dataset.name}: {task}")
                raise

    @abstractmethod
    def build_multi_choice_prompt(self, task: str, data: list[any]):
        raise NotImplementedError
    
    @abstractmethod
    def build_multi_choice_chinese_prompt(self, task: str, data: list[any]):
        raise NotImplementedError

    @abstractmethod
    def build_dialogue_prompt(self, task: str, data: list[any]):
        raise NotImplementedError
    
    @abstractmethod
    def build_dialogue_with_answer_prompt(self, task: str, data: list[any]):
        raise NotImplementedError

    @abstractmethod
    def build_dialogue_list_prompt(self, task: str, data: list[any]):
        raise NotImplementedError
    
    @abstractmethod
    def build_dialogue_chinese_prompt(self, task: str, data: list[any]):
        raise NotImplementedError