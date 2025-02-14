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
from eval_anything.utils.data_type import InferenceInput
from typing import List, Dict

class BaseDataLoader:
    task_type_map = {
        'Dialogue': 'build_dialogue_prompt',
        'MultiChoice': 'build_multi_choice_prompt',
    }

    def __init__(self, cfgs):
        self.eval_cfgs, self.data_cfgs = (
            cfgs.default.eval_cfgs,
            cfgs.default.data_cfgs,
        )
        self.action = self.eval_cfgs.action if self.eval_cfgs.action else 'generation'
        self.num_shot = self.eval_cfgs.n_shot if self.eval_cfgs.n_shot else 0
        self.cot = self.eval_cfgs.cot if self.eval_cfgs.cot else False
        self.split = self.data_cfgs.split
        self.task_dir = self.data_cfgs.task_dir
        self.task_info = self.get_task_info()

    def get_task_info(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [self.data_cfgs.task]
            return task_names

    def load_dataset(self) -> Dict[str, List[InferenceInput]]:
        prompts = {}
        for task in self.task_info:
            dataset = load_dataset(self.task_dir, task)
            self.few_shot_data = self.set_fewshot_dataset(dataset, task)
            prompt_builder = getattr(self, self.task_type_map[task['type']])
            prompts[task] = prompt_builder(dataset)
        return prompts

    def set_fewshot_dataset(self, task):
        if self.cot:
            if not self.data_cfgs.cot_fewshot_data_dir:
                raise ValueError(f"Chain of thought fewshot is not supported for task {self.data_cfgs.name}: {task}")
            cot_fewshot_data_split = self.data_cfgs.cot_fewshot_data_split if self.data_cfgs.cot_fewshot_data_split else 'train'
            return load_dataset(self.data_cfgs.cot_fewshot_data_dir, self.data_cfgs.cot_fewshot_data_file, split=cot_fewshot_data_split)
        else:
            fewshot_data_split = self.data_cfgs.fewshot_data_split if self.data_cfgs.fewshot_data_split else 'train'
            return load_dataset(self.data_cfgs.fewshot_data_dir, self.data_cfgs.fewshot_data_file, split=fewshot_data_split)

    @abstractmethod
    def get_task_dict(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def build_multi_choice_prompt(self, data):
        raise NotImplementedError

    @abstractmethod
    def build_dialogue_prompt(self, data):
        raise NotImplementedError