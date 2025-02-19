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

    def __init__(self, eval_cfgs, data_cfgs):
        self.eval_cfgs, self.data_cfgs = (
            eval_cfgs,
            data_cfgs,
        )
        self.benchmark_name = self.data_cfgs.dataset['name']
        self.num_shot = self.eval_cfgs.n_shot[self.benchmark_name] if self.eval_cfgs.n_shot[self.benchmark_name] else 0
        self.cot = self.eval_cfgs.cot[self.benchmark_name] if self.eval_cfgs.cot[self.benchmark_name] else False
        self.split = self.data_cfgs.split
        self.data_dir = self.data_cfgs.path
        self.task_info = self.get_task_info()

    def get_task_info(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            tasks = [self.data_cfgs.task]
            return tasks

    def load_dataset(self, task_list: list[str]) -> Dict[str, List[InferenceInput]]:
        prompts = {}
        if task_list == []:
            task_list = self.data_cfgs.dataset['default_task_list']
        for task in self.task_info:
            if task['name'] not in task_list:
                continue
            if task['data_files']:
                dataset = load_dataset(self.data_dir, data_files=task['data_files'])
            else:
                dataset = load_dataset(self.data_dir, task['name'])
            self.few_shot_data = self.set_fewshot_dataset(dataset, task)
            prompt_builder = getattr(self, self.task_type_map[task['type']])
            prompts[task['name']] = prompt_builder(task, dataset)
        return prompts

    def set_fewshot_dataset(self, task):
        if self.cot:
            if not self.data_cfgs.cot_fewshot_data_dir:
                raise ValueError(f"Chain of thought fewshot is not supported for task {self.data_cfgs.name}: {task}")
            cot_fewshot_data_split = self.data_cfgs.cot_fewshot_data_split if self.data_cfgs.cot_fewshot_data_split else 'train'
            return load_dataset(self.data_cfgs.cot_fewshot_data_path, data_files=self.data_cfgs.cot_fewshot_data_file, split=cot_fewshot_data_split)
        else:
            if not self.data_cfgs.fewshot_data_path:
                raise ValueError(f"Fewshot is not supported for task {self.data_cfgs.name}: {task}")
            fewshot_data_split = self.data_cfgs.fewshot_data_split if self.data_cfgs.fewshot_data_split else 'train'
            return load_dataset(self.data_cfgs.fewshot_data_path, data_files=self.data_cfgs.fewshot_data_file, split=fewshot_data_split)

    @abstractmethod
    def get_task_dict(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def build_multi_choice_prompt(self, task: str, data: list[any]):
        raise NotImplementedError

    @abstractmethod
    def build_dialogue_prompt(self, task: str, data: list[any]):
        raise NotImplementedError