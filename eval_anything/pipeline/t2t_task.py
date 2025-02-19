"""
t2t任务基类，不直接使用，而是继承后实现具体任务的逻辑
输入：
    - 数据集路径
    - 模型路径
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型推理方式(调用eval-anything/models中的推理方式)
    - ...
输出：
    - EvaluationResult类
"""

from eval_anything.pipeline.base_task import BaseTask
from eval_anything.dataloader.t2t_dataloader import T2TDataLoader
from eval_anything.utils.data_type import EvaluationResult
import json
import os
from typing import Dict, List
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import pair_data_via_uuid

class T2TTask(BaseTask):
    def load_data(self, dataset_cfgs: dict):
        dataset = T2TDataLoader(dataset_cfgs)
        return dataset

    def save_benchmark_details(self, save_path: str, benchmark_name: str, inputs: Dict[str, List[InferenceInput]], results: Dict[str, List[EvaluationResult]]):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        output_dir = os.path.join(save_path, benchmark_name)
        os.makedirs(output_dir, exist_ok=True)
        for task, input_data in inputs.items():
            result_data = results[task]
            details = pair_data_via_uuid(input_data, result_data)

            with open(os.path.join(output_dir, f"{task}.jsonl"), 'a') as f:
                for detail in details:
                    f.write(json.dumps(detail.to_dict(), ensure_ascii=False) + '\n')