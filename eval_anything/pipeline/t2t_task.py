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

class T2TTask(BaseTask):
    def load_data(self, dataset_cfgs: dict):
        dataset = T2TDataLoader(dataset_cfgs)
        return dataset
    
    def display_task_results(self, results: list[EvaluationResult]):
        pass
    
    def save_single_result(self, save_path: str, result: EvaluationResult):
        pass
    
    def save_task_results(self, save_path: str, results: list[EvaluationResult]):
        pass