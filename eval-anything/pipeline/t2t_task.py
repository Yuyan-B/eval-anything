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

class T2TTask(BaseTask):
    def __init__(self, yaml_path: str):
        super().__init__(yaml_path)
        
    def load_data(self, dataset_cfgs: dict):
        dataset = T2TDataLoader(dataset_cfgs)
        return dataset.load_dataset()
    
