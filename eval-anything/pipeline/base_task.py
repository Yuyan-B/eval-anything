"""
任务基类，不直接使用，而是继承后实现具体任务的逻辑
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
from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self, config: dict):
        # TODO 初始化数据集、模型、logger、任务列表
        self.config = config
    
    def get_configs(self):
        # TODO 获取配置
        pass

    def get_task_name(self):
        # TODO 模态无感，在此基类开发
        pass

    @abstractmethod
    def load_data(self):
        # TODO 初始化一个dataloader类
        pass

    @abstractmethod
    def load_model(self):
        # TODO 初始化一个model类
        pass
    
    def iterate_run(self):
        # TODO 迭代任务列表，调用run执行任务
        pass
    
    def run(self):
        # TODO 运行任务，调用inference进行推理，保存cache、调用calculate_metrics计算评测指标
        pass
    
    def get_cache_path(self):
        # TODO 获取cache路径并检查是否存在
        pass
    
    def save_cache(self):
        # TODO 将中间结果保存为cache
        pass
    
    def load_cache(self):
        # TODO 从cache中加载中间结果
        pass
    
    def get_cache(self):
        # TODO 获取cache
        pass
    
    def model_inference(self):
        # TODO 实现模型推理（调用models中的推理方式），需要支持多轮
        pass
    
    @abstractmethod
    def calculate_metrics(self):
        # TODO 给定metrics list，迭代执行calculate_single_metric
        pass
    
    def display_results(self):
        # TODO 在命令行中打印结果
        pass
    
    def save_results(self):
        # TODO 保存结果到指定路径
        pass