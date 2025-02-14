"""
任务基类，不直接使用，而是继承后实现具体任务的逻辑

TODO 
    - 代码鲁棒性
    - logger
"""
from abc import ABC, abstractmethod
from eval_anything.utils.logger import Logger
from eval_anything.utils.data_types import InferenceInput, InferenceOutput, EvaluationResult
from eval_anything.models.base_model import MODEL_MAP
import importlib


class BaseTask(ABC):
    def __init__(self, yaml_path: str):
        # TODO 初始化数据集、模型、logger、任务列表
        self.logger = Logger()
        self.task_cfgs, self.model_cfgs = self.load_configs(yaml_path)
        self.save_cache = True if self.task_cfgs["save_cache"] else False # TODO 确定具体的key值
        self.output_path = self.task_cfgs["output_path"]  # TODO 确定具体的key值
        self.benchmark_list = self.get_benchmark_name()
        self.model = self.load_model(self.model_cfgs)
        
    def get_overall_configs(self, yaml_path: str):
        # TODO 获取配置，模态无感，在此基类开发
        """Get configs from yaml file
        Args:
            yaml_path (str): yaml file path
            
        Returns:
            task_cfgs (dict): task configs, including
            model_cfgs (dict): model configs
        """
        pass
        
    def get_benchmark_name(self) -> list[str]:
        # TODO 获取任务列表，模态无感，在此基类开发
        """Get benchmark name from self.task_cfgs
            
        Returns:
            benchmark_name (list[str]): benchmark name list
        """
        pass
    
    def get_benchmark_cfgs(self, benchmark_name: str) -> dict:
        # TODO 获取任务配置
        """Get benchmark configs from yaml file in benchmark folder
        Args:
            benchmark_name (str): benchmark name
            
        Returns:
            benchmark_cfgs (dict): benchmark configs, including
        """
        pass

    @abstractmethod
    def load_data(self, dataset_cfgs: dict):
        # TODO 初始化一个dataloader类
        """Load evaluation dataset
        Args:
            dataset_cfgs (dict): dataset configs
            
        Returns:
            dataloader (BaseDataLoader): dataloader
        """
        pass

    def load_model(self, model_cfgs: dict, infer_cfgs: dict):
        module_name = f"eval_anything.models.{MODEL_MAP[f'{infer_cfgs["infer_backend"]}_{model_cfgs["model_type"]}']}"
        module = importlib.import_module(module_name)
        model = module(model_cfgs)
        return model
    
    def iterate_run(self):
        # TODO 迭代任务列表，调用run执行任务
        """Iterate benchmark list and run benchmarks"""
        self.results = []
        for benchmark_name in self.benchmark_list:
            result = self.run(benchmark_name)
            self.results.append(result)
        
        self.display_task_results(self.results)
        self.save_task_results(self.output_path, self.results)
        pass
    
    def run(self, benchmark_name: str):
        # TODO 运行任务，调用inference进行推理，保存cache、调用calculate_metrics计算评测指标
        """Run benchmark
        Args:
            benchmark_name (str): benchmark name
            
        Returns:
            result (EvaluationResult): evaluation result
        """
        benchmark_cfgs = self.get_benchmark_cfgs(benchmark_name)
            
        dataloader = self.load_data(benchmark_cfgs)
        input_data = dataloader.load_data()    # Input_data: list[InferenceInput]
        
        inference_outputs = self.model_inference(input_data)
        
        if self.save_cache:
            cache_path, cache_exist = self.get_cache_path(benchmark_cfgs, inference_outputs)
            if cache_exist:
                self.load_cache(cache_path)
            else:
                self.save_cache(cache_path, inference_outputs)
        
        result = self.calculate_metrics(benchmark_name, inference_outputs, benchmark_cfgs["metrics"])
        self.save_single_result(self.output_path, result)
        return result
    
    def get_cache_path(self, benchmark_cfgs: dict, inference_outputs: list[InferenceOutput]):
        # TODO 获取cache路径并检查是否存在
        """Get cache path and check if it exists
        Args:
            benchmark_cfgs (dict): benchmark configs
            inference_outputs (list[InferenceOutput]): inference outputs
            
        Returns:
            cache_path (str): cache path
            cache_exist (bool): whether the cache exists
        """
    
    def save_cache(self, cache_path: str, inference_outputs: list[InferenceOutput]):
        # TODO 将中间结果保存为cache
        """Save inference outputs as cache
        Args:
            cache_path (str): cache path
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        pass
    
    def load_cache(self, cache_path: str):
        # TODO 从cache中加载中间结果
        """Load inference outputs from cache
        Args:
            cache_path (str): cache path
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        pass
    
    def model_inference(self, model, input_data: list[InferenceInput]):
        # TODO 实现模型推理（调用models中的推理方式），需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        return model.generation(input_data)
    
    def shutdown_model(self):
        # TODO 关闭模型
        """Shutdown model"""
        self.model.shutdown()
    
    @abstractmethod
    def calculate_metrics(self, benchmark_name: str, inference_outputs: list[InferenceOutput], metrics_list: list[str]):
        # TODO 给定metrics list，迭代执行calculate_single_metric
        """Calculate metrics
        Args:
            benchmark_name (str): benchmark name
            inference_outputs (list[InferenceOutput]): inference outputs
            metrics_list (list[str]): metrics list
            
        Returns:
            result (EvaluationResult): evaluation result
        """
        
        pass
    
    @abstractmethod
    def display_task_results(self, results: list[EvaluationResult]):
        # TODO 在命令行中打印结果
        """Display overall evaluation results in command line.
        Args:
            results (list[EvaluationResult]): evaluation results
        """
        pass
    
    @abstractmethod
    def save_single_result(self, save_path: str, result: EvaluationResult):
        # TODO 保存结果到指定路径
        
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            result (list[EvaluationResult]): evaluation result
        """
        pass
    
    @abstractmethod
    def save_task_results(self, save_path: str, results: list[EvaluationResult]):
        # TODO 保存所有结果到指定路径
        """Save overall evaluation results, including all benchmark results.
        Args:
            save_path (str): save path
            results (list[EvaluationResult]): evaluation results
        """
        pass
    