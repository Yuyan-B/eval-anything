"""
任务基类，不直接使用，而是继承后实现具体任务的逻辑

TODO 
    - 代码鲁棒性
    - logger
"""
from abc import ABC, abstractmethod
import os
import hashlib
import importlib
import json

# Third-party imports
from vllm.sequence import Logprob
from vllm.sequence import RequestOutput

# Local imports
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.data_type import InferenceInput, InferenceOutput, EvaluationResult
from eval_anything.models.base_model import MODEL_MAP, CLASS_MAP
from eval_anything.evaluate_tools.t2t_tools import *
from eval_anything.evaluate_tools.metrics import MetricCalculator
from eval_anything.utils.utils import (
    UUIDGenerator, read_cfgs_from_yaml, update_dict, 
    custom_cfgs_to_dict, BENCHMARK_MODALITY_MAP
)
from eval_anything.utils.cache_manager import CacheManager


class BaseTask(ABC):
    def __init__(self, overall_cfgs_name: str, **kwargs):
        # TODO 初始化数据集、模型、logger、任务列表
        self.logger = EvalLogger('Evaluation')
        self.eval_cfgs, self.model_cfgs, self.infer_cfgs = self.get_overall_configs(overall_cfgs_name, **kwargs)
        self.enable_cache = True if self.eval_cfgs["cache_dir"] else False
        self.output_path = self.eval_cfgs["output_dir"]
        self.model = self.load_model(self.model_cfgs, self.infer_cfgs)
        self.cache_manager = CacheManager(self.eval_cfgs["cache_dir"]) if self.enable_cache else None
    
    def load_configs(self, yaml_path: str):
        # TODO 获取配置，模态无感，在此基类开发
        """Load configs from yaml file
        Args:
            task_cfgs_path (str): task configs path
        """
        
        pass

    def get_overall_configs(self, overall_cfgs_name: str, **kwargs):
        # TODO 获取配置，模态无感，在此基类开发
        """Get configs from yaml file
        Args:
            overall_cfgs_name (str): YAML filename for evaluation configurations in eval-anything/configs/.
            
        Returns:
            eval_cfgs (dict): eval configs, including
            model_cfgs (dict): model configs
            infer_cfgs (dict): infer configs
        """
        overall_cfgs = read_cfgs_from_yaml(yaml_relative_dir='configs', yaml_name=overall_cfgs_name)
        eval_cfgs = overall_cfgs['eval_cfgs']
        model_cfgs = overall_cfgs['model_cfgs']
        infer_cfgs = overall_cfgs['infer_cfgs']
        for k, v in kwargs.items():
            if v == '' or v is None:
                continue
            eval_cfgs = update_dict(eval_cfgs, custom_cfgs_to_dict(k, v))
            model_cfgs = update_dict(model_cfgs, custom_cfgs_to_dict(k, v))
            infer_cfgs = update_dict(infer_cfgs, custom_cfgs_to_dict(k, v))
        return eval_cfgs, model_cfgs, infer_cfgs
        
    def get_benchmark_dict(self) -> dict[str, list[str]]:
        # TODO 获取任务列表，模态无感，在此基类开发
        """Get benchmark name from self.task_cfgs
            
        Returns:
            benchmark_dict (dict[str, list[str]]): {benchmark_name: [task_name1, task_name2, ...]}
        """
        return self.eval_cfgs['benchmarks']
    
    def get_benchmark_cfgs(self, benchmark_name: str) -> dict:
        # TODO 获取任务配置
        """Get benchmark configs from yaml file in benchmark folder
        Args:
            benchmark_name (str): benchmark name
            
        Returns:
            benchmark_cfgs (dict): benchmark configs, including
        """
        benchmark_cfgs = read_cfgs_from_yaml(yaml_relative_dir=f'benchmarks/{BENCHMARK_MODALITY_MAP[benchmark_name.lower()]}/{benchmark_name}', yaml_name=f'{benchmark_name}.yaml')
        return benchmark_cfgs

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
        backend_type = f"{infer_cfgs['infer_backend']}_{model_cfgs['model_type']}"
        module_name = f"eval_anything.models.{MODEL_MAP[backend_type]}"
        module = importlib.import_module(module_name)
        model_class = getattr(module, CLASS_MAP[backend_type])
        model = model_class(model_cfgs, infer_cfgs)
        return model
    
    def batch_inference(self, model, input_dict: dict[str, list[InferenceInput]]):
        # TODO 实现模型推理（调用models中的推理方式），需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        
        input_list = []
        for task, data_list in input_dict.items():
            for data in data_list:
                data.uuid = UUIDGenerator()(data)
                data.task = task
                input_list.append(data)
        
        batch_size = 100
        input_data_batches = [input_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]
        inference_outputs = []
        for input_data_batch in input_data_batches:
            if self.enable_cache:
                cache_path, cache_exist = self.cache_manager.get_cache_path(
                    self.model_cfgs, 
                    input_data_batch
                )
                if cache_exist:
                    inference_outputs.extend(self.cache_manager.load(cache_path))
                else:
                    batch_outputs = self.model_inference(model, input_data_batch)
                    inference_outputs.extend(batch_outputs)
                    self.cache_manager.save(cache_path, batch_outputs)
            else:
                inference_outputs.extend(self.model_inference(model, input_data_batch))

        outputs = {task: [] for task in input_dict.keys()}
        for output in inference_outputs:
            outputs[output.task].append(output)
        return outputs

    def model_inference(self, model, input_data: list[InferenceInput]):
        # TODO 实现模型推理（调用models中的推理方式），需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        return model.generation(input_data)
    
    def iterate_run(self):
        # TODO 迭代任务列表，调用run执行任务
        """Iterate benchmark list and run benchmarks"""
        self.results = []
        self.benchmark_dict = self.get_benchmark_dict()
        for benchmark_name, task_list in self.benchmark_dict.items():
            result = self.run(benchmark_name, task_list)
            self.results.append(result)
        
        self.display_task_results(self.results)
        self.save_task_results(self.output_path, self.results)
        return self.results
    
    def run(self, benchmark_name: str, task_list: list[str]):
        # TODO 运行任务，调用inference进行推理，保存cache、调用calculate_metrics计算评测指标
        """Run benchmark
        Args:
            benchmark_name (str): benchmark name
            
        Returns:
            result (EvaluationResult): evaluation result
        """
        self.benchmark_cfgs = self.get_benchmark_cfgs(benchmark_name)
        
        dataloader = self.load_data(self.eval_cfgs, self.benchmark_cfgs)
        input_data = dataloader.load_data(task_list)    # Input_data: list[InferenceInput]
        
        inference_outputs = self.batch_inference(self.model, input_data)

        result = self.calculate_metrics(benchmark_name, inference_outputs, self.benchmark_cfgs["metrics"])
        self.save_single_result(self.output_path, result)
        return result        
    
    def shutdown_model(self):
        # TODO 关闭模型
        """Shutdown model"""
        self.model.shutdown()
    
    def calculate_metrics(self, benchmark_name: str, inference_outputs: list[InferenceOutput], evaluate_tools: list[str], judge_methods: list[str], metrics_list: list[dict]):
        # TODO 执行metric_calculator
        """Calculate metrics
        Args:
            benchmark_name (str): benchmark name
            inference_outputs (list[InferenceOutput]): inference outputs
            evaluate_tools (list[str]): evaluate tool list
            judge_methods (list[str]): judge method list
            metrics_list (list[dict]): metrics list
            
        Returns:
            result (EvaluationResult): evaluation result
        """
        extracted_results = {evaluate_tool: getattr(str, evaluate_tool)(inference_outputs) for evaluate_tool in evaluate_tools}
        ground_truths = [self.get_ground_truth(inference_output) for inference_output in inference_outputs]
        evaluation_results = [EvaluationResult(benchmark_name, inference_output, extracted_result, ground_truth, judge_methods) for inference_output, extracted_result, ground_truth in zip(inference_outputs, extracted_results, ground_truths)]
        metric_calculator = MetricCalculator(metrics_list)
        statistics_results = metric_calculator(evaluation_results)
        return statistics_results

    def get_ground_truth(self, inference_output: InferenceOutput):
        # TODO 获取Ground Truth，待开发
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
    