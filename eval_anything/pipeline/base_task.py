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
import yaml
from typing import Dict, List

# Third-party imports
# from vllm.sequence import Logprob
# from vllm.sequence import RequestOutput

# Local imports
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.data_type import InferenceInput, InferenceOutput, EvaluationResult
from eval_anything.models.base_model import MODEL_MAP, CLASS_MAP
from eval_anything.evaluate_tools.t2t_tools import *
from eval_anything.evaluate_tools.metrics import MetricCalculator, OverallMetricCalculator
from eval_anything.utils.utils import (
    UUIDGenerator, read_cfgs_from_yaml, update_dict, 
    custom_cfgs_to_dict, BENCHMARK_MODALITY_MAP, pair_data_via_uuid
)
from eval_anything.utils.cache_manager import CacheManager


class BaseTask(ABC):
    def __init__(self, overall_cfgs_name: str, **kwargs):
        self.eval_cfgs, self.model_cfgs, self.infer_cfgs = self.get_overall_configs(overall_cfgs_name, **kwargs)
        self.enable_cache = True if self.eval_cfgs["cache_dir"] else False
        self.output_path = self.eval_cfgs["output_dir"]
        self.logger = EvalLogger('Evaluation', log_dir=self.output_path)
        self.model = self.load_model(self.model_cfgs, self.infer_cfgs)
        self.cache_manager = CacheManager(self.eval_cfgs["cache_dir"]) if self.enable_cache else None

    def get_overall_configs(self, overall_cfgs_name: str, **kwargs):
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
        """Get benchmark name from self.task_cfgs
            
        Returns:
            benchmark_dict (dict[str, list[str]]): {benchmark_name: [task_name1, task_name2, ...]}
        """
        return self.eval_cfgs['benchmarks']
    
    def get_benchmark_cfgs(self, benchmark_name: str) -> dict:
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
    
    def batch_inference(self, model, input_dict: dict[str, list[InferenceInput]]) -> dict[str, list[InferenceOutput]]:
        # TODO 需要支持多轮
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
        # TODO 需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        return model.generation(input_data)
    
    def iterate_run(self) -> list[dict[str, dict[str, float]]]:
        """Iterate benchmark list and run benchmarks"""
        self.results = {}
        self.benchmark_dict = self.get_benchmark_dict()
        for benchmark_name, task_list in self.benchmark_dict.items():
            evaluation_details, evaluation_results = self.run(benchmark_name, task_list)
            self.results[benchmark_name] = evaluation_results
        
        self.display_task_results(self.benchmark_dict, self.results)
        self.save_task_details()
        return self.results
    
    def run(self, benchmark_name: str, task_list: list[str]) -> tuple[dict[str, list[EvaluationResult]], dict[str, dict[str, float]]]:
        """Run benchmark
        Args:
            benchmark_name (str): benchmark name
            
        Returns:
            evaluation_details (dict[str, list[EvaluationResult]]): evaluation details
            evaluation_results (dict[str, dict[str, float]]): evaluation results
        """
        self.benchmark_cfgs = self.get_benchmark_cfgs(benchmark_name)

        self.logger.log('info', f'Evaluating {benchmark_name}...')

        dataloader = self.load_data(self.eval_cfgs, self.benchmark_cfgs)
        input_data = dataloader.load_data(task_list)    # Input_data: list[InferenceInput]
        
        inference_outputs = self.batch_inference(self.model, input_data)
        ref_answers = {task: self.get_ref_answer(input_data[task], inference_outputs[task]) for task in task_list}
        evaluation_details = {}
        evaluation_results = {}
        for task in task_list:
            evaluation_details[task], evaluation_results[task] = self.calculate_metrics(benchmark_name, inference_outputs[task], ref_answers[task], self.benchmark_cfgs["metrics"])

        overall_result = self.calculate_overall_metrics(result=evaluation_results)
        evaluation_results['Overall'] = overall_result
        self.display_benchmark_results(benchmark_name, evaluation_results)
        self.save_benchmark_details(self.output_path, benchmark_name, input_data, evaluation_details)
        return evaluation_details, evaluation_results
    
    def shutdown_model(self):
        """Shutdown model"""
        self.model.shutdown()
    
    def calculate_metrics(self, benchmark_name: str, inference_outputs: list[InferenceOutput], ref_answers: list[str], evaluate_tools: list[str], judge_methods: list[str], metrics_list: list[dict]):
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
        evaluation_details = [EvaluationResult(benchmark_name, inference_output, extracted_result, ref_answer, judge_methods) for inference_output, extracted_result, ref_answer in zip(inference_outputs, extracted_results, ref_answers)]
        metric_calculator = MetricCalculator(metrics_list)
        evaluation_results = metric_calculator(evaluation_details)
        return evaluation_details, evaluation_results

    def get_ref_answer(self, input_data: list[InferenceInput], inference_outputs: list[InferenceOutput]) -> list[any]:
        """Get reference answers from input data
        Args:
            input_data (list[InferenceInput]): input data
            inference_outputs (list[InferenceOutput]): inference outputs
            
        Returns:
            ref_answers (list[any]): reference answers
        """
        return [pair_data_via_uuid(input_data, inference_outputs)[0].ref_answer for input_data, inference_outputs in zip(input_data, inference_outputs)]
    
    def display_benchmark_results(self, benchmark_name: str, result: Dict[str, Dict[str, float]]):
        """Display single benchmark results in command line.
        Args:
            benchmark_name (str): benchmark name
            result (Dict[str, Dict[str, float]): evaluation result
        """
        self.logger.print_table(title=f'{benchmark_name} Benchmark', data=result, to_csv=True)
        self.logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        self.logger.log('info', f'Benchmark: {benchmark_name}')
        self.logger.log('info', f"model_id: {self.model_cfgs['model_id'][0]},")
        self.logger.log('info', f"num_fewshot: {self.eval_cfgs['num_fewshot'][0]},")
        self.logger.log('info', f"chain_of_thought: {self.eval_cfgs['chain_of_thought'][0]},")
        self.logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    def display_task_results(self, results: dict[str, dict[str, dict[str, float]]]):
        """Display overall evaluation results in command line.
        Args:
            results (list[dict[str, dict[str, float]]]): evaluation results
        """
        for benchmark_name, result in results.items():
            current_result = {benchmark_name: result['Overall']}
            self.logger.print_table(title=f'Brief Report of {benchmark_name}', data=current_result, to_csv=True, csv_file=f'{benchmark_name}_brief_report.csv')
    
    def save_task_details(self, save_path: str):
        """Save overall evaluation results, including all benchmark results.
        Args:
            save_path (str): save path
            results (list[EvaluationResult]): evaluation results
        """
        output_yaml = {
            'eval_cfgs': self.eval_cfgs,
            'model_cfgs': self.model_cfgs,
            'infer_cfgs': self.infer_cfgs,
            'results': self.results
        }
        with open(os.path.join(save_path, 'evaluation_details.yaml'), 'w') as f:
            yaml.dump(output_yaml, f)
    
    def calculate_overall_metrics(self, metric_name: str | None = None, result: dict[str, dict[str, float]] = None):
        """Calculate overall metrics
        Args:
            metric_name (str): metric name
            result (dict[str, dict[str, float]]): evaluation result
            
        Returns:
            overall_metrics (dict[str, float]): overall metrics
        """
        overall_metric_calculator = OverallMetricCalculator(metric_name)
        return overall_metric_calculator(result)
    
    @abstractmethod
    def save_benchmark_details(self, save_path: str, benchmark_name: str, inputs: Dict[str, List[InferenceInput]], results: Dict[str, List[EvaluationResult]]):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            benchmark_name (str): benchmark name
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        pass