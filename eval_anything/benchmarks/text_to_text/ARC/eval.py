"""
继承自eval-anything.pipeline.t2t_task
"""
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from collections import namedtuple
from eval_anything.utils.data_type import EvaluationResult
from eval_anything.evaluate_tools.t2t_tools import RegexMatchLetter
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry
from eval_anything.utils.cache_manager import CacheManager

@BenchmarkRegistry.register('ARC')
class ARCBenchmark(T2TBenchmark):
    def __init__(self, 
                 model: BaseModel, 
                 eval_cfgs: namedtuple, 
                 model_cfgs: namedtuple, 
                 infer_cfgs: namedtuple, 
                 output_path: str,
                 cache_manager: CacheManager,
                 logger: EvalLogger):
        super().__init__(model, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger)
        self.benchmark_name = "ARC"
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def run(self, 
            task_list: list[str],) -> tuple[dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Run benchmark
        Args:
            task_list (list[str]): task list
            
        Returns:
            evaluation_details (dict[str, list[EvaluationResult]]): evaluation details
            evaluation_results (dict[str, dict[str, float]]): evaluation results
        """
        self.logger.log('info', f'Evaluating {self.benchmark_name}...')

        dataloader = self.init_dataloader(self.eval_cfgs, self.benchmark_cfgs)
        input_data = dataloader.load_dataset(task_list)
        
        inference_outputs = self.batch_inference(self.model, input_data)
        ref_answers = {task: self.get_ref_answer(input_data[task], inference_outputs[task]) for task in task_list}
        evaluation_details = {}
        evaluation_results = {}
        num_to_char = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E'}

        for task in task_list:
            ground_truth = [num_to_char[ref_answer] for ref_answer in ref_answers[task]]
            evaluation_details[task], evaluation_results[task] = self.calculate_metrics(self.benchmark_name, inference_outputs[task], ground_truth, self.benchmark_cfgs.answer_extractor, self.benchmark_cfgs.metrics)

        if len(task_list) > 1:
            overall_result = self.calculate_overall_metrics(self.benchmark_cfgs.overall_metrics, result=evaluation_results)
        else:
            overall_result = {}

        self.display_benchmark_results(self.benchmark_name, evaluation_results)
        if overall_result != {}:
            self.display_benchmark_results(self.benchmark_name, overall_result)
        self.save_benchmark_details(self.output_path, self.benchmark_name, input_data, evaluation_details)
        return evaluation_details, evaluation_results, overall_result