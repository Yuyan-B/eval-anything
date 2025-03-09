"""
继承自eval-anything.pipeline.t2t_task
"""
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from collections import namedtuple
from eval_anything.utils.data_type import EvaluationResult
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry
from eval_anything.utils.cache_manager import CacheManager
from collections import defaultdict
import torch

@BenchmarkRegistry.register('TruthfulQA')
class TruthfulQABenchmark(T2TBenchmark):
    def __init__(self, 
                 model: BaseModel, 
                 eval_cfgs: namedtuple, 
                 model_cfgs: namedtuple, 
                 infer_cfgs: namedtuple, 
                 output_path: str,
                 cache_manager: CacheManager,
                 logger: EvalLogger):
        super().__init__(model, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger)
        self.benchmark_name = "TruthfulQA"
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
        evaluation_details = {}
        evaluation_results = {}
        
        ref_answers = defaultdict(list)
        for task in task_list:
            scores = defaultdict(list)
            for input, output in zip(input_data[task], inference_outputs[task]):
                ref_answers[task].append(input.ref_answer)
                log_probs = torch.tensor([
                    list(token_probs.values())[0].logprob
                    for token_probs in output.response_logprobs[0]
                ])
                log_probs = log_probs[3:]
                scores[input.text_id].append(log_probs.sum().item())
                if input.text_id != 0 and input.text_id % 2 == 0:
                    output.response_logprobs = {
                        'scores_true': scores[input.text_id - 2],
                        'scores_false': scores[input.text_id - 1]
                    }
                else:
                    output.response_logprobs = None

        for task in task_list:
            # judge_method = 'judge_mc1'
            judge_method = 'judge_mc2'
            evaluation_details[task], evaluation_results[task] = self.calculate_metrics(self.benchmark_name, inference_outputs[task], ref_answers[task], self.benchmark_cfgs.answer_extractor, self.benchmark_cfgs.metrics, judge_method=judge_method)

        if len(task_list) > 1:
            overall_result = self.calculate_overall_metrics(self.benchmark_cfgs.overall_metrics, result=evaluation_results)
        else:
            overall_result = {}

        self.display_benchmark_results(self.benchmark_name, evaluation_results)
        if overall_result != {}:
            self.display_benchmark_results(self.benchmark_name, overall_result)
        self.save_benchmark_details(self.output_path, self.benchmark_name, input_data, evaluation_details)
        return evaluation_details, evaluation_results, overall_result