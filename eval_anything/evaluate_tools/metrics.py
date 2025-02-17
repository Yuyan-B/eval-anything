from abc import ABC, abstractmethod
from typing import List, Iterable, Union
from eval_anything.evaluate_tools.base_tools import BaseTool, BaseMetric
from eval_anything.utils.register import MetricRegistry
from eval_anything.utils.data_type import EvaluationResult

# TODO 为每个metric提供一个_repr_
class MetricCalculator(BaseTool):
    def __init__(self, metrics_list: List[dict]):
        self.metrics = []
        self.metrics_args = []
        for metric in metrics_list:
            self.metrics.append(MetricRegistry.get_metric(metric['function']))
            self.metrics_args.append(metric['args'])
        
    def apply(self, evaluation_results: List[EvaluationResult]):
        results = {}
        for metric, args in zip(self.metrics, self.metrics_args):
            results[metric.__name__] = metric(evaluation_results, **args)
        return results
    
@MetricRegistry.register('accuracy')
class Accuracy(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        correct = 0
        for evaluation_result in evaluation_results:
            if evaluation_result.extracted_result.lower().strip() == evaluation_result.ground_truth.lower().strip():
                correct += 1
        return correct / len(evaluation_results)

@MetricRegistry.register('false_positive_rate')
class FalsePositiveRate(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        false_positive = 0
        positive_label = kwargs.get('positive_label', 'true')
        for evaluation_result in evaluation_results:
            if evaluation_result.extracted_result.lower().strip() == positive_label and evaluation_result.ground_truth.lower().strip() != positive_label:
                false_positive += 1
        return false_positive / len(evaluation_results)

@MetricRegistry.register('false_negative_rate')
class FalseNegativeRate(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        false_negative = 0
        positive_label = kwargs.get('positive_label', 'true')
        for evaluation_result in evaluation_results:
            if evaluation_result.extracted_result.lower().strip() != positive_label and evaluation_result.ground_truth.lower().strip() == positive_label:
                false_negative += 1
        return false_negative / len(evaluation_results)

@MetricRegistry.register('true_positive_rate')
class TruePositiveRate(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        true_positive = 0
        positive_label = kwargs.get('positive_label', 'true')
        for evaluation_result in evaluation_results:
            if evaluation_result.extracted_result.lower().strip() == positive_label and evaluation_result.ground_truth.lower().strip() == positive_label:
                true_positive += 1
        return true_positive / len(evaluation_results)

@MetricRegistry.register('true_negative_rate')
class TrueNegativeRate(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        true_negative = 0
        positive_label = kwargs.get('positive_label', 'true')
        for evaluation_result in evaluation_results:
            if evaluation_result.extracted_result.lower().strip() != positive_label and evaluation_result.ground_truth.lower().strip() != positive_label:
                true_negative += 1
        return true_negative / len(evaluation_results)

@MetricRegistry.register('precision')
class Precision(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        true_positive = TruePositiveRate(evaluation_results, **kwargs)
        false_positive = FalsePositiveRate(evaluation_results, **kwargs)
        return true_positive / (true_positive + false_positive)

@MetricRegistry.register('recall')
class Recall(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        true_positive = TruePositiveRate(evaluation_results, **kwargs)
        false_negative = FalseNegativeRate(evaluation_results, **kwargs)
        return true_positive / (true_positive + false_negative)

@MetricRegistry.register('f1_score')
class F1Score(BaseMetric):
    def calculate(self, evaluation_results: List[EvaluationResult], **kwargs) -> float:
        precision = Precision(evaluation_results, **kwargs)
        recall = Recall(evaluation_results, **kwargs)
        return 2 * precision * recall / (precision + recall)

