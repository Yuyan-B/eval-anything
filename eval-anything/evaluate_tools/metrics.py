from abc import ABC, abstractmethod
from typing import List, Iterable, Union
from eval_anything.evaluate_tools.base_tools import BaseTool, BaseMetric
from eval_anything.utils.register import MetricRegistry

class MetricCalculator(BaseTool):
    def __init__(self, metrics_list: List[str]):
        self.metrics = []
        for metric in metrics_list:
            self.metrics.append(MetricRegistry.get_metric(metric))
        
    def apply(self, preds: Union[List, Iterable], targets: Union[List, Iterable]):
        results = {}
        for metric in self.metrics:
            results[metric.__name__] = metric(preds, targets)
        return results
    
@MetricRegistry.register('accuracy')
class Accuracy(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable]) -> float:
        for pred, target in zip(preds, targets):
            if pred == target:
                correct += 1
        return correct / len(preds)

@MetricRegistry.register('false_positive_rate')
class FalsePositiveRate(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        false_positive = 0
        for pred, target in zip(preds, targets):
            if pred == positive_label and target != positive_label:
                false_positive += 1
        return false_positive / len(preds)

@MetricRegistry.register('false_negative_rate')
class FalseNegativeRate(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        false_negative = 0
        for pred, target in zip(preds, targets):
            if pred != positive_label and target == positive_label:
                false_negative += 1
        return false_negative / len(preds)

@MetricRegistry.register('true_positive_rate')
class TruePositiveRate(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        true_positive = 0
        for pred, target in zip(preds, targets):
            if pred == positive_label and target == positive_label:
                true_positive += 1
        return true_positive / len(preds)

@MetricRegistry.register('true_negative_rate')
class TrueNegativeRate(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        true_negative = 0
        for pred, target in zip(preds, targets):
            if pred != positive_label and target != positive_label:
                true_negative += 1
        return true_negative / len(preds)

@MetricRegistry.register('precision')
class Precision(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        true_positive = TruePositiveRate(preds, targets, positive_label)
        false_positive = FalsePositiveRate(preds, targets, positive_label)
        return true_positive / (true_positive + false_positive)

@MetricRegistry.register('recall')
class Recall(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        true_positive = TruePositiveRate(preds, targets, positive_label)
        false_negative = FalseNegativeRate(preds, targets, positive_label)
        return true_positive / (true_positive + false_negative)

@MetricRegistry.register('f1_score')
class F1Score(BaseMetric):
    def calculate(self, preds: Union[List, Iterable], targets: Union[List, Iterable], positive_label) -> float:
        precision = Precision(preds, targets, positive_label)
        recall = Recall(preds, targets, positive_label)
        return 2 * precision * recall / (precision + recall)

