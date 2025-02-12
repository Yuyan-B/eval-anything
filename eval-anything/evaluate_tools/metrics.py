from abc import ABC, abstractmethod
from typing import List, Iterable, Union
from eval_anything.evaluate_tools.base_tools import BaseTools

class MetricRegistry:
    _registry = {}

    @classmethod
    def register_metric(cls, name: str):
        def decorator(metric_cls):
            cls._registry[name] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def get_metric(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)
    
class MetricCalculator(BaseTools):
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        
    def apply(self):
        pass

        