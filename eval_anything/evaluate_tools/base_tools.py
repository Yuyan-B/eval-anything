from abc import ABC, abstractmethod
from eval_anything.utils.logger import EvalLogger

metric_logger = EvalLogger(name="MetricLogger")
tool_logger = EvalLogger(name="ToolLogger")

class BaseTool(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = tool_logger
        
    @abstractmethod
    def apply(self, **kwargs):
        pass
    
    def __call__(self, **kwargs):
        return self.apply(**kwargs)

class BaseMetric(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = metric_logger
            
    @abstractmethod
    def calculate(self):
        pass
    
    def __call__(self, **kwargs):
        return self.calculate(**kwargs)
