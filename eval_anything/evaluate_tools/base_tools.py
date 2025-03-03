from abc import ABC, abstractmethod
from typing import List

class BaseTool(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def apply(self, **kwargs):
        pass
    
    def __call__(self, **kwargs):
        return self.apply(**kwargs)

class BaseMetric(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    @abstractmethod
    def calculate(self):
        pass
    
    def __call__(self, **kwargs):
        return self.calculate(**kwargs)
