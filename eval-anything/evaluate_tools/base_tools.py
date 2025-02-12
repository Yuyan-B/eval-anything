from abc import ABC, abstractmethod

class BaseTools(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def apply(self, **kwargs):
        pass
    
    def __call__(self, **kwargs):
        return self.apply(**kwargs)

class BaseMetrics(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
            
                
