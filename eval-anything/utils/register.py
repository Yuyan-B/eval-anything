"""
注册器，实现metric和template的注册
"""

class MetricRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_cls):
            cls._registry[name] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def get_metric(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)
    
class TemplateRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(template_cls):
            cls._registry[name] = template_cls
            return template_cls
        return decorator

    @classmethod
    def get_template(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Template '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)