
# 开发文档

## 项目简介

### 评测任务流

- `eval_anything/pipeline/base_task.py`: 实现了从读取评测参数到保存评测结果的全流程，一般不做改动

### benchmark相关

- `eval_anything/benchmark/text-to-text/gsm8k/eval.py`: 继承自`t2t_benchmark.py`中的`T2TBenchmark`基类，实现每个benchmark评测流程的适配，包括读取benchmark设置、读取数据集、模型推理、结果评测、结果保存。一般而言只需根据需要重写部分函数。

- `eval_anything/benchmark/text-to-text/gsm8k/configs.yaml`: 包含了benchmark相关的各种设置

### dataloader相关

- `eval_anything/dataloader/t2t_dataloader.py`: 继承自`base_dataloader.py`中的`BaseDataLoader`基类，实现评测数据的读取、拼接等操作

### inference相关

- `eval_anything/models`: 文件夹下实现了不同backend的infer

### evaluate相关

- `eval_anything/evaluate_tools/metrics.py`: 实现了常用的metric计算，若有相关需求直接使用现成的，否则请在metric中进行添加（尽量不要在benchmark中单独实现metric）

- `eval_anything/evaluate_tools/t2t_tools.py`: 实现了常用评测工具的集成，例如正则匹配、判断相同等，若有相关需求直接使用现成的，否则考虑该需求的泛用性，尽量以面向对象的形式集成进去

### 其他（`eval_anything/utils/`）

- `utils.py`: 项目开发常用工具包

- `data_type.py`: 常用数据类型，用来实现不同模块之间的数据传递

- `cache_manager.py`: cache机制设计

- `regiser.py`: 实现了常用的注册表（template, metric, benchmark...）

- `logger.py`: 实现了log机制的开发

## 开发指南

1. 在`benchmarks`文件夹下进行相关文件夹创建
    - 配置参数文件 `configs.yaml`
    - 评测代码 `eval.py`: 继承自benchmark基类进行开发
2. 如有新的metric以及evaluate_tools请在`evaluate_tools`下进行新增
3. 在`eval_anything/configs/evaluate.yaml`文件夹下进行评测任务配置
4. 运行`eval_anything/scripts/run.sh`进行测试


## Tips

- 尽量遵循奥卡姆剃刀原则进行开发，减少对现有函数的重写
- 如有需要新增的依赖库，请在`pyproject.toml`中进行添加
- 对现有基类有任何建议欢迎提出!