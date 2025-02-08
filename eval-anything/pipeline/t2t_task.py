"""
t2t任务基类，不直接使用，而是继承后实现具体任务的逻辑
输入：
    - 数据集路径
    - 模型路径
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型推理方式(调用eval-anything/models中的推理方式)
    - ...
输出：
    - EvaluationResult类
"""