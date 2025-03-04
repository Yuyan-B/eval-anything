import importlib

benchmark_modules = [
    "benchmarks.text_to_text.gsm8k.eval",
    "benchmarks.text_to_text.mmlu.eval",
    "benchmarks.text_to_text.HumanEval.eval",
]

for module in benchmark_modules:
    importlib.import_module(module)