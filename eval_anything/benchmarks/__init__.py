import importlib

benchmark_modules = [
    "benchmarks.text_to_text.gsm8k.eval",
]

for module in benchmark_modules:
    importlib.import_module(module)