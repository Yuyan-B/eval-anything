import importlib

benchmark_modules = [
    "benchmarks.text_to_text.gsm8k.eval",
    "benchmarks.text_image_to_text.mmmu.eval",
]

for module in benchmark_modules:
    importlib.import_module(module)