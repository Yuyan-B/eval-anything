import importlib

benchmark_modules = [
    "benchmarks.text_to_text.gsm8k.eval",
<<<<<<< HEAD
    "benchmarks.text_image_to_text.mmmu.eval",
=======
    "benchmarks.text_to_text.mmlu.eval",
    "benchmarks.text_to_text.HumanEval.eval",
>>>>>>> refs/remotes/origin/main
]

for module in benchmark_modules:
    importlib.import_module(module)