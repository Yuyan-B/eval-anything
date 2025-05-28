from eval_anything.utils.data_type import InferenceInput, InferenceOutput, EvaluationResult
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from collections import namedtuple
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.logger import EvalLogger
from datasets import load_dataset
from eval_anything.utils.cached_requests import cached_requests
from eval_anything.benchmarks.text_to_text.DoAnythingNow.utils import (
    EVALUATE_PROMPT,
    check_eval_response,
)
from typing import Optional
import os
from eval_anything.utils.register import BenchmarkRegistry


def gpt_evaluate(
    question: str,
    response: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """
    Extract an answer from a model response for a specific item.

    Args:
        question: Question(jailbreak question)
        response: Model's response
        model: Name of the extractor model (e.g., "gpt-4o-mini")
        api_key: API key for the extractor model
        api_base: Base URL for the extractor model API

    Returns:
        Extracted answer
    """

    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE")

    user_prompt = EVALUATE_PROMPT.format(question=question, response=response)

    messages = [{"role": "user", "content": user_prompt}]

    extraction = cached_requests(
        messages=messages,
        model=model,
        max_completion_tokens=1024,
        temperature=0.0,
        api_key=api_key,
        api_base=api_base,
    )
    return extraction


@BenchmarkRegistry.register('DoAnythingNow')
class DoAnythingNow(T2TBenchmark):
    def __init__(
        self,
        model: BaseModel,
        eval_cfgs: namedtuple,
        model_cfgs: namedtuple,
        infer_cfgs: namedtuple,
        output_path: str,
        cache_manager: CacheManager,
        logger: EvalLogger,
    ):
        super().__init__(
            model, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger
        )
        self.benchmark_name = "DoAnythingNow"
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        dataset = load_dataset(
            path="TrustAIRLab/in-the-wild-jailbreak-prompts",
            name='jailbreak_2023_12_25',
            split="train",
        )

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task="DoAnythingNow",
                    conversation=[{"role": "user", "content": item["prompt"]}],
                    ref_answer=None,
                    metadata={
                        "prompt": item["prompt"],
                    },
                )
            )

        return {'DoAnythingNow': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        for input_item, output_item in zip(
            inference_inputs['DoAnythingNow'], inference_outputs['DoAnythingNow']
        ):
            jailbreak_question = input_item.metadata["prompt"]
            llm_response = output_item.response
            eval_response = gpt_evaluate(jailbreak_question, llm_response)

            try:
                eval_data = check_eval_response(llm_response, eval_response)
                label = eval_data["label"]
            except Exception as e:
                label = None

            output_item.label = label

        return inference_outputs

    def to_EvaluationResult(
        self, task_list: list[str], inference_outputs: dict[str, list[InferenceOutput]]
    ) -> tuple[
        dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]
    ]:
        """
        Convert a InferenceOutput dict instances to evaluation details, evaluation results, and
        overall results
        """
        jailbreak_count = 0
        for output_item in inference_outputs['DoAnythingNow']:
            if output_item.lablel:
                jailbreak_count += 1

        asr = jailbreak_count / len(inference_outputs['DoAnythingNow'])

        return (
            {"DoAnythingNow": inference_outputs},
            {"DoAnythingNow": {"Attack Success Rate": asr}},
            {},
        )
