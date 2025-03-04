from abc import ABC, abstractmethod
from typing import Optional
from eval_anything.utils.data_type import InferenceInput, InferenceOutput, EvaluationResult
from hashlib import sha256
from eval_anything.utils.data_type import ModalityType
import os
import json
import yaml
from typing import Any
from collections import namedtuple
from pathlib import Path

BENCHMARK_MODALITY_MAP = {
    'gsm8k': 'text_to_text',
    'mmlu': 'text_to_text',
    'truthfulqa': 'text_to_text',
    'humaneval': 'text_to_text',
    'agieval': 'text_to_text',
    'beavertails': 'text_to_text',
    'mmmu': 'text_image_to_text',
}

class MultiChoicePromptBuilder():
    def __init__(self, candidate_labels: list[str], multi_choice_prompt: Optional[str] = None, cot_context: Optional[str] = None, few_shot_examples: Optional[list[str]] = None, cot: bool = False):
        self.multi_choice_prompt = multi_choice_prompt if multi_choice_prompt else "The following are multiple choice questions (with answers)."
        self.candidate_labels = candidate_labels
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = "") -> str:
        # Handle the label answer format difference
        if ground_truth.isdigit():
            ground_truth = chr(65 + int(ground_truth))
        prompt = f"{question}\n"
        for label, answer in zip(self.candidate_labels, candidate_answers):
            prompt += f"({label}) {answer} "
        
        answer = f"\nAnswer: {self.cot_context} {ground_truth}" if self.enable_cot else f"\nAnswer: {ground_truth}"
        return prompt + answer + "\n"

    def build_prompt(self, question: str, candidate_answers: list[str]) -> str:
        prompt = f"{self.multi_choice_prompt}\n\n"
        
        if self.few_shot_examples:
            # Add few-shot examples
            for q, c, a in zip(self.few_shot_examples['question'], 
                             self.few_shot_examples['choices'],
                             self.few_shot_examples['answer']):
                prompt += self.marge_QA(q, c, str(a))

        # Add the current question
        prompt += self.marge_QA(question, candidate_answers)
        return prompt


class DialoguePromptBuilder():
    def __init__(self, few_shot_examples: Optional[list[str]] = None, cot_context: Optional[str] = None, cot: bool = False):
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, ground_truth: str = "") -> str:
        prompt = f"{question}\n"
        answer = f"\nAnswer: {self.cot_context} {ground_truth}" if self.enable_cot else f"\nAnswer: {ground_truth}"
        return prompt + answer

    def build_prompt(self, input: str) -> str:
        context = ""
        if self.few_shot_examples:
            for question, answer in zip(self.few_shot_examples['question'], self.few_shot_examples['answer']):
                context += self.marge_QA(question, answer)
            context = context + "\n" if context else ""

        question = self.marge_QA(input)
        prompt = f"{context}{question}"
        return prompt


class UUIDGenerator():
    
    def __init__(self):
        pass

    def __call__(self, data: InferenceInput) -> InferenceInput:
        modality_dict = {modality.value: "" for modality in ModalityType}
        
        modality_dict = {"text": data.text}
        for mm_data in data.mm_data:
            modality_dict[mm_data.modality] = mm_data.url if mm_data.url else mm_data.file

        uuid_dict = {}
        for modality, content in modality_dict.items():
            uuid_dict[modality] = self.generate_uuid(content, modality)
        return uuid_dict

    # TODO 根据data和modality生成不同模态的uuid
    def generate_uuid(self, data: str, modality: str) -> str:
        if modality == 'text':
            return sha256(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported modality: {modality}")

def read_cfgs_from_yaml(yaml_relative_dir: str, yaml_name: str) -> dict[str, Any]:
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    yaml_path = os.path.join(parent_path, yaml_relative_dir, yaml_name)
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc
    # if backend.lower() == 'vllm':
    #     infer_cfgs_path = os.path.join(
    #         parent_path,
    #         'configs',
    #         'evaluation',
    #         'vllm',
    #         configs['infer_cfgs']['vllm_cfgs'],
    #     )
    # else:
    #     infer_cfgs_path = os.path.join(
    #         parent_path,
    #         'configs',
    #         'evaluation',
    #         'deepspeed',
    #         configs['infer_cfgs']['ds_cfgs'],
    #     )
    # with open(infer_cfgs_path) as f:
    #     infer_cfgs = json.load(f)

    return configs

def update_dict(total_dict: dict[str, Any], item_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in total_dict.items():
        if key in item_dict:
            total_dict[key] = item_dict[key]
        if isinstance(value, dict):
            update_dict(value, item_dict)
    return total_dict

def is_convertible_to_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict."""
    if value == 'True':
        value = True
    elif value == 'False':
        value = False
    elif is_convertible_to_float(value):
        value = float(value)
    elif value.isdigit():
        value = int(value)
    elif value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
        value = value.split(',')
        value = list(filter(None, value))
    elif ',' in value:
        value = value.split(',')
        value = list(filter(None, value))
    else:
        value = str(value)
    keys_split = key_list.replace('-', '_').split(':')
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace('-', '_'): return_dict}

def dict_to_namedtuple(dic):
    def convert(value):
        if isinstance(value, dict):
            return dict_to_namedtuple(value)
        elif isinstance(value, list):
            return [convert(item) for item in value]
        else:
            return value

    class EnhancedNamedTuple(namedtuple('configs', dic.keys())):
        __slots__ = ()

        def __getattr__(self, item):
            return None

    cfgs = EnhancedNamedTuple(**{k: convert(v) for k, v in dic.items()})
    return cfgs

def namedtuple_to_dict(obj):
    if hasattr(obj, '_asdict'):
        return {k: namedtuple_to_dict(v) for k, v in obj._asdict().items()}
    elif isinstance(obj, list):
        return [namedtuple_to_dict(i) for i in obj]
    else:
        return obj

def parse_unknown_args(args):
    """
    Parse unknown arguments, support no-value parameters and return dict.
    """
    unknown_args = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            key = arg[2:]
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                unknown_args[key] = args[i + 1]
                i += 2
            else:
                unknown_args[key] = True
                i += 1
        else:
            i += 1
    return unknown_args

def pair_data_via_uuid(inputs: list[InferenceInput], outputs: list[InferenceOutput | EvaluationResult]):
    def get_uuid_key(item: InferenceInput):
        uuid_key = ""
        for modality, uuid in item.uuid.items():
            uuid_key += f"{modality}:{uuid}"
        return uuid_key
    uuid_inputs = {get_uuid_key(item): item for item in inputs}
    results = []
    for output in outputs:
        uuid_key = get_uuid_key(output)
        if uuid_key in uuid_inputs:
            results.append((uuid_inputs[uuid_key], output))
    return results

def get_messages(modality, prompt):
    messages = {
        "t2t": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "ti2t": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ],
    }
    return messages.get(modality, [])


def get_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return None
