from abc import ABC, abstractmethod
from typing import Optional
from eval_anything.utils.data_type import InferenceInput
from hashlib import sha256
from eval_anything.utils.data_type import ModalityType

class MultiChoicePromptBuilder():
    def __init__(self, candidate_labels: list[str], multi_choice_prompt: Optional[str] = None, cot_context: Optional[str] = None, few_shot_examples: Optional[list[str]] = None, cot: bool = False):
        self.multi_choice_prompt = multi_choice_prompt if multi_choice_prompt else "The following are multiple choice questions (with answers)."
        self.candidate_labels = candidate_labels
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = "") -> str:
        prompt = f"{self.multi_choice_prompt}\n\n{question}\n"
        for label, answer in zip(self.candidate_labels, candidate_answers):
            prompt += f"({label}) {answer} "
        
        answer = f"\nAnswer: {self.cot_context} {ground_truth}" if self.cot else f"\nAnswer: {ground_truth}"
        return prompt + answer

    def build_prompt(self, question: str, candidate_answers: list[str]) -> str:
        context = ""
        for few_shot in self.few_shot_examples:
            context += self.marge_QA(few_shot['question'], few_shot['candidate_answers'], few_shot['ground_truth'])
        context = context.join("\n") if context else ""

        question = self.marge_QA(question, candidate_answers)
        prompt = f"{self.multi_choice_prompt}\n\n{context}{question}"
        return prompt


class DialoguePromptBuilder():
    def __init__(self, few_shot_examples: Optional[list[str]] = None, cot_context: Optional[str] = None, cot: bool = False):
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.cot = cot

    def marge_QA(self, question: str, ground_truth: str = "") -> str:
        prompt = f"{question}\n"
        answer = f"\nAnswer: {self.cot_context} {ground_truth}" if self.cot else f"\nAnswer: {ground_truth}"
        return prompt + answer

    def build_prompt(self, question: str) -> str:
        context = ""
        for few_shot in self.few_shot_examples:
            context += self.marge_QA(few_shot['question'], few_shot['ground_truth'])
        context = context.join("\n") if context else ""

        question = self.marge_QA(question)
        prompt = f"{self.multi_choice_prompt}\n\n{context}{question}"
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
        for modality, data in modality_dict.items():
            uuid_dict[modality] = self.generate_uuid(data, modality)

        data.uuid = uuid_dict
        return data

    # TODO 根据data和modality生成不同模态的uuid
    def generate_uuid(self, data: str, modality: str) -> str:
        pass