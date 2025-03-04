"""
评估工具包（包括各种指标的计算以及指定pattern的提取）
"""
from abc import ABC, abstractmethod
from eval_anything.evaluate_tools.base_tools import BaseTool
from typing import Union, List, Iterable

T2T_EXTRACTOR_MAP = {
    "regex_match_number": "RegexMatchNumber",
    "regex_match_letter": "RegexMatchLetter",
    "regex_match_code": "RegexMatchCode"
}

T2T_JUDGER_MAP = {
    "judge_equal": "JudgeEqual",
}


class RegexMatch(BaseTool):
    def __init__(self, pattern: str, match_index: int = None):
        self.pattern = pattern
        self.match_index = match_index  

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """
        Match the specified pattern in the text.
        Args:
            data (list/iterable): the text to be matched
        Returns:
            list/None: the matched result
        """
        def match_text(text):
            import re
            pattern = re.compile(self.pattern)
            match = list(pattern.finditer(text))
            if match:
                if self.match_index is not None:
                    return match[self.match_index].group()
                else:
                    return match.group()
            else:
                return None
        matches = [match_text(item) for item in data]
        return matches
    
    def __call__(self, data: Union[List, Iterable]) -> Union[List, None]:
        return self.apply(data)
    
class RegexMatchNumber(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        pattern_match_number = r"(?:[+-]?(?:\d+/\d+|(?:\d*\.\d+)|\d+)|√(?:\([+-]?(?:\d+/\d+|(?:\d*\.\d+)|\d+)\)|[+-]?(?:\d+/\d+|(?:\d*\.\d+)|\d+)))"
        self.pattern = additional_pattern.format(original_pattern=pattern_match_number) if additional_pattern else pattern_match_number
        self.match_index = match_index
        
class RegexMatchText(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        # Pattern to match single letter answers A, B, C, D (case insensitive)
        pattern_match_text = r"[A-Da-d]"
        self.pattern = additional_pattern.format(original_pattern=pattern_match_text) if additional_pattern else pattern_match_text
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """
        Match letter answers in the text and convert to uppercase.
        Args:
            data (list/iterable): the text to be matched
        Returns:
            list/None: the matched result in uppercase
        """
        matches = super().apply(data)
        # Convert matched letters to uppercase for consistency
        return [match.upper() if match else None for match in matches]

class JudgeEqual(BaseTool):
    def __init__(self):
        super().__init__()
    
    def apply(self, data_1, data_2) -> bool:
        return data_1 == data_2
    
    def __call__(self, data_1, data_2) -> bool:
        return self.apply(data_1, data_2)

class RegexMatchLetter(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        # Base pattern to match letters in parentheses (A-D)
        pattern_match_letter = r"\(([A-Da-d])\)"  # Capture the letter between parentheses
        self.pattern = additional_pattern.format(original_pattern=pattern_match_letter) if additional_pattern else pattern_match_letter
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        def match_text(text):
            import re
            pattern = re.compile(self.pattern, re.IGNORECASE)
            match = list(pattern.finditer(text))
            if match:
                if self.match_index is not None:
                    # Extract just the letter part (group 1) and convert to uppercase
                    return match[self.match_index].group(1).upper()
                else:
                    return match.group(1).upper()
            else:
                return None
        matches = [match_text(item) for item in data]
        return matches

class RegexMatchCode(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        # Pattern to match code blocks between ```python ``` or ``` ```
        pattern_match_code = r"```(?:python)?\s*(.*?)\s*```"
        self.pattern = additional_pattern.format(original_pattern=pattern_match_code) if additional_pattern else pattern_match_code
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        def match_text(text):
            import re
            pattern = re.compile(self.pattern, re.DOTALL)  # Use DOTALL to match across newlines
            matches = list(pattern.finditer(text))
            if matches:
                if self.match_index is not None:
                    # Handle negative index (last match)
                    if self.match_index < 0:
                        self.match_index = len(matches) + self.match_index
                    if 0 <= self.match_index < len(matches):
                        return matches[self.match_index].group(1).strip()
                    return None
                else:
                    return matches[0].group(1).strip()
            return None
            
        matches = [match_text(item) for item in data]
        return matches
    


