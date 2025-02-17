"""
评估工具包（包括各种指标的计算以及指定pattern的提取）
"""
from abc import ABC, abstractmethod
from eval_anything.evaluate_tools.base_tools import BaseTool
from typing import Union, List, Iterable

T2T_EVALUATE_TOOLS_MAP = {
    "regex_match_number": "RegexMatchNumber",
    "regex_match_text": "RegexMatch",
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
            match = pattern.finditer(text)
            if match:
                if self.match_index:
                    return match.group(self.match_index)
                else:
                    return match.group()
            else:
                return None
        matches = [match_text(item) for item in data]
        return matches
    
class RegexMatchNumber(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        pattern_match_number = r"^(?:[+-]?(?:\d+/\d+|(?:\d*\.\d+|\d+\.\d*)|\d+)|√(?:\([+-]?(?:\d+/\d+|(?:\d*\.\d+|\d+\.\d*)|\d+)\)|[+-]?(?:\d+/\d+|(?:\d*\.\d+|\d+\.\d*)|\d+)))$"
        self.pattern = additional_pattern.format(original_pattern=pattern_match_number) if additional_pattern else pattern_match_number
        self.match_index = match_index
        
class JudgeEqual(BaseTool):
    def __init__(self):
        super().__init__()
    
    def apply(self, data_1, data_2) -> bool:
        return data_1 == data_2