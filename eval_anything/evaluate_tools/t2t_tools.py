"""
Evaluation tools package (including calculation of various metrics and extraction of specified patterns)
"""
from abc import ABC, abstractmethod
from eval_anything.evaluate_tools.base_tools import BaseTool
from typing import Union, List, Iterable, Optional
import numpy as np
import re

T2T_EXTRACTOR_MAP = {
    "regex_match_number": "RegexMatchNumber",
    "regex_match_letter": "RegexMatchLetter",
    "regex_match_multi_letter": "RegexMatchMultiLetter",
    "regex_match_code": "RegexMatchCode",
    "regex_match_math": "RegexMatchMath",
    "regex_match_multi_open": "RegexMatchMultiOpen",
}

T2T_JUDGER_MAP = {
    "judge_equal": "JudgeEqual",
    "judge_equal_list": "JudgeEqualList",
    "judge_mc1": "JudgeMC1",
    "judge_mc2": "JudgeMC2",
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

    def apply(self, data: Union[List[str], Iterable[str]]) -> List[Union[str, None]]:
        import re
        from eval_anything.utils.utils import _strip_string, remove_few_shot_prefix, remove_boxed
        def match_text(text):
            if "\\boxed" in text:
                return remove_boxed(text)
            matches = re.findall(self.pattern, text)
            return matches[self.match_index] if matches else None

        return [_strip_string(match_text(remove_few_shot_prefix(item))) for item in data]

    def __call__(self, data: Union[List[str], Iterable[str]]) -> List[Union[str, None]]:
        return self.apply(data)
        
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
    
class JudgeEqualList(BaseTool):
    def __init__(self):
        super().__init__()

    def apply(self, data_1, data_2) -> bool:
        """Compare model answer list with ground truth
        
        Args:
            data_1: Model answer list
            data_2: Ground truth (str representing float, list of str, or str)
            
        Returns:
            bool: True if any ground truth matches any answer in model's list
        """
        if data_1 is None:
            return False
        
        try:
            gold_answer = eval(data_2)
        except:
            gold_answer = data_2

        if isinstance(gold_answer, list):
            correct = False
            for gold_answer_i in gold_answer:
                for pred_answer in data_1:
                    if isinstance(pred_answer, str):
                        if isinstance(gold_answer_i, str):
                            if gold_answer_i in pred_answer:
                                correct = True
                                break
                    else:
                        try:
                            gold_answer_i = float(gold_answer_i)
                        except:
                            gold_answer_i = gold_answer_i
                        if gold_answer_i == pred_answer:
                            correct = True
                            break
            return correct
        else:
            return gold_answer in data_1

    def __call__(self, data_1, data_2) -> bool:
        return self.apply(data_1, data_2)
    

class RegexMatchLetter(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        # Base pattern to match letters in parentheses (A-Z)
        pattern_match_letter = r"\(([A-Za-z])\)"  # Capture the letter between parentheses
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

class RegexMatchMultiLetter(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        pattern_match_letter = r"\(([A-Za-z])\)"
        self.pattern = additional_pattern.format(original_pattern=pattern_match_letter) if additional_pattern else pattern_match_letter
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        def match_text(text):
            import re
            pattern = re.compile(self.pattern, re.IGNORECASE)
            match = list(pattern.finditer(text))
            if match:
                if self.match_index is not None:
                    return [match[self.match_index].group(1).upper()]
                else:
                    return [match.group(1).upper()]
            else:
                return None
        matches = [match_text(item) for item in data]
        return matches
    
class RegexMatchCode(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None, language: str = "python"):
        # Pattern to match code blocks between ```python ``` or ``` ```
        pattern_match_code = r"```(?:{language})?\s*([\s\S]*?)\s*```"
        self.pattern = additional_pattern.format(original_pattern=pattern_match_code) if additional_pattern else pattern_match_code
        self.match_index = match_index
        self.language = language

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        def match_text(text):
            import re
            # Find all positions of ```language and ``` markers
            language_pattern = r"```{}".format(self.language)
            close_pattern = r"```"
            language_positions = [m.start() for m in re.finditer(language_pattern, text)]
            close_positions = [m.start() for m in re.finditer(close_pattern, text)]
            
            if not language_positions or not close_positions:
                return ""
            
            # Match each ```language with its next ``` marker
            code_blocks = []
            for lang_start in language_positions:
                # Find the next closing marker after this language marker
                next_close = None
                for close_pos in close_positions:
                    if close_pos > lang_start:
                        next_close = close_pos
                        break
                
                if next_close is not None:
                    block_content = text[lang_start:next_close + 3]  # Include the closing ```
                    # Clean up the content
                    content = re.sub(r'^```{}\s*'.format(self.language), '', block_content)
                    content = re.sub(r'\s*```$', '', content)
                    code_blocks.append(content)
            
            if not code_blocks:
                return ""
            
            # Return the specific match based on index
            if self.match_index is not None:
                if self.match_index < 0:
                    self.match_index = len(code_blocks) + self.match_index
                if 0 <= self.match_index < len(code_blocks):
                    return code_blocks[self.match_index]
                return ""
            else:
                return code_blocks[0]  # Default to first match
            
        matches = [match_text(item) for item in data]
        return matches

class RegexMatchMultiOpen(RegexMatch):
    """
    Handle multi-choice and open-ended mixed tasks.
    """
    def __init__(self, additional_pattern: str = None, match_index: int = None, letter_pattern: str = None):
        super().__init__(additional_pattern, match_index)
        self.letter_pattern = letter_pattern if letter_pattern else r"\b([A-D])\b|\((?P<letter>[A-D])\)"
        self.indicators_of_keys = [
            'could be ', 'so ', 'is ',
            'thus ', 'therefore ', 'final ', 
            'answer ', 'result '
        ]

    def _match_letter(self, text: str):
        """Match the last uppercase letter in the string, either in parentheses or standalone.
    
        Args:
            text (str): Input text
            match_index (int): Index of the matched letter
        Returns:
            Optional[str]: Matched letter (uppercase) or None
        """
        try:
            pattern = re.compile(self.letter_pattern, re.IGNORECASE)
            matches = list(pattern.finditer(text))
        
            if not matches:
                return None
        
            selected_match = matches[self.match_index]
            if selected_match.group('letter'):
                return selected_match.group('letter').upper()  # From parentheses
            return selected_match.group(1).upper()  # Standalone letter
                
        except (IndexError, AttributeError, TypeError):
            return None

    def _check_is_number(self, string: str) -> bool:
        """Check if the given string is a number"""
        try:
            float(string.replace(',', ''))
            return True
        except ValueError:
            return False

    def _normalize_str(self, string):
        """
        Normalize the str to lower case and make them float numbers if possible.
        """
        # if number, numerize it.
        string = string.strip()

        is_number = self._check_is_number(string)

        if is_number:
            string = string.replace(',', '')
            string = float(string)
            # leave 2 decimal
            string = round(string, 2)
            return [string]
        else: # it's likely to be a string
            # lower it 
            string = string.lower()
            if len(string) == 1:
                return [" " + string, string + " "] # avoid trivial matches
            return [string]

    def _extract_numbers(self, string):
        """
        Exact all forms of numbers from a string with regex.
        """
        # Pattern for numbers with commas
        pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
        # Pattern for scientific notation
        pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
        # Pattern for simple numbers without commas
        pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

        # Extract numbers with commas
        numbers_with_commas = re.findall(pattern_commas, string)
        # Extract numbers in scientific notation
        numbers_scientific = re.findall(pattern_scientific, string)
        # Extract simple numbers without commas
        numbers_simple = re.findall(pattern_simple, string)

        # Combine all extracted numbers
        all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
        return all_numbers

    def _get_key_subresponses(self, response: str) -> List[str]:
        """Extract key responses from the model output"""
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        
        for index, resp in enumerate(sub_responses):
            indicators = self.indicators_of_keys + (['='] if index == len(sub_responses) - 1 else [])
            
            shortest_key_response = None
            for indicator in indicators:
                if indicator in resp:
                    current_response = resp.split(indicator)[-1].strip()
                    if not shortest_key_response or len(current_response) < len(shortest_key_response):
                        shortest_key_response = current_response
            
            if shortest_key_response and shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                key_responses.append(shortest_key_response)
        
        return key_responses if key_responses else [response]

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """Process responses with both letter matching and open-ended analysis"""
        try:
            def process_single_response(response: str) -> List:
                # Ensure response is string
                if not isinstance(response, str):
                    response = str(response)
                
                # Match letter
                letter_match = self._match_letter(response)
                
                # Process as open-ended response
                key_responses = self._get_key_subresponses(response)
                pred_list = key_responses.copy()
                
                for resp in key_responses:
                    pred_list.extend(self._extract_numbers(resp))
                
                tmp_pred_list = []
                for pred in pred_list:
                    tmp_pred_list.extend(self._normalize_str(str(pred)))
                
                final_pred_list = [letter_match]
                final_pred_list.extend(list(set(tmp_pred_list)))
                
                return final_pred_list

            if not data:
                return []
            
            return [process_single_response(response) for response in data]
        except Exception as e:
            print(f"Error processing responses: {e}")
            return None

    def __call__(self, data: Union[List, Iterable]) -> Union[List, None]:
        return self.apply(data)
    
class JudgeMC1(BaseTool):
    def __init__(self):
        super().__init__()
    
    def apply(self, scores, best_answer_index) -> bool:
        return True if scores['scores_true'][best_answer_index] > max(scores['scores_false']) else False
    
    def __call__(self, scores, best_answer_index) -> bool:
        return self.apply(scores, best_answer_index)
    
class JudgeMC2(BaseTool):
    def __init__(self):
        super().__init__()

    def apply(self, scores):
        scores_true = scores['scores_true']
        scores_false = scores['scores_false']

        probs_true = np.exp(scores_true)
        probs_false = np.exp(scores_false)
        probs_true = probs_true / (sum(probs_true) + sum(probs_false))
        
        return sum(probs_true)

    def __call__(self, scores):
        return self.apply(scores)