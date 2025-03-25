"""
Evaluation tools package (including calculation of various metrics and extraction of specified patterns)
"""
from abc import ABC, abstractmethod
from eval_anything.evaluate_tools.base_tools import BaseTool
from typing import Union, List, Iterable, Optional
from latex2sympy2 import latex2sympy
from math import *
import numpy as np
import re

T2T_EXTRACTOR_MAP = {
    "regex_match_number": "RegexMatchNumber",
    "regex_match_letter": "RegexMatchLetter",
    "regex_match_multi_letter": "RegexMatchMultiLetter",
    "regex_match_code": "RegexMatchCode",
    "regex_match_math": "RegexMatchMath",
    "regex_match_multi_open": "RegexMatchMultiOpen",
    "regex_match_latex_math" : "RegexMatchLatexMath"
}

T2T_JUDGER_MAP = {
    "judge_equal": "JudgeEqual",
    "judge_equal_list": "JudgeEqualList",
    "judge_latex_equal":"JudgeLatexEqual",
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
    
    
# refer: https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py
class JudgeLatexEqual(BaseTool):
    def __init__(self):
        super().__init__()


    def apply(self, data_1, data_2) -> bool:
        def eval_tuple(s):
            """
            Evaluates the mathematical expressions within tuples or lists represented as strings.
            
            Args:
                s (str): The string representation of a tuple or list.
                         E.g., "(a,b,c,...)" or "[a,b,c,...]"
            
            Returns:
                str: A string representation of the tuple or list with evaluated expressions.
                     Returns the original string if it doesn't match the expected format or if an error occurs.
            
            Example:
                eval_tuple("(2*3, 5+2)") -> "(6,7)"
            
            Note:
                This function relies on the latex2sympy function which is assumed to be defined elsewhere in the code.
            """
            # Split the string by commas to get individual elements
            sl = s[1:-1].split(',')
            
            try:
                # Check if string is a tuple representation and has more than one element
                if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
                    # Evaluate each element using latex2sympy and round the result to 2 decimal places
                    # Skip evaluation if element is 'infty', 'a', or '-a'
                    s = ','.join([str(round(eval(str(latex2sympy(sub))),2)) 
                                  if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
                    return f"({s})"
                
                # Check if string is a list representation and has more than one element
                elif s[0] == '[' and s[-1] == ']' and len(sl) > 1:
                    # Same evaluation process as for tuples
                    s = ','.join([str(round(eval(str(latex2sympy(sub))),2)) 
                                  if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
                    return f"[{s}]"
            
            except Exception:  # Catch any exceptions and return the original string
                return s
            
            # Return original string if it doesn't match tuple or list format
            return s

        def is_equal(asw: str, gt_asw: str) -> bool:
            """
            Judge if `asw` is equivalent to `gt_asw`.

            This function checks if the given answers are equivalent, considering
            various scenarios such as tuples, lists separated by commas, and
            mathematical equivalence in LaTeX format.

            Args:
                asw (str): The answer string to be checked.
                gt_asw (str): The ground truth answer string to be matched against.

            Returns:
                bool: True if the answers are equivalent, otherwise False.

            """

            # return gt_asw == asw

            # Check for empty strings after removing spaces and return False if any of them is empty.
            asw = asw.lower()
            gt_asw = gt_asw.lower()
            
            if asw.replace(' ', '') == '' or gt_asw.replace(' ', '') == '':
                return False

            if gt_asw.strip() == asw.strip():
                return True
           
            # Convert the string to a tuple format.
            asw = eval_tuple(asw)
            gt_asw = eval_tuple(gt_asw)

            # Check for simple tuple containment. Return True if one tuple is contained in the other.
            if gt_asw == asw:
                return True

            

            try:
                # Convert LaTeX format to a sympy expression and evaluate both expressions.
                # If the evaluated results are close enough (up to 2 decimal places), return True.
                if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
                    return True

                else:
                    return False
            except:
                # If any error occurs during comparison, return False.
                return False
        
        return is_equal(data_1, data_2)
    
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
    
# refer: https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py
class RegexMatchLatexMath(RegexMatch):
    """
    Handle multi-choice and open-ended mixed tasks.
    """
    def __init__(self, additional_pattern: str = None, match_index: int = None, letter_pattern: str = None):
        super().__init__(additional_pattern, match_index)



    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """Process responses with both letter matching and open-ended analysis"""
        def _fix_fracs(string):
            # Split the string based on occurrences of '\frac'.
            substrs = string.split("\\frac")
            new_str = substrs[0]

            # Check if there are any occurrences of '\frac' in the string.
            if len(substrs) > 1:
                # Exclude the part of the string before the first '\frac'.
                substrs = substrs[1:]

                for substr in substrs:
                    new_str += "\\frac"
                    # If the current substring already starts with a brace, 
                    # it's likely formatted correctly.
                    if len(substr) > 0 and substr[0] == "{":
                        new_str += substr
                    else:
                        # Ensure that the substring has at least 2 characters 
                        # for numerator and denominator.
                        try:
                            assert len(substr) >= 2
                        except:
                            return string

                        a = substr[0]  # Potential numerator.
                        b = substr[1]  # Potential denominator.

                        # Check if the denominator (b) is already braced.
                        if b != "{":
                            if len(substr) > 2:
                                post_substr = substr[2:]
                                new_str += "{" + a + "}{" + b + "}" + post_substr
                            else:
                                new_str += "{" + a + "}{" + b + "}"
                        else:
                            if len(substr) > 2:
                                post_substr = substr[2:]
                                new_str += "{" + a + "}" + b + post_substr
                            else:
                                new_str += "{" + a + "}" + b

            # Update the string to the newly formatted version.
            string = new_str
            return string

        def _fix_a_slash_b(string):
            # Check if the string contains exactly one slash, which may indicate it's a fraction.
            if len(string.split("/")) != 2:
                return string

            # Split the string by slash to extract potential numerator and denominator.
            a, b = string.split("/")

            try:
                # Try to convert the parts to integers.
                a = int(a)
                b = int(b)

                # Check if the string is in the expected format after conversion.
                assert string == "{}/{}".format(a, b)

                # Convert the fraction to LaTeX representation.
                new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
                return new_string

            # Handle exceptions for non-integer fractions or other unexpected formats.
            except:
                return string

        def _remove_right_units(string):
            # Split the string using "\\text{ " as the delimiter.
            splits = string.split("\\text{ ")
            
            # Return the part of the string before the last occurrence of "\\text{ ".
            return splits[0]

        def _fix_sqrt(string):
            # Check if "\sqrt" is not in the string. If not, return the string as is.
            if "\\sqrt" not in string:
                return string

            # Split the string based on the "\sqrt" substring.
            splits = string.split("\\sqrt")
            
            # The initial portion of the string before the first occurrence of "\sqrt".
            new_string = splits[0]

            # Loop through each split portion (after the initial one).
            for split in splits[1:]:
                # If the split portion is non-empty and the first character isn't a '{',
                # then it means the argument of the sqrt is not enclosed in braces.
                if len(split) > 0 and split[0] != "{":
                    a = split[0]
                    # Add braces around the first character and append the rest of the split portion.
                    new_substr = "\\sqrt{" + a + "}" + split[1:]
                else:
                    # If the split portion starts with a '{', then it's already correct.
                    new_substr = "\\sqrt" + split
                # Add the new substring to our result string.
                new_string += new_substr

            return new_string

        def _strip_string(string):
            # Remove linebreaks
            string = string.replace("\n", "")

            # Remove inverse spaces
            string = string.replace("\\!", "")

            # Replace double backslashes with a single backslash
            string = string.replace("\\\\", "\\")

            # Replace tfrac and dfrac with frac
            string = string.replace("tfrac", "frac")
            string = string.replace("dfrac", "frac")

            # Remove \left and \right LaTeX commands
            string = string.replace("\\left", "")
            string = string.replace("\\right", "")

            # Remove degree notation
            string = string.replace("^{\\circ}", "")
            string = string.replace("^\\circ", "")

            # Remove dollar signs (potentially used for inline math in LaTeX)
            string = string.replace("\\$", "")
            string = string.replace("$", "")

            # Remove units (assumed to be on the right). Note: The function _remove_right_units is not provided.
            string = _remove_right_units(string)

            # Remove percentage notations
            string = string.replace("\\%", "")
            string = string.replace("\%", "")

            # Handle floating numbers starting with "."
            string = string.replace(" .", " 0.")
            string = string.replace("{.", "{0.")
            if len(string) == 0:
                return string
            if string[0] == ".":
                string = "0" + string

            # If there are equalities or approximations, only consider the value after them
            if len(string.split("=")) == 2:
                string = string.split("=")[-1]
            if len(string.split("\\approx")) == 2:
                string = string.split("\\approx")[-1]

            # Fix sqrt values not wrapped in curly braces. Note: The function _fix_sqrt is not provided.
            if 'sqrt' in string:
                string = _fix_sqrt(string)

            # Remove all spaces
            string = string.replace(" ", "")

            # Transform certain fraction notations to the desired format. Note: The function _fix_fracs is not provided.
            if 'sqrt' in string:
                string = _fix_fracs(string)

            # Convert 0.5 to its fraction representation
            if string == "0.5":
                string = "\\frac{1}{2}"

            # Fix fractions represented with a slash. Note: The function _fix_a_slash_b is not provided.
            string = _fix_a_slash_b(string)

            return string

        def find_math_answer(s: str) -> str:
            s = s.lower()
            if '{}' in s:
                s = s.replace('{}', '')

            try:
                pattern = re.compile('oxed{(.*)}', flags=re.S)
                ans = pattern.findall(s)[-1]
            except:     
                ans = s  # If the pattern is not found, consider the entire string as the answer.

            # If there's a closing bracket without an opening bracket before it, consider everything before it.
            if ans.find('}') != -1 and (ans.find('{') == -1 or  ans.find('}') < ans.find('{')):
                ans = ans.split('}')[0]

            # Extract the value after the equals sign or approx symbol.
            ans = ans.split('=')[-1]
            ans = ans.split('\\approx')[-1]

            # Clean the string from various LaTeX formatting.
            ans = ans.replace(" ", "").replace("\\,", "").replace('∞', '\\infty')
            ans = ans.replace("+\infty", "\infty").replace("\\\\", "\\").replace("\n", "")
            ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
            ans = ans.replace("\\left", "").replace('\\right', '').replace("^{\\circ}", "")
            ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
            ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")

            return _strip_string(ans)         
        

        
        try:
            def process_single_response(response: str) -> List:
                response = '\\boxed{' + response.split('oxed{')[-1]
                response = find_math_answer(response).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
                return response
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