import re
import string
import random
import numpy as np
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are no matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 1 or more matches, return the last one
    return matches[-1].group(1).strip()


# def reward_len(prompts, completions, context, ground_truths, **kwargs):
#     breakpoint()
#     return [-abs(20 - len(completion)) for completion in completions]


def compute_score_em(completions, ground_truths, format_bonus_val = 0.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truths: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=completions)
    if completions is None:
        return 0
    
    if "<format:" in completions:
        format_bonus = format_bonus_val
    else:
        format_bonus = 0.0

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truths):
            return 1.0 + format_bonus
        else:
            return format_bonus


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

def is_valid_sequence(text):
   # Find the position of "<|im_start|>assistant" with potential whitespace
   assistant_pattern = r"<\|im_start\|>assistant\s*"
   assistant_match = re.search(assistant_pattern, text)

   if not assistant_match:
       return False

   # Extract the content after the assistant marker
   start_pos = assistant_match.end()
   content = text[start_pos:]

   # Check for balanced <think> and <answer> tags
   tags_to_check = ["think", "answer"]
   for tag in tags_to_check:
       opening_count = len(re.findall(f"<{tag}>", content))
       closing_count = len(re.findall(f"</{tag}>", content))
       if opening_count != closing_count:
           return False

   # Check for balanced <format:...> and </format:...> tags
   format_open_tags = re.findall(r"<format:([^>]+)>", content)
   format_close_tags = re.findall(r"</format:([^>]+)>", content)

   if format_open_tags != format_close_tags:
       return False

   # Now check for proper sequence pattern and no extraneous content
   split_pattern = r"(</?(?:think|answer)>|<format:[^>]+>|</format:[^>]+>)"
   parts = re.split(split_pattern, content)

   state = "start"  # start -> in_think -> after_think -> in_format -> after_format -> ... -> in_think -> in_answer -> end

   for part in parts:
       if not part.strip():
           continue  # Skip whitespace
       # Recognize tag transitions
       if re.fullmatch(r"<think>", part) and state in ("start", "after_format"):
           state = "in_think"
       elif re.fullmatch(r"</think>", part) and state == "in_think":
           state = "after_think"
       elif re.fullmatch(r"<format:[^>]+>", part) and state == "after_think":
           state = "in_format"
       elif re.fullmatch(r"</format:[^>]+>", part) and state == "in_format":
           state = "after_format"
       elif re.fullmatch(r"<answer>", part) and state == "after_think":
           state = "in_answer"
       elif re.fullmatch(r"</answer>", part) and state == "in_answer":
           state = "end"
       elif part.startswith("<") and part.endswith(">"):
           return False  # invalid or misplaced tag
       else:
           # Non-tag content
           if state in ("in_think", "in_format", "in_answer"):
               continue
           elif state in ("start", "after_think", "after_format"):
               if part.strip():  # Disallow stray content between tags
                   return False
           else:
               return False
   return state == "end"

def compute_score_em_format(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.2, score=1.):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    answer = extract_solution(solution_str=solution_str)
    is_valid_format = is_valid_sequence(solution_str)

    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"Is valid format: {is_valid_format}")

    if answer is None:
        if is_valid_format:
            return format_score
        else:
            return 0.0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        elif is_valid_format:
            return score - format_score
        else:
            return 0.0
        

def calculate_metric_scores_f1(gold_answers, predicted_answers, aggregation_fn):
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    def compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        transformed_predicted = extract_solution(predicted)
        if transformed_predicted is None:
            predicted_tokens = normalize_answer(predicted).split()
        else:
            predicted_tokens = normalize_answer(transformed_predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(predicted_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    example_eval_results = []
    total_f1 = 0.0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
        aggregated_f1 = aggregation_fn(f1_scores)
        example_eval_results.append({"F1": aggregated_f1})
        total_f1 += aggregated_f1

    avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"F1": avg_f1}

    return pooled_eval_results, example_eval_results

def calculate_f1(gold_answers, predicted_answers):
    overall_qa_f1_result, example_qa_f1_results = calculate_metric_scores_f1(
        gold_answers=gold_answers, predicted_answers=predicted_answers,
        aggregation_fn=np.max)
    return overall_qa_f1_result["F1"]