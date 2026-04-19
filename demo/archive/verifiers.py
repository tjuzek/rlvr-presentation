"""
Verifier functions for RLVR training.

These are the functions that assign rewards to model completions.
They check whether a model's response contains the correct answer.
"""

import json
import re


def verify_gsm8k(response: str, gold_answer: str, reward_value: float = 10.0) -> float:
    """
    GSM8K verifier: extract the last number from the response
    and check if it matches the gold answer.

    Args:
        response: The model's full text response.
        gold_answer: The expected numerical answer as a string.
        reward_value: Reward to give if correct (default: 10.0).

    Returns:
        reward_value if correct, 0.0 otherwise.
    """
    # Parse gold answer
    try:
        gold = float(gold_answer.replace(",", "").strip())
    except ValueError:
        # Try extracting number from "The answer is 6" format
        nums = re.findall(r"-?[\d,]+\.?\d*", gold_answer)
        if not nums:
            return 0.0
        gold = float(nums[-1].replace(",", ""))

    # Extract the last number from the response
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if not numbers:
        return 0.0

    try:
        predicted = float(numbers[-1].replace(",", ""))
    except ValueError:
        return 0.0

    return reward_value if abs(predicted - gold) < 1e-6 else 0.0


def verify_math(response: str, gold_answer: str, reward_value: float = 10.0) -> float:
    """
    MATH verifier: flexible answer extraction and matching.
    Tries three strategies (following Tulu 3's "flex" approach):
      1. Look for \\boxed{...} format
      2. Look for the last instance of <<answer>>
      3. Extract the last mathematical expression

    Args:
        response: The model's full text response.
        gold_answer: The expected answer string.
        reward_value: Reward to give if correct (default: 10.0).

    Returns:
        reward_value if correct, 0.0 otherwise.
    """
    gold_clean = gold_answer.strip()

    # Strategy 1: Check for \boxed{answer}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        if _normalize_math(boxed[-1]) == _normalize_math(gold_clean):
            return reward_value

    # Strategy 2: Check for explicit "answer is" patterns
    answer_patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so|hence)\s*,?\s*(.+?)(?:\.|$)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            if _normalize_math(matches[-1].strip()) == _normalize_math(gold_clean):
                return reward_value

    # Strategy 3: Extract the last mathematical expression/number
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if numbers:
        try:
            predicted = float(numbers[-1].replace(",", ""))
            gold_num = float(gold_clean.replace(",", ""))
            if abs(predicted - gold_num) < 1e-6:
                return reward_value
        except ValueError:
            pass

    # Final: direct string match
    if _normalize_math(response.strip().split("\n")[-1]) == _normalize_math(gold_clean):
        return reward_value

    return 0.0


def _normalize_math(s: str) -> str:
    """Normalize a math expression for comparison."""
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.lower()
    # Remove trailing period
    s = s.rstrip(".")
    # Remove dollar signs (LaTeX)
    s = s.replace("$", "")
    return s


def verify_constraint(response: str, constraint_type: str, **kwargs) -> float:
    """
    Instruction-following constraint verifier.
    Checks if the response satisfies a format constraint.

    Args:
        response: The model's full text response.
        constraint_type: The type of constraint to check.
        **kwargs: Additional parameters for the constraint.

    Returns:
        10.0 if constraint satisfied, 0.0 otherwise.
    """
    verifiers = {
        "All Lowercase": _check_lowercase,
        "All Uppercase": _check_uppercase,
        "Number of Paragraphs": _check_paragraph_count,
        "Number of Sentences": _check_sentence_count,
        "Forbidden Words": _check_forbidden_words,
        "JSON Format": _check_json_format,
        "Word Count": _check_word_count,
    }

    checker = verifiers.get(constraint_type)
    if checker is None:
        return 0.0

    return 10.0 if checker(response, **kwargs) else 0.0


def _check_lowercase(response: str, **kwargs) -> bool:
    return response == response.lower()


def _check_uppercase(response: str, **kwargs) -> bool:
    return response == response.upper()


def _check_paragraph_count(response: str, N: int = 1, **kwargs) -> bool:
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    return len(paragraphs) == N


def _check_sentence_count(response: str, N: int = 1, **kwargs) -> bool:
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) == N


def _check_forbidden_words(response: str, words: list[str] | None = None, **kwargs) -> bool:
    if words is None:
        return True
    response_lower = response.lower()
    return not any(w.lower() in response_lower for w in words)


def _check_json_format(response: str, **kwargs) -> bool:
    try:
        json.loads(response.strip())
        return True
    except json.JSONDecodeError:
        return False


def _check_word_count(response: str, min_words: int = 0, max_words: int = 999999, **kwargs) -> bool:
    count = len(response.split())
    return min_words <= count <= max_words


# ----- Example usage -----
if __name__ == "__main__":
    # GSM8K
    print("=== GSM8K Verifier ===")
    response1 = "Let me solve this step by step.\n15 trees + 6 more = 21 trees.\nSo 21 - 15 = 6 trees were planted."
    print(f"Response: {response1}")
    print(f"Reward: {verify_gsm8k(response1, '6')}")

    response2 = "The trees are beautiful in the grove. Many workers came to help."
    print(f"\nResponse: {response2}")
    print(f"Reward: {verify_gsm8k(response2, '6')}")

    # IF constraint
    print("\n=== Constraint Verifier ===")
    response3 = "this is all lowercase text about ipv6."
    print(f"Lowercase check: {verify_constraint(response3, 'All Lowercase')}")

    response4 = "This Has Capital Letters."
    print(f"Lowercase check: {verify_constraint(response4, 'All Lowercase')}")
