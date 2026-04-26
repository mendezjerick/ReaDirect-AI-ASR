from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from readirect_asr.text.normalization import normalize_transcript


def normalize_answer(text: str) -> str:
    return normalize_transcript(text)


def parse_accepted_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    separator = "|" if "|" in text else "," if "," in text else None
    if separator:
        return [item.strip() for item in text.split(separator) if item.strip()]
    return [text]


def is_exact_match(expected: str, actual: str) -> bool:
    return normalize_answer(expected) == normalize_answer(actual)


def is_accepted_answer(accepted_answers: list[str], actual: str) -> bool:
    actual_norm = normalize_answer(actual)
    return bool(actual_norm) and any(normalize_answer(answer) == actual_norm for answer in accepted_answers)


def compute_character_similarity(expected: str, actual: str) -> float:
    left = normalize_answer(expected)
    right = normalize_answer(actual)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return round(float(fuzz.ratio(left, right)) / 100.0, 6)
    except Exception:
        return round(SequenceMatcher(a=left, b=right).ratio(), 6)


def compute_token_similarity(expected: str, actual: str) -> float:
    left = normalize_answer(expected)
    right = normalize_answer(actual)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(a=left.split(), b=right.split()).ratio(), 6)


def similarity_label(character_similarity: float, normalized_actual: str, is_exact: bool, normalized_expected: str = "") -> str:
    if not normalized_actual:
        return "blank"
    if is_exact or character_similarity == 1.0:
        return "exact"
    if max(len(normalized_expected), len(normalized_actual)) <= 3 and character_similarity >= 0.66:
        return "very_close"
    if character_similarity >= 0.80:
        return "very_close"
    if character_similarity >= 0.60:
        return "close"
    if character_similarity >= 0.40:
        return "somewhat_close"
    return "far"


def match_answer(expected: str, actual: str, accepted_answers: Any = None) -> dict[str, object]:
    accepted = parse_accepted_answers(accepted_answers)
    normalized_expected = normalize_answer(expected)
    normalized_actual = normalize_answer(actual)
    exact = is_exact_match(expected, actual)
    accepted_match = is_accepted_answer(accepted, actual)
    char_similarity = compute_character_similarity(expected, actual)
    token_similarity = compute_token_similarity(expected, actual)
    return {
        "expected_text": expected,
        "actual_text": actual,
        "normalized_expected": normalized_expected,
        "normalized_actual": normalized_actual,
        "is_exact": exact,
        "is_accepted": accepted_match,
        "is_correct": exact or accepted_match,
        "character_similarity": char_similarity,
        "token_similarity": token_similarity,
        "similarity_label": similarity_label(char_similarity, normalized_actual, exact, normalized_expected),
    }
