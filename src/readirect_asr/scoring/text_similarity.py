from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    lowered = str(text or "").lower()
    no_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def levenshtein_distance(expected: str, actual: str) -> int:
    left = normalize_text(expected)
    right = normalize_text(actual)
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def similarity_percentage(expected: str, actual: str) -> float:
    left = normalize_text(expected)
    right = normalize_text(actual)
    if not left and not right:
        return 100.0
    if not left or not right:
        return 0.0

    max_len = max(len(left), len(right))
    distance = levenshtein_distance(left, right)
    return round(max(0.0, (1 - distance / max_len) * 100), 2)


def classify_similarity(expected: str, actual: str) -> str:
    left = normalize_text(expected)
    right = normalize_text(actual)
    if not right:
        return "blank"
    if left == right:
        return "exact"

    score = similarity_percentage(left, right)
    if score >= 90:
        return "very_close"
    if score >= 75:
        return "close"
    if score >= 50:
        return "somewhat_close"
    return "far"

