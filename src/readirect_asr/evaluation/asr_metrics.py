from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from readirect_asr.text.normalization import normalize_for_wer


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        from jiwer import wer

        return float(wer(ref, hyp))
    except Exception:
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        distance = _levenshtein(ref_tokens, hyp_tokens)
        return distance / max(1, len(ref_tokens))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        from jiwer import cer

        return float(cer(ref, hyp))
    except Exception:
        distance = _levenshtein(list(ref), list(hyp))
        return distance / max(1, len(ref))


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalize_for_wer(reference) == normalize_for_wer(hypothesis)


def token_accuracy(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_for_wer(reference).split()
    hyp_tokens = normalize_for_wer(hypothesis).split()
    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0
    matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    correct = sum(block.size for block in matcher.get_matching_blocks())
    return correct / len(ref_tokens)


def evaluate_asr_dataframe(
    df: pd.DataFrame,
    reference_col: str,
    hypothesis_col: str,
) -> dict[str, Any]:
    rows = []
    blank_reference_count = 0
    blank_hypothesis_count = 0
    skipped_rows = 0

    for _, row in df.iterrows():
        reference = str(row.get(reference_col, "") if pd.notna(row.get(reference_col, "")) else "")
        hypothesis = str(row.get(hypothesis_col, "") if pd.notna(row.get(hypothesis_col, "")) else "")
        if not normalize_for_wer(reference):
            blank_reference_count += 1
            skipped_rows += 1
            continue
        if not normalize_for_wer(hypothesis):
            blank_hypothesis_count += 1
        rows.append(
            {
                "wer": compute_wer(reference, hypothesis),
                "cer": compute_cer(reference, hypothesis),
                "exact": exact_match(reference, hypothesis),
                "token_accuracy": token_accuracy(reference, hypothesis),
            }
        )

    evaluated_rows = len(rows)
    return {
        "wer": _mean([row["wer"] for row in rows]),
        "cer": _mean([row["cer"] for row in rows]),
        "exact_match_rate": _mean([1.0 if row["exact"] else 0.0 for row in rows]),
        "average_token_accuracy": _mean([row["token_accuracy"] for row in rows]),
        "total_rows": len(df),
        "evaluated_rows": evaluated_rows,
        "skipped_rows": skipped_rows,
        "blank_reference_count": blank_reference_count,
        "blank_hypothesis_count": blank_hypothesis_count,
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i]
        for j, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]

