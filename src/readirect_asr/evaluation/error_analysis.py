from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from readirect_asr.evaluation.asr_metrics import compute_cer, exact_match
from readirect_asr.text.normalization import normalize_for_wer


def categorize_asr_error(reference: str, hypothesis: str) -> str:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return "blank_reference"
    if not hyp:
        return "blank_hypothesis"
    if ref == hyp:
        return "exact"

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if len(hyp_tokens) > len(ref_tokens) and _is_subsequence(ref_tokens, hyp_tokens):
        return "insertion"
    if len(hyp_tokens) < len(ref_tokens) and _is_subsequence(hyp_tokens, ref_tokens):
        return "deletion"
    if len(ref_tokens) == len(hyp_tokens):
        differing = sum(left != right for left, right in zip(ref_tokens, hyp_tokens))
        if differing == 1:
            return "substitution"
    ratio = SequenceMatcher(a=ref, b=hyp).ratio()
    return "partial_match" if ratio >= 0.45 else "unrelated"


def summarize_common_substitutions(
    df: pd.DataFrame,
    reference_col: str,
    hypothesis_col: str,
    limit: int = 25,
) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        ref_tokens = normalize_for_wer(str(row.get(reference_col, ""))).split()
        hyp_tokens = normalize_for_wer(str(row.get(hypothesis_col, ""))).split()
        if len(ref_tokens) != len(hyp_tokens):
            continue
        for ref_token, hyp_token in zip(ref_tokens, hyp_tokens):
            if ref_token != hyp_token:
                counter[(ref_token, hyp_token)] += 1
    return [
        {"reference": pair[0], "hypothesis": pair[1], "count": count}
        for pair, count in counter.most_common(limit)
    ]


def summarize_short_word_accuracy(
    df: pd.DataFrame,
    reference_col: str = "manual_transcript",
    hypothesis_col: str = "normalized_transcript",
) -> dict[str, Any]:
    rows = []
    confusions: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        ref = normalize_for_wer(str(row.get(reference_col, "")))
        hyp = normalize_for_wer(str(row.get(hypothesis_col, "")))
        if len(ref.split()) != 1 or len(ref) > 5 or len(ref) < 1:
            continue
        rows.append((ref, hyp))
        if ref != hyp:
            confusions[(ref, hyp)] += 1
    exact_count = sum(1 for ref, hyp in rows if ref == hyp)
    return {
        "short_word_rows": len(rows),
        "exact_match_rate": round(exact_count / len(rows), 6) if rows else 0.0,
        "average_cer": round(sum(compute_cer(ref, hyp) for ref, hyp in rows) / len(rows), 6) if rows else 0.0,
        "common_confusions": [
            {"reference": pair[0], "hypothesis": pair[1], "count": count}
            for pair, count in confusions.most_common(25)
        ],
    }


def summarize_by_text_length(
    df: pd.DataFrame,
    reference_col: str = "manual_transcript",
    hypothesis_col: str = "normalized_transcript",
) -> dict[str, Any]:
    buckets = {"1_word": [], "2_5_words": [], "6_plus_words": []}
    for _, row in df.iterrows():
        ref = str(row.get(reference_col, ""))
        hyp = str(row.get(hypothesis_col, ""))
        word_count = len(normalize_for_wer(ref).split())
        bucket = "1_word" if word_count == 1 else "2_5_words" if word_count <= 5 else "6_plus_words"
        buckets[bucket].append(exact_match(ref, hyp))
    return {
        bucket: {
            "rows": len(values),
            "exact_match_rate": round(sum(values) / len(values), 6) if values else 0.0,
        }
        for bucket, values in buckets.items()
    }


def summarize_by_speaker_type(df: pd.DataFrame) -> dict[str, int]:
    return _value_counts(df, "speaker_type")


def summarize_by_age_group(df: pd.DataFrame) -> dict[str, int]:
    return _value_counts(df, "age_group")


def summarize_by_score_bucket(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for column in ("sentence_score", "word_score", "phoneme_score"):
        if column not in df.columns:
            continue
        counts = {"missing": 0, "low": 0, "medium": 0, "high": 0}
        for value in pd.to_numeric(df[column], errors="coerce"):
            if pd.isna(value):
                counts["missing"] += 1
            elif value < 5:
                counts["low"] += 1
            elif value < 8:
                counts["medium"] += 1
            else:
                counts["high"] += 1
        summaries[column] = counts
    return summaries


def _is_subsequence(shorter: list[str], longer: list[str]) -> bool:
    iterator = iter(longer)
    return all(token in iterator for token in shorter)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return df[column].fillna("").astype(str).value_counts().to_dict()

