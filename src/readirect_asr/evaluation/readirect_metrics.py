from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from readirect_asr.evaluation.asr_metrics import compute_cer, exact_match
from readirect_asr.text.normalization import normalize_for_wer


SHORT_WORD_EXAMPLES = {"cat", "dog", "sun", "pen", "map", "cup", "hat", "pig", "run", "box"}


def is_short_word(text: str) -> bool:
    normalized = normalize_for_wer(text)
    tokens = normalized.split()
    return len(tokens) == 1 and tokens[0].isalpha() and 1 <= len(tokens[0]) <= 5


def evaluate_short_words(df: pd.DataFrame, reference_col: str, hypothesis_col: str) -> dict[str, Any]:
    rows: list[tuple[str, str]] = []
    confusions: Counter[tuple[str, str]] = Counter()
    blanks = 0
    near = 0
    for _, row in df.iterrows():
        ref = normalize_for_wer(str(row.get(reference_col, "") if pd.notna(row.get(reference_col, "")) else ""))
        hyp = normalize_for_wer(str(row.get(hypothesis_col, "") if pd.notna(row.get(hypothesis_col, "")) else ""))
        if not is_short_word(ref):
            continue
        rows.append((ref, hyp))
        if not hyp:
            blanks += 1
        if ref != hyp:
            confusions[(ref, hyp)] += 1
        if ref != hyp and compute_cer(ref, hyp) <= 0.4:
            near += 1
    exact_count = sum(1 for ref, hyp in rows if exact_match(ref, hyp))
    total = len(rows)
    return {
        "total_short_word_rows": total,
        "exact_match_rate": round(exact_count / total, 6) if total else 0.0,
        "average_cer": round(sum(compute_cer(ref, hyp) for ref, hyp in rows) / total, 6) if total else 0.0,
        "common_confusions": [
            {"reference": pair[0], "hypothesis": pair[1], "count": count}
            for pair, count in confusions.most_common(25)
        ],
        "blank_rate": round(blanks / total, 6) if total else 0.0,
        "near_match_rate": round(near / total, 6) if total else 0.0,
    }


def evaluate_cvc_words(df: pd.DataFrame, reference_col: str, hypothesis_col: str, cmudict_loader: Any | None = None) -> dict[str, Any]:
    selected = []
    for _, row in df.iterrows():
        ref = normalize_for_wer(str(row.get(reference_col, "")))
        if _is_cvc(ref, cmudict_loader):
            selected.append(row)
    subset = pd.DataFrame(selected)
    metrics = evaluate_short_words(subset, reference_col, hypothesis_col) if not subset.empty else evaluate_short_words(pd.DataFrame(columns=[reference_col, hypothesis_col]), reference_col, hypothesis_col)
    metrics["total_cvc_rows"] = metrics.pop("total_short_word_rows")
    return metrics


def evaluate_readirect_keywords(df: pd.DataFrame, reference_col: str, hypothesis_col: str, keyword_list: list[str] | None = None) -> dict[str, Any]:
    keywords = {normalize_for_wer(word) for word in (keyword_list or sorted(SHORT_WORD_EXAMPLES))}
    selected = []
    for _, row in df.iterrows():
        ref = normalize_for_wer(str(row.get(reference_col, "")))
        if ref in keywords:
            selected.append(row)
    subset = pd.DataFrame(selected)
    metrics = evaluate_short_words(subset, reference_col, hypothesis_col) if not subset.empty else evaluate_short_words(pd.DataFrame(columns=[reference_col, hypothesis_col]), reference_col, hypothesis_col)
    metrics["keywords_evaluated"] = sorted(keywords)
    return metrics


def _is_cvc(word: str, cmudict_loader: Any | None = None) -> bool:
    normalized = normalize_for_wer(word)
    if not is_short_word(normalized) or len(normalized) != 3:
        return False
    if cmudict_loader:
        pronunciations = cmudict_loader.get_pronunciations(normalized)
        if pronunciations:
            phones = pronunciations[0]
            return len(phones) == 3 and _phone_kind(phones[0]) == "C" and _phone_kind(phones[1]) == "V" and _phone_kind(phones[2]) == "C"
    return normalized[0] not in "aeiou" and normalized[1] in "aeiou" and normalized[2] not in "aeiou"


def _phone_kind(phone: str) -> str:
    base = "".join(ch for ch in str(phone) if not ch.isdigit())
    return "V" if base in {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"} else "C"
