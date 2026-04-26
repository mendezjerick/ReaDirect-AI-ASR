from __future__ import annotations

import re
from typing import Any


DIGRAPHS = {"CH", "SH", "TH", "WH", "PH", "NG"}


def compute_difficulty(row: dict[str, Any], enrichment: dict[str, Any]) -> dict[str, object]:
    text = str(row.get("expected_text", "") or row.get("prompt_text", "") or "")
    text_type = str(enrichment.get("item_text_type", "unknown"))
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)
    phoneme_count = int(enrichment.get("phoneme_count") or 0)
    syllables = int(enrichment.get("syllable_estimate") or 0)
    missing = bool(enrichment.get("cmudict_missing_words"))
    pattern = str(enrichment.get("phoneme_pattern", ""))

    factors: dict[str, float] = {}
    if text_type in {"single_letter", "single_word"}:
        factors["word_length"] = min(len(text) / 10.0, 1.0)
        factors["phoneme_count"] = min(phoneme_count / 8.0, 1.0)
        factors["syllables"] = min(max(syllables, 1) / 4.0, 1.0)
        factors["blend_or_digraph"] = 0.2 if _has_blend_or_digraph(pattern, enrichment) else 0.0
        factors["missing_cmudict"] = 0.2 if missing else 0.0
    else:
        avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0.0
        punctuation = 1.0 if re.search(r"[,:;!?]", text) else 0.0
        factors["word_count"] = min(word_count / 15.0, 1.0)
        factors["average_word_length"] = min(avg_word_length / 8.0, 1.0)
        factors["punctuation"] = punctuation * 0.15
        factors["missing_cmudict"] = 0.15 if missing else 0.0

    if str(row.get("difficulty", "")).lower() in {"very_easy", "easy", "medium", "hard", "very_hard"}:
        factors["source_difficulty"] = {
            "very_easy": 0.1,
            "easy": 0.25,
            "medium": 0.5,
            "hard": 0.75,
            "very_hard": 0.9,
        }[str(row.get("difficulty")).lower()]

    score = min(1.0, sum(factors.values()) / max(1, len(factors)))
    return {
        "difficulty_score": round(score, 3),
        "difficulty_level": _level(score),
        "difficulty_factors": factors,
    }


def _has_blend_or_digraph(pattern: str, enrichment: dict[str, Any]) -> bool:
    phonemes = str(enrichment.get("expected_phonemes", "")).split()
    if any(phone in DIGRAPHS for phone in phonemes):
        return True
    return "CC" in pattern


def _level(score: float) -> str:
    if score <= 0.20:
        return "very_easy"
    if score <= 0.40:
        return "easy"
    if score <= 0.60:
        return "medium"
    if score <= 0.80:
        return "hard"
    return "very_hard"

