from __future__ import annotations

import re


HESITATIONS = {"um", "uh", "ah", "hmm"}


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9'\s]", " ", str(text or ""))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def remove_common_hesitations(text: str, hesitations: set[str] | None = None) -> str:
    active_hesitations = hesitations or HESITATIONS
    tokens = [
        token
        for token in normalize_whitespace(text).split(" ")
        if token.lower() not in active_hesitations
    ]
    return " ".join(tokens)


def normalize_transcript(text: str) -> str:
    no_punctuation = remove_punctuation(str(text or "").lower())
    return normalize_whitespace(no_punctuation)


def normalize_for_wer(text: str) -> str:
    normalized = normalize_transcript(text)
    return remove_common_hesitations(normalized)

