from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any


NOISE_PATTERN = re.compile(r"\[[^\]]+\]|\([^\)]+\)|<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
SMART_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
}
SPECIAL_TOKENS = {
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "|",
}


def load_tokenizer_vocab(model_path: str | Path) -> set[str] | None:
    """Load a local tokenizer vocabulary without downloading anything."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        return set(tokenizer.get_vocab().keys())
    except Exception:
        return None


def infer_case(vocab: set[str] | None) -> str:
    if not vocab:
        return "upper"
    letters = [token for token in vocab if len(token) == 1 and token.isalpha()]
    upper_count = sum(1 for token in letters if token.isupper())
    lower_count = sum(1 for token in letters if token.islower())
    return "lower" if lower_count > upper_count else "upper"


def tokenizer_character_set(vocab: set[str] | None) -> set[str] | None:
    if not vocab:
        return None
    chars = {token for token in vocab if len(token) == 1 and token not in SPECIAL_TOKENS}
    if "|" in vocab:
        chars.add(" ")
    if " " in vocab:
        chars.add(" ")
    return chars or None


def normalize_asr_text(
    text: Any,
    vocab: set[str] | None = None,
    *,
    remove_noise_markers: bool = True,
    english_only: bool = True,
) -> str:
    """Normalize transcript text for Wav2Vec2 CTC labels.

    The function preserves single-letter targets as letters. It does not expand
    letters such as "Q" into spoken words such as "cue".
    """
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    for source, replacement in SMART_QUOTES.items():
        value = value.replace(source, replacement)
    if remove_noise_markers:
        value = NOISE_PATTERN.sub(" ", value)

    case = infer_case(vocab)
    value = value.lower() if case == "lower" else value.upper()

    supported_chars = tokenizer_character_set(vocab)
    keep_apostrophe = supported_chars is None or "'" in supported_chars
    normalized_chars: list[str] = []
    for char in value:
        if char.isspace() or char in {"|", "\t", "\n", "\r"}:
            normalized_chars.append(" ")
            continue
        if english_only and not (char.isascii() and (char.isalpha() or char == "'")):
            normalized_chars.append(" ")
            continue
        if char == "'" and not keep_apostrophe:
            normalized_chars.append(" ")
            continue
        if supported_chars is not None and char not in supported_chars:
            normalized_chars.append(" ")
            continue
        normalized_chars.append(char)

    return WHITESPACE_PATTERN.sub(" ", "".join(normalized_chars)).strip()


def normalize_with_model_vocab(text: Any, model_path: str | Path) -> str:
    return normalize_asr_text(text, load_tokenizer_vocab(model_path))

