from __future__ import annotations

import re

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.phonemes.phoneme_schema import DEFAULT_VOWEL_PHONEMES, PhonemeSchema


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def word_to_phonemes(word: str, loader: CMUDictLoader | None = None) -> list[str]:
    active_loader = loader or CMUDictLoader().load()
    pronunciation = active_loader.get_primary_pronunciation(word)
    return pronunciation or []


def text_to_phonemes(text: str, loader: CMUDictLoader | None = None) -> list[str]:
    active_loader = loader or CMUDictLoader().load()
    phonemes: list[str] = []
    for word in WORD_RE.findall(str(text or "")):
        word_phones = active_loader.get_primary_pronunciation(word)
        if not word_phones:
            return []
        phonemes.extend(word_phones)
    return phonemes


def extract_initial_phoneme(phonemes: list[str]) -> str | None:
    return phonemes[0] if phonemes else None


def extract_final_phoneme(phonemes: list[str]) -> str | None:
    return phonemes[-1] if phonemes else None


def extract_vowel_phonemes(
    phonemes: list[str],
    schema: PhonemeSchema | None = None,
) -> list[str]:
    vowels = schema.vowel_phonemes if schema else DEFAULT_VOWEL_PHONEMES
    return [phoneme for phoneme in phonemes if phoneme.upper() in vowels]


def classify_phoneme_pattern(
    phonemes: list[str],
    schema: PhonemeSchema | None = None,
) -> str | None:
    if not phonemes:
        return None
    vowels = schema.vowel_phonemes if schema else DEFAULT_VOWEL_PHONEMES
    return "".join("V" if phoneme.upper() in vowels else "C" for phoneme in phonemes)


def infer_error_focus(expected_text: str, phonemes: list[str]) -> str | None:
    if not phonemes:
        return None
    words = WORD_RE.findall(str(expected_text or ""))
    if len(words) > 1:
        return "word_sequence"

    pattern = classify_phoneme_pattern(phonemes)
    if pattern == "CVC":
        return "vowel_sound"
    if pattern and pattern.endswith("C"):
        return "final_consonant"
    if pattern and pattern.startswith("C"):
        return "initial_consonant"
    if "V" in (pattern or ""):
        return "vowel_sound"
    return None


def enrich_text_phonemes(
    text: str,
    loader: CMUDictLoader | None = None,
    schema: PhonemeSchema | None = None,
) -> dict[str, object]:
    active_loader = loader or CMUDictLoader().load()
    active_schema = schema or PhonemeSchema(active_loader.phone_categories, active_loader.symbols)
    phonemes = text_to_phonemes(text, active_loader)
    return {
        "expected_phonemes": " ".join(phonemes),
        "initial_phoneme": extract_initial_phoneme(phonemes),
        "vowel_phonemes": " ".join(extract_vowel_phonemes(phonemes, active_schema)),
        "final_phoneme": extract_final_phoneme(phonemes),
        "phoneme_pattern": classify_phoneme_pattern(phonemes, active_schema),
        "error_focus": infer_error_focus(text, phonemes),
    }

