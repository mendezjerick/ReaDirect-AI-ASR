from __future__ import annotations

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.phonemes.phoneme_enricher import extract_vowel_phonemes, text_to_phonemes


def get_text_phonemes(text: str, cmudict_loader: CMUDictLoader | None = None) -> list[str]:
    return text_to_phonemes(text, cmudict_loader)


def phoneme_edit_distance(expected_phonemes: list[str], actual_phonemes: list[str]) -> int:
    previous = list(range(len(actual_phonemes) + 1))
    for i, expected in enumerate(expected_phonemes, start=1):
        current = [i]
        for j, actual in enumerate(actual_phonemes, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def phoneme_similarity(expected_phonemes: list[str], actual_phonemes: list[str]) -> float:
    if not expected_phonemes and not actual_phonemes:
        return 1.0
    if not expected_phonemes or not actual_phonemes:
        return 0.0
    distance = phoneme_edit_distance(expected_phonemes, actual_phonemes)
    return round(max(0.0, 1.0 - distance / max(len(expected_phonemes), len(actual_phonemes))), 6)


def detect_initial_phoneme_match(expected_phonemes: list[str], actual_phonemes: list[str]) -> bool | None:
    if not expected_phonemes or not actual_phonemes:
        return None
    return expected_phonemes[0] == actual_phonemes[0]


def detect_final_phoneme_match(expected_phonemes: list[str], actual_phonemes: list[str]) -> bool | None:
    if not expected_phonemes or not actual_phonemes:
        return None
    return expected_phonemes[-1] == actual_phonemes[-1]


def detect_vowel_phoneme_match(expected_phonemes: list[str], actual_phonemes: list[str]) -> bool | None:
    expected_vowels = extract_vowel_phonemes(expected_phonemes)
    actual_vowels = extract_vowel_phonemes(actual_phonemes)
    if not expected_vowels or not actual_vowels:
        return None
    return expected_vowels == actual_vowels


def compare_phonemes(expected_phonemes: list[str], actual_phonemes: list[str]) -> dict[str, object]:
    expected_vowels = extract_vowel_phonemes(expected_phonemes)
    actual_vowels = extract_vowel_phonemes(actual_phonemes)
    return {
        "expected_phonemes": expected_phonemes,
        "actual_phonemes": actual_phonemes,
        "phoneme_edit_distance": phoneme_edit_distance(expected_phonemes, actual_phonemes),
        "phoneme_similarity": phoneme_similarity(expected_phonemes, actual_phonemes),
        "initial_phoneme_match": detect_initial_phoneme_match(expected_phonemes, actual_phonemes),
        "final_phoneme_match": detect_final_phoneme_match(expected_phonemes, actual_phonemes),
        "vowel_phoneme_match": detect_vowel_phoneme_match(expected_phonemes, actual_phonemes),
        "expected_initial_phoneme": expected_phonemes[0] if expected_phonemes else None,
        "actual_initial_phoneme": actual_phonemes[0] if actual_phonemes else None,
        "expected_final_phoneme": expected_phonemes[-1] if expected_phonemes else None,
        "actual_final_phoneme": actual_phonemes[-1] if actual_phonemes else None,
        "expected_vowel_phonemes": expected_vowels,
        "actual_vowel_phonemes": actual_vowels,
    }

