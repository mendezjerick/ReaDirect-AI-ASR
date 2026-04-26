from __future__ import annotations

from dataclasses import dataclass


DEFAULT_VOWEL_PHONEMES = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


@dataclass(frozen=True)
class PhonemeSchema:
    phone_categories: dict[str, str]
    valid_symbols: set[str]

    @property
    def vowel_phonemes(self) -> set[str]:
        vowels = {
            phone
            for phone, category in self.phone_categories.items()
            if category == "vowel"
        }
        return vowels or DEFAULT_VOWEL_PHONEMES

    @property
    def consonant_phonemes(self) -> set[str]:
        return {
            phone
            for phone in self.phone_categories
            if phone not in self.vowel_phonemes
        }

    def is_vowel(self, phoneme: str) -> bool:
        return phoneme.upper() in self.vowel_phonemes

