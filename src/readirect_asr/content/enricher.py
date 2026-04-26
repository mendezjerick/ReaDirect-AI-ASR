from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from readirect_asr.content.adaptive_tags import generate_adaptive_metadata
from readirect_asr.content.difficulty import compute_difficulty
from readirect_asr.content.enrichment_schema import ENRICHMENT_COLUMNS
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.phonemes.phoneme_enricher import extract_vowel_phonemes, text_to_phonemes


VOWEL_GRAPHEMES = set("aeiou")
SHORT_VOWEL_BY_PHONE = {"AE": "short_a", "EH": "short_e", "IH": "short_i", "AA": "short_o", "AH": "short_u"}
LETTER_PHONEMES = {
    "a": "AE",
    "b": "B",
    "c": "K",
    "d": "D",
    "e": "EH",
    "f": "F",
    "g": "G",
    "h": "HH",
    "i": "IH",
    "j": "JH",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "o": "AA",
    "p": "P",
    "q": "K",
    "r": "R",
    "s": "S",
    "t": "T",
    "u": "AH",
    "v": "V",
    "w": "W",
    "x": "K S",
    "y": "Y",
    "z": "Z",
}


class ContentEnricher:
    def __init__(self, cmudict_loader: CMUDictLoader | None = None) -> None:
        self.cmudict_loader = cmudict_loader or CMUDictLoader().load()

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in df.fillna("").iterrows():
            row_dict = row.to_dict()
            enrichment = self.enrich_row(row_dict)
            rows.append({**row_dict, **enrichment})
        return pd.DataFrame(rows)

    def enrich_row(self, row: dict[str, Any]) -> dict[str, Any]:
        text = str(row.get("expected_text") or row.get("target_word") or row.get("prompt_text") or "")
        text_type = self.classify_item_text_type(text, row)
        if text_type == "single_letter":
            enrichment = self.enrich_letter_item(text, row)
        elif text_type == "single_word":
            enrichment = self.enrich_word_item(text, row)
        elif text_type in {"phrase", "sentence"}:
            enrichment = self.enrich_sentence_item(text, row)
        else:
            enrichment = self._base_enrichment(text, row, [], [], text_type)
            enrichment["skill_group"] = self._skill_group(row, text_type)
            enrichment["error_focus"] = self.infer_error_focus(row, [])

        difficulty = compute_difficulty(row, enrichment)
        enrichment.update(difficulty)
        adaptive = generate_adaptive_metadata(row, enrichment)
        enrichment.update(adaptive)
        for column in ENRICHMENT_COLUMNS:
            enrichment.setdefault(column, row.get(column, ""))
        enrichment["enrichment_status"] = "needs_review" if enrichment["needs_manual_review"] else "ok"
        return enrichment

    def classify_item_text_type(self, text: str, row: dict[str, Any] | None = None) -> str:
        clean = str(text or "").strip()
        row = row or {}
        if str(row.get("source_file", "")).startswith("comprehension_questions") or str(row.get("task_type", "")).lower() == "multiple_choice":
            return "comprehension_question"
        if len(clean) == 1 and clean.isalpha():
            return "single_letter"
        words = re.findall(r"[A-Za-z']+", clean)
        if len(words) == 1:
            return "single_word"
        if len(words) <= 4 and not re.search(r"[.!?]", clean):
            return "phrase"
        return "sentence"

    def enrich_word_item(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        words = self._words(text)
        phonemes = text_to_phonemes(text, self.cmudict_loader)
        missing = [] if phonemes else words
        enrichment = self._base_enrichment(text, row, phonemes, missing, "single_word")
        onset_rime = self.infer_onset_rime(words[0] if words else text, phonemes)
        enrichment.update(onset_rime)
        enrichment["word_family"] = self.infer_word_family(words[0] if words else text) or ""
        enrichment["skill_group"] = self._skill_group(row, "single_word")
        enrichment["skill_tag"] = self.infer_skill_tag(row, phonemes, enrichment)
        enrichment["error_focus"] = self.infer_error_focus(row, phonemes, enrichment)
        enrichment["target_position"] = self._target_position(enrichment["error_focus"])
        enrichment["target_phoneme"] = self._target_phoneme(phonemes, enrichment["target_position"])
        enrichment["target_grapheme"] = self._target_grapheme(words[0] if words else text, enrichment["target_position"])
        return enrichment

    def enrich_sentence_item(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        words = self._words(text)
        phonemes = text_to_phonemes(text, self.cmudict_loader)
        missing = self._missing_words(words)
        enrichment = self._base_enrichment(text, row, phonemes, missing, "sentence" if len(words) > 4 else "phrase")
        enrichment["word_count"] = len(words)
        enrichment["sentence_length_bucket"] = "short" if len(words) <= 5 else "medium" if len(words) <= 12 else "long"
        enrichment["skill_group"] = self._skill_group(row, "sentence")
        enrichment["skill_tag"] = "word_accuracy_in_sentence" if enrichment["skill_group"] == "sentence_reading" else "fluency_pacing"
        enrichment["error_focus"] = self.infer_error_focus(row, phonemes, enrichment)
        enrichment["target_position"] = ""
        enrichment["target_phoneme"] = ""
        enrichment["target_grapheme"] = ""
        return enrichment

    def enrich_letter_item(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        letter = str(text or "").strip().lower()[:1]
        phonemes = LETTER_PHONEMES.get(letter, "").split()
        enrichment = self._base_enrichment(text, row, phonemes, [] if phonemes else [letter], "single_letter")
        enrichment["skill_group"] = "letter_sound"
        enrichment["skill_tag"] = "letter_sound"
        enrichment["error_focus"] = "initial_consonant" if phonemes and phonemes[0] not in {"AE", "EH", "IH", "AA", "AH"} else "vowel_sound"
        enrichment["target_position"] = "initial"
        enrichment["target_phoneme"] = " ".join(phonemes)
        enrichment["target_grapheme"] = letter.upper()
        enrichment["word_family"] = ""
        enrichment["rime_unit"] = ""
        enrichment["onset_unit"] = letter.upper()
        return enrichment

    def estimate_syllables(self, text: str) -> int:
        words = self._words(text)
        count = 0
        for word in words:
            groups = re.findall(r"[aeiouy]+", word.lower())
            count += max(1, len(groups))
        return count

    def infer_phoneme_pattern(self, phonemes: list[str]) -> str:
        vowels = set(extract_vowel_phonemes(phonemes))
        return "".join("V" if phone in vowels else "C" for phone in phonemes)

    def infer_word_family(self, word: str) -> str | None:
        clean = re.sub(r"[^a-z]", "", word.lower())
        if len(clean) < 2:
            return None
        rime = clean[-2:]
        return f"{rime}_family"

    def infer_onset_rime(self, word: str, phonemes: list[str]) -> dict[str, str]:
        clean = re.sub(r"[^a-z]", "", word.lower())
        first_vowel = next((i for i, char in enumerate(clean) if char in VOWEL_GRAPHEMES), None)
        if first_vowel is None:
            return {"onset_unit": clean, "rime_unit": ""}
        return {"onset_unit": clean[:first_vowel], "rime_unit": clean[first_vowel:]}

    def infer_skill_tag(self, row: dict[str, Any], phonemes: list[str], enrichment: dict[str, Any] | None = None) -> str:
        enrichment = enrichment or {}
        module = str(row.get("module_key", "")).lower()
        activity = str(row.get("activity_type", "")).lower()
        if "module_1" in module or "letter" in activity or "sound" in activity:
            return "letter_sound"
        if "module_3" in module or "sentence" in activity:
            return "sentence_tracking"
        pattern = str(enrichment.get("phoneme_pattern") or self.infer_phoneme_pattern(phonemes))
        vowels = extract_vowel_phonemes(phonemes)
        if pattern == "CVC" and vowels:
            return f"CVC_{SHORT_VOWEL_BY_PHONE.get(vowels[0], 'vowel')}"
        word_family = str(enrichment.get("word_family", ""))
        if word_family:
            return word_family
        return "word_reading"

    def infer_error_focus(self, row: dict[str, Any], phonemes: list[str], enrichment: dict[str, Any] | None = None) -> str:
        enrichment = enrichment or {}
        source = str(row.get("source_file", "")).lower()
        task_type = str(row.get("task_type", "")).lower()
        activity = str(row.get("activity_type", "")).lower()
        if "comprehension_questions" in source or task_type == "multiple_choice":
            question = str(row.get("prompt_text", "")).lower()
            if "why" in question or "how" in question:
                return "comprehension_inference"
            if "sequence" in question or "first" in question or "next" in question:
                return "sequencing"
            return "comprehension_detail"
        if "sentence" in activity or "fluency" in activity:
            return "fluency_pacing" if "timed" in activity or "fluency" in activity else "sentence_tracking"
        vowels = extract_vowel_phonemes(phonemes)
        pattern = str(enrichment.get("phoneme_pattern") or self.infer_phoneme_pattern(phonemes))
        if pattern == "CVC" and vowels:
            return "vowel_sound"
        if pattern.endswith("C"):
            return "final_consonant"
        if pattern.startswith("C"):
            return "initial_consonant"
        return "unknown"

    def infer_difficulty(self, row: dict[str, Any], phonemes: list[str]) -> dict[str, object]:
        return compute_difficulty(row, {"phoneme_count": len(phonemes), "expected_phonemes": " ".join(phonemes)})

    def infer_adaptive_bucket(self, row: dict[str, Any], enrichment: dict[str, Any]) -> str:
        return str(generate_adaptive_metadata(row, enrichment)["adaptive_bucket"])

    def _base_enrichment(self, text: str, row: dict[str, Any], phonemes: list[str], missing_words: list[str], text_type: str) -> dict[str, Any]:
        vowels = extract_vowel_phonemes(phonemes)
        warnings = []
        if missing_words:
            warnings.append("missing_cmudict_words")
        if text_type == "comprehension_question":
            warnings.append("minimal_phoneme_enrichment")
        return {
            "item_text_type": text_type,
            "expected_phonemes": " ".join(phonemes),
            "initial_phoneme": phonemes[0] if phonemes else "",
            "vowel_phonemes": " ".join(vowels),
            "final_phoneme": phonemes[-1] if phonemes else "",
            "phoneme_pattern": self.infer_phoneme_pattern(phonemes),
            "phoneme_count": len(phonemes),
            "syllable_estimate": self.estimate_syllables(text),
            "has_cmudict_match": bool(phonemes) and not missing_words,
            "cmudict_missing_words": "|".join(missing_words),
            "skill_tag": "",
            "skill_group": self._skill_group(row, text_type),
            "error_focus": "unknown",
            "target_position": "",
            "target_phoneme": "",
            "target_grapheme": "",
            "word_family": "",
            "rime_unit": "",
            "onset_unit": "",
            "enrichment_warnings": "|".join(warnings),
            "needs_manual_review": bool(warnings),
        }

    def _skill_group(self, row: dict[str, Any], text_type: str) -> str:
        module = str(row.get("module_key", "")).lower()
        source = str(row.get("source_file", "")).lower()
        activity = str(row.get("activity_type", "")).lower()
        task_type = str(row.get("task_type", "")).lower()
        if "comprehension" in source or task_type == "multiple_choice":
            return "comprehension"
        if "reading_passages" in source:
            return "fluency"
        if "module_1" in module or text_type == "single_letter" or "letter" in activity:
            return "letter_sound"
        if "rhyme" in task_type:
            return "phonemic_awareness"
        if "module_2" in module or text_type == "single_word":
            return "word_reading"
        if "module_3" in module or text_type in {"sentence", "phrase"}:
            return "sentence_reading"
        return "unknown"

    def _missing_words(self, words: list[str]) -> list[str]:
        return [word for word in words if not self.cmudict_loader.get_primary_pronunciation(word)]

    def _target_position(self, error_focus: str) -> str:
        return {"initial_consonant": "initial", "final_consonant": "final", "vowel_sound": "medial"}.get(error_focus, "")

    def _target_phoneme(self, phonemes: list[str], position: str) -> str:
        if not phonemes:
            return ""
        if position == "initial":
            return phonemes[0]
        if position == "final":
            return phonemes[-1]
        if position == "medial":
            vowels = extract_vowel_phonemes(phonemes)
            return vowels[0] if vowels else ""
        return ""

    def _target_grapheme(self, word: str, position: str) -> str:
        clean = re.sub(r"[^a-z]", "", word.lower())
        if not clean:
            return ""
        if position == "initial":
            return clean[0]
        if position == "final":
            return clean[-1]
        if position == "medial":
            return next((char for char in clean if char in VOWEL_GRAPHEMES), "")
        return ""

    def _words(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z']+", str(text or ""))

    @staticmethod
    def metadata_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

