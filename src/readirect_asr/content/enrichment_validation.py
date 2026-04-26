from __future__ import annotations

import pandas as pd

from readirect_asr.content.enrichment_schema import (
    ENRICHMENT_COLUMNS,
    VALID_DIFFICULTY_LEVELS,
    VALID_ERROR_FOCUS,
    VALID_PRACTICE_ROLES,
    VALID_SKILL_GROUPS,
)


VALID_RECOMMENDED_ERROR_TYPES = {
    "final_sound_error",
    "initial_sound_error",
    "vowel_error",
    "skipped_word",
    "partial_sentence",
    "comprehension_detail_error",
    "comprehension_inference_error",
    "word_family_error",
    "incorrect_general",
}


def validate_required_enrichment_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ENRICHMENT_COLUMNS if column not in df.columns]


def validate_phoneme_fields(df: pd.DataFrame) -> dict[str, int]:
    return {
        "missing_expected_phonemes": int(df.get("expected_phonemes", pd.Series(dtype=str)).fillna("").astype(str).str.strip().eq("").sum()),
        "missing_cmudict_words_count": int(df.get("cmudict_missing_words", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()),
        "needs_manual_review_count": _truthy_count(df, "needs_manual_review"),
    }


def validate_skill_tags(df: pd.DataFrame) -> dict[str, object]:
    skill_values = set(df.get("skill_group", pd.Series(dtype=str)).fillna("").astype(str))
    focus_values = set(df.get("error_focus", pd.Series(dtype=str)).fillna("").astype(str))
    return {
        "invalid_skill_groups": sorted(value for value in skill_values if value and value not in VALID_SKILL_GROUPS),
        "invalid_error_focus": sorted(value for value in focus_values if value and value not in VALID_ERROR_FOCUS),
    }


def validate_adaptive_metadata(df: pd.DataFrame) -> dict[str, object]:
    difficulty_values = set(df.get("difficulty_level", pd.Series(dtype=str)).fillna("").astype(str))
    practice_values = set(df.get("practice_role", pd.Series(dtype=str)).fillna("").astype(str))
    recommended_values = set(df.get("recommended_for_error_type", pd.Series(dtype=str)).fillna("").astype(str))
    return {
        "invalid_difficulty_levels": sorted(value for value in difficulty_values if value and value not in VALID_DIFFICULTY_LEVELS),
        "invalid_practice_roles": sorted(value for value in practice_values if value and value not in VALID_PRACTICE_ROLES),
        "invalid_recommended_error_types": sorted(value for value in recommended_values if value and value not in VALID_RECOMMENDED_ERROR_TYPES),
    }


def validate_enriched_dataframe(df: pd.DataFrame) -> dict[str, object]:
    duplicate_prompt_ids: list[str] = []
    if "prompt_id" in df.columns:
        counts = df["prompt_id"].fillna("").astype(str).value_counts()
        duplicate_prompt_ids = sorted(value for value, count in counts.items() if value and count > 1)
    expected_missing = 0
    if "expected_text" in df.columns:
        expected_missing = int(df["expected_text"].fillna("").astype(str).str.strip().eq("").sum())
    report = {
        "missing_columns": validate_required_enrichment_columns(df),
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "rows_missing_expected_text": expected_missing,
        **validate_phoneme_fields(df),
        **validate_skill_tags(df),
        **validate_adaptive_metadata(df),
    }
    report["ok"] = not report["missing_columns"] and not report["invalid_skill_groups"] and not report["invalid_error_focus"] and not report["invalid_difficulty_levels"]
    return report


def _truthy_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(str).str.lower().isin({"1", "true", "yes"}).sum())
