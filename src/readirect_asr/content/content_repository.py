from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from readirect_asr.scoring.answer_matching import normalize_answer


DEFAULT_METADATA_FIELDS = [
    "prompt_id",
    "source_file",
    "source_group",
    "module_key",
    "task_type",
    "activity_type",
    "prompt_text",
    "expected_text",
    "accepted_answers",
]

DEFAULT_ENRICHMENT_FIELDS = [
    "expected_phonemes",
    "initial_phoneme",
    "vowel_phonemes",
    "final_phoneme",
    "phoneme_pattern",
    "skill_tag",
    "skill_group",
    "error_focus",
    "target_position",
    "target_phoneme",
    "difficulty_level",
    "difficulty_score",
    "adaptive_bucket",
    "recommended_for_error_type",
    "practice_role",
    "mastery_candidate",
    "needs_manual_review",
]


class ContentRepository:
    def __init__(
        self,
        content_index_path: str | Path = "data/manifests/content_index.csv",
        enriched_content_index_path: str | Path = "content_bank_enriched/enriched_content_index.csv",
        prefer_enriched_content: bool = True,
    ) -> None:
        self.content_index_path = Path(content_index_path)
        self.enriched_content_index_path = Path(enriched_content_index_path)
        self.prefer_enriched_content = prefer_enriched_content
        self._df: pd.DataFrame | None = None
        self.loaded_path: Path | None = None

    def load(self) -> "ContentRepository":
        candidates = (
            [self.enriched_content_index_path, self.content_index_path]
            if self.prefer_enriched_content
            else [self.content_index_path, self.enriched_content_index_path]
        )
        for candidate in candidates:
            if candidate.exists():
                self._df = pd.read_csv(candidate)
                self.loaded_path = candidate
                return self
        self._df = pd.DataFrame()
        self.loaded_path = None
        return self

    def is_loaded(self) -> bool:
        if self._df is None:
            self.load()
        return self._df is not None and not self._df.empty

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df if self._df is not None else pd.DataFrame()

    def get_by_prompt_id(self, prompt_id: str | None) -> dict[str, Any] | None:
        if not prompt_id or "prompt_id" not in self.df.columns:
            return None
        matches = self.df[self.df["prompt_id"].fillna("").astype(str) == str(prompt_id)]
        if matches.empty:
            return None
        return _clean_row(matches.iloc[0].to_dict())

    def find_by_expected_text(self, expected_text: str | None) -> dict[str, Any] | None:
        if not expected_text or "expected_text" not in self.df.columns:
            return None
        query = normalize_answer(expected_text)
        if not query:
            return None
        normalized = self.df["expected_text"].fillna("").astype(str).map(normalize_answer)
        matches = self.df[normalized == query]
        if matches.empty:
            return None
        return _clean_row(matches.iloc[0].to_dict())

    def get_metadata(self, prompt_id: str | None = None, expected_text: str | None = None) -> dict[str, Any]:
        row = self.get_by_prompt_id(prompt_id) or self.find_by_expected_text(expected_text)
        if not row:
            return {}
        return {field: row.get(field) for field in DEFAULT_METADATA_FIELDS if field in row}

    def get_enrichment(self, prompt_id: str | None = None, expected_text: str | None = None) -> dict[str, Any]:
        row = self.get_by_prompt_id(prompt_id) or self.find_by_expected_text(expected_text)
        if not row:
            return {}
        return {field: row.get(field) for field in DEFAULT_ENRICHMENT_FIELDS if field in row}


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            clean[key] = None
        elif isinstance(value, float) and math.isnan(value):
            clean[key] = ""
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean

