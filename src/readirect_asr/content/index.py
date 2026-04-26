from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ContentItem:
    prompt_id: str
    source_file: str
    source_group: str
    module_key: str | None = None
    task_type: str | None = None
    activity_type: str | None = None
    prompt_text: str = ""
    expected_text: str = ""
    accepted_answers: str = ""
    difficulty: str | None = None
    points: float | None = None
    is_active: bool = True
    is_mastery_item: bool | None = None
    expected_phonemes: str | None = None
    initial_phoneme: str | None = None
    vowel_phonemes: str | None = None
    final_phoneme: str | None = None
    phoneme_pattern: str | None = None
    skill_tag: str | None = None
    error_focus: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ContentBankIndex:
    def __init__(self, items: list[ContentItem] | None = None) -> None:
        self.items = items or []

    def duplicate_prompt_ids(self) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in self.items:
            if item.prompt_id in seen:
                duplicates.add(item.prompt_id)
            seen.add(item.prompt_id)
        return sorted(duplicates)

    def get_item(self, prompt_id: str) -> ContentItem | None:
        for item in self.items:
            if item.prompt_id == prompt_id:
                return item
        return None

    def find_by_module(self, module_key: str) -> list[ContentItem]:
        return [item for item in self.items if item.module_key == module_key]

    def find_by_activity_type(self, activity_type: str) -> list[ContentItem]:
        return [item for item in self.items if item.activity_type == activity_type]

    def find_by_task_type(self, task_type: str) -> list[ContentItem]:
        return [item for item in self.items if item.task_type == task_type]

    def search_expected_text(self, text: str) -> list[ContentItem]:
        query = str(text or "").lower()
        return [item for item in self.items if query in item.expected_text.lower()]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for item in self.items:
            row = asdict(item)
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=True, sort_keys=True)
            rows.append(row)
        return pd.DataFrame(rows)

    def save_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(output_path, index=False)


def content_index_from_dataframe(df: pd.DataFrame) -> ContentBankIndex:
    items: list[ContentItem] = []
    for _, row in df.fillna("").iterrows():
        metadata_raw = row.get("metadata", {})
        if isinstance(metadata_raw, str) and metadata_raw.strip():
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": metadata_raw}
        else:
            metadata = {}
        items.append(
            ContentItem(
                prompt_id=str(row.get("prompt_id", "")),
                source_file=str(row.get("source_file", "")),
                source_group=str(row.get("source_group", "")),
                module_key=str(row.get("module_key", "")) or None,
                task_type=str(row.get("task_type", "")) or None,
                activity_type=str(row.get("activity_type", "")) or None,
                prompt_text=str(row.get("prompt_text", "")),
                expected_text=str(row.get("expected_text", "")),
                accepted_answers=str(row.get("accepted_answers", "")),
                difficulty=str(row.get("difficulty", "")) or None,
                points=float(row["points"]) if str(row.get("points", "")).strip() else None,
                is_active=str(row.get("is_active", "1")).lower() not in {"0", "false", "no"},
                is_mastery_item=str(row.get("is_mastery_item", "")).lower() in {"1", "true", "yes"}
                if str(row.get("is_mastery_item", "")).strip()
                else None,
                expected_phonemes=str(row.get("expected_phonemes", "")) or None,
                initial_phoneme=str(row.get("initial_phoneme", "")) or None,
                vowel_phonemes=str(row.get("vowel_phonemes", "")) or None,
                final_phoneme=str(row.get("final_phoneme", "")) or None,
                phoneme_pattern=str(row.get("phoneme_pattern", "")) or None,
                skill_tag=str(row.get("skill_tag", "")) or None,
                error_focus=str(row.get("error_focus", "")) or None,
                metadata=metadata,
            )
        )
    return ContentBankIndex(items)

