from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from readirect_asr.content.index import ContentBankIndex, ContentItem
from readirect_asr.content.validation import resolve_content_bank_root
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.phonemes.phoneme_enricher import enrich_text_phonemes
from readirect_asr.phonemes.phoneme_schema import PhonemeSchema


def _load_group(content_bank_path: str | Path, group: str) -> dict[str, pd.DataFrame]:
    root = resolve_content_bank_root(content_bank_path)
    group_path = root / group
    if not group_path.exists():
        return {}
    return {
        path.name: pd.read_csv(path)
        for path in sorted(group_path.glob("*.csv"))
    }


def load_assessment_content(content_bank_path: str | Path) -> dict[str, pd.DataFrame]:
    return _load_group(content_bank_path, "assessment")


def load_module_content(content_bank_path: str | Path) -> dict[str, pd.DataFrame]:
    return _load_group(content_bank_path, "modules")


def load_rules(content_bank_path: str | Path) -> dict[str, pd.DataFrame]:
    return _load_group(content_bank_path, "rules")


def _text_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _bool_value(value: Any, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "inactive"}


def _float_value(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_row(
    row: pd.Series,
    source_file: str,
    source_group: str,
) -> ContentItem | None:
    prompt_id = _text_value(row, "id", "prompt_id")
    if not prompt_id:
        return None

    prompt_text = _text_value(row, "prompt_text", "sentence_text", "passage_text", "question_text")
    expected_text = _text_value(row, "expected_answer", "target_word", "correct_answer", "expected_rhyme_family")
    if not expected_text and source_file == "reading_passages.csv":
        expected_text = _text_value(row, "passage_text")

    task_type = _text_value(row, "content_type", "question_type") or Path(source_file).stem
    module_key = _text_value(row, "module_key", "module")
    activity_type = _text_value(row, "activity_type")
    metadata = {
        key: value
        for key, value in row.to_dict().items()
        if pd.notna(value)
    }

    return ContentItem(
        prompt_id=prompt_id,
        source_file=source_file,
        source_group=source_group,
        module_key=module_key or None,
        task_type=task_type or None,
        activity_type=activity_type or None,
        prompt_text=prompt_text,
        expected_text=expected_text,
        accepted_answers=_text_value(row, "accepted_answers"),
        difficulty=_text_value(row, "difficulty") or None,
        points=_float_value(row.get("points")),
        is_active=_bool_value(row.get("is_active"), True),
        is_mastery_item=_bool_value(row.get("is_mastery_item"), False)
        if "is_mastery_item" in row
        else None,
        skill_tag=_text_value(row, "word_family", "skill_tag") or None,
        metadata=metadata,
    )


def build_content_index(
    content_bank_path: str | Path,
    cmudict_loader: CMUDictLoader | None = None,
    enrich_phonemes: bool = True,
) -> ContentBankIndex:
    root = resolve_content_bank_root(content_bank_path)
    items: list[ContentItem] = []
    loader = cmudict_loader
    schema: PhonemeSchema | None = None
    if enrich_phonemes:
        loader = loader or CMUDictLoader().load()
        schema = PhonemeSchema(loader.phone_categories, loader.symbols)

    for group in ("assessment", "modules"):
        group_path = root / group
        if not group_path.exists():
            continue
        for csv_path in sorted(group_path.glob("*.csv")):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                item = _normalize_row(row, csv_path.name, group)
                if not item:
                    continue
                if enrich_phonemes and item.expected_text and loader and schema:
                    enriched = enrich_text_phonemes(item.expected_text, loader, schema)
                    item.expected_phonemes = str(enriched["expected_phonemes"]) or None
                    item.initial_phoneme = enriched["initial_phoneme"]  # type: ignore[assignment]
                    item.vowel_phonemes = str(enriched["vowel_phonemes"]) or None
                    item.final_phoneme = enriched["final_phoneme"]  # type: ignore[assignment]
                    item.phoneme_pattern = enriched["phoneme_pattern"]  # type: ignore[assignment]
                    item.error_focus = enriched["error_focus"]  # type: ignore[assignment]
                items.append(item)
    return ContentBankIndex(items)


def load_all_content(content_bank_path: str | Path) -> ContentBankIndex:
    return build_content_index(content_bank_path, enrich_phonemes=False)

