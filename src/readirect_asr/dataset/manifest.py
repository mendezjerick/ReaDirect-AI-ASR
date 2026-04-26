from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = [
    "recording_id",
    "dataset_source",
    "learner_id_anonymized",
    "speaker_id_anonymized",
    "speaker_type",
    "age_group",
    "gender",
    "l1_language",
    "grade_level",
    "prompt_id",
    "prompt_type",
    "module_key",
    "activity_type",
    "prompt_text",
    "expected_text",
    "accepted_answers",
    "expected_phonemes",
    "initial_phoneme",
    "vowel_phonemes",
    "final_phoneme",
    "phoneme_pattern",
    "audio_path",
    "duration_seconds",
    "sentence_score",
    "word_score",
    "phoneme_score",
    "word_labels",
    "phoneme_labels",
    "timestamps",
    "manual_transcript",
    "asr_transcript",
    "normalized_transcript",
    "human_correct",
    "error_type",
    "similarity_label",
    "stt_confidence",
    "recording_condition",
    "noise_flag",
    "license_notes",
    "split",
    "notes",
    "row_status",
]


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def validate_manifest_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_MANIFEST_COLUMNS,
) -> list[str]:
    required = list(required_columns)
    return [column for column in required if column not in df.columns]


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_MANIFEST_COLUMNS)
