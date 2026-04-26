from __future__ import annotations

from pathlib import Path

import pandas as pd

from readirect_asr.dataset.manifest import REQUIRED_MANIFEST_COLUMNS, validate_manifest_columns
from readirect_asr.audio.preprocessing import is_supported_audio_file


def _resolve_audio_candidate(raw_path: str, audio_base_path: str | Path) -> Path:
    audio_path = Path(raw_path)
    if audio_path.is_absolute():
        return audio_path
    if audio_path.exists():
        return audio_path
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("data/raw/") or normalized.startswith(str(audio_base_path).replace("\\", "/") + "/"):
        return audio_path
    return Path(audio_base_path) / audio_path


def missing_audio_paths(df: pd.DataFrame, audio_base_path: str | Path = ".") -> list[str]:
    missing: list[str] = []
    if "audio_path" not in df.columns:
        return missing

    for value in df["audio_path"].fillna(""):
        audio_path = Path(str(value))
        candidate = _resolve_audio_candidate(str(value), audio_base_path)
        if str(value).strip() and not candidate.exists():
            missing.append(str(value))
    return missing


def validate_manifest_audio_paths(
    df: pd.DataFrame,
    audio_base_path: str | Path = ".",
) -> dict[str, object]:
    missing: list[str] = []
    unsupported: list[str] = []
    if "audio_path" not in df.columns:
        return {"missing_audio_files": missing, "unsupported_audio_files": unsupported}

    for value in df["audio_path"].fillna(""):
        raw_path = str(value).strip()
        if not raw_path:
            continue
        candidate = _resolve_audio_candidate(raw_path, audio_base_path)
        if not candidate.exists():
            missing.append(raw_path)
        if not is_supported_audio_file(raw_path):
            unsupported.append(raw_path)
    return {
        "missing_audio_files": missing,
        "unsupported_audio_files": unsupported,
    }


def validate_manifest_prompt_ids(
    df: pd.DataFrame,
    content_index: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    missing_prompt_ids: list[str] = []
    prompt_ids_not_found: list[str] = []

    if "prompt_id" not in df.columns:
        return {
            "missing_prompt_ids": missing_prompt_ids,
            "prompt_ids_not_found": prompt_ids_not_found,
        }

    known_ids: set[str] = set()
    if content_index is not None and "prompt_id" in content_index.columns:
        known_ids = set(content_index["prompt_id"].fillna("").astype(str))

    for value in df["prompt_id"].fillna(""):
        prompt_id = str(value).strip()
        if not prompt_id:
            missing_prompt_ids.append(prompt_id)
        elif known_ids and prompt_id not in known_ids:
            prompt_ids_not_found.append(prompt_id)

    return {
        "missing_prompt_ids": missing_prompt_ids,
        "prompt_ids_not_found": sorted(set(prompt_ids_not_found)),
    }


def summarize_manifest(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "total_recordings": len(df),
        "total_duration_seconds": None,
        "rows_missing_expected_text": 0,
        "rows_missing_manual_transcript": 0,
        "counts_by_dataset_source": {},
        "counts_by_prompt_type": {},
        "counts_by_module_key": {},
        "counts_by_activity_type": {},
        "counts_by_error_type": {},
    }
    if "duration_seconds" in df.columns:
        durations = pd.to_numeric(df["duration_seconds"], errors="coerce")
        summary["total_duration_seconds"] = round(float(durations.fillna(0).sum()), 3)
    if "expected_text" in df.columns:
        summary["rows_missing_expected_text"] = int(df["expected_text"].fillna("").astype(str).str.strip().eq("").sum())
    if "manual_transcript" in df.columns:
        summary["rows_missing_manual_transcript"] = int(df["manual_transcript"].fillna("").astype(str).str.strip().eq("").sum())

    for column, key in (
        ("dataset_source", "counts_by_dataset_source"),
        ("prompt_type", "counts_by_prompt_type"),
        ("module_key", "counts_by_module_key"),
        ("activity_type", "counts_by_activity_type"),
        ("error_type", "counts_by_error_type"),
    ):
        if column in df.columns:
            summary[key] = df[column].fillna("").astype(str).value_counts().to_dict()
    return summary


def validate_manifest_frame(
    df: pd.DataFrame,
    audio_base_path: str | Path = ".",
    content_index: pd.DataFrame | None = None,
) -> dict[str, object]:
    audio_report = validate_manifest_audio_paths(df, audio_base_path)
    prompt_report = validate_manifest_prompt_ids(df, content_index)
    return {
        "missing_columns": validate_manifest_columns(df, REQUIRED_MANIFEST_COLUMNS),
        **audio_report,
        **prompt_report,
        "summary": summarize_manifest(df),
    }
