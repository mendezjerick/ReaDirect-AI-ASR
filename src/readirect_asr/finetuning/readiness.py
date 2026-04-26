from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def check_finetuning_readiness(
    manifest_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None = None,
    min_total_hours: float = 2.0,
    min_rows: int = 500,
    min_transcript_coverage: float = 0.90,
) -> dict[str, Any]:
    df = manifest_df.copy() if manifest_df is not None else pd.DataFrame()
    issues: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    total_rows = len(df)
    transcript_col = _choose_transcript_col(df)
    transcript_coverage = _coverage(df, transcript_col) if transcript_col else 0.0
    audio_available_rate = _audio_available_rate(df)
    usable_mask = _usable_rows(df, transcript_col)
    usable_rows = int(usable_mask.sum()) if not df.empty else 0
    total_hours = _total_hours(df)

    if total_rows < min_rows:
        issues.append("too_few_rows")
        recommendations.append(f"Collect or convert at least {min_rows} labeled rows before fine-tuning.")
    if total_hours is not None and total_hours < min_total_hours:
        issues.append("too_little_audio_duration")
        recommendations.append(f"Target at least {min_total_hours} total labeled audio hours.")
    if total_hours is None:
        warnings.append("duration_missing_or_unusable")
        recommendations.append("Add duration_seconds where possible for better readiness estimates.")
    if transcript_coverage < min_transcript_coverage:
        issues.append("low_transcript_coverage")
        recommendations.append("Increase manual transcript coverage before training.")
    if audio_available_rate < 0.95:
        warnings.append("some_audio_paths_missing")
    if _duplicate_count(df, "audio_path") > 0:
        warnings.append("duplicate_audio_paths_detected")
    if _blank_count(df, transcript_col) > 0:
        warnings.append("blank_reference_transcripts_detected")
    if baseline_df is not None and not baseline_df.empty:
        if not any(col in baseline_df.columns for col in ("asr_transcript", "normalized_transcript")):
            warnings.append("baseline_missing_transcript_columns")
    elif baseline_df is not None:
        warnings.append("baseline_dataframe_empty")

    ready = not issues
    status = "ready" if ready else "needs_more_data" if "too_few_rows" in issues or "too_little_audio_duration" in issues else "not_ready"
    return {
        "ready": ready,
        "status": status,
        "total_rows": total_rows,
        "usable_rows": usable_rows,
        "total_hours": round(total_hours, 6) if total_hours is not None else None,
        "transcript_coverage": transcript_coverage,
        "audio_available_rate": audio_available_rate,
        "issues": issues,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _choose_transcript_col(df: pd.DataFrame) -> str | None:
    for col in ("manual_transcript", "expected_text", "sentence"):
        if col in df.columns:
            return col
    return None


def _coverage(df: pd.DataFrame, column: str | None) -> float:
    if df.empty or not column or column not in df.columns:
        return 0.0
    present = df[column].fillna("").astype(str).str.strip().ne("").sum()
    return round(float(present / len(df)), 6)


def _audio_available_rate(df: pd.DataFrame) -> float:
    if df.empty or "audio_path" not in df.columns:
        return 0.0
    paths = df["audio_path"].fillna("").astype(str)
    existing = sum(1 for value in paths if value and Path(value).exists())
    nonblank = paths.str.strip().ne("").sum()
    if nonblank == 0:
        return 0.0
    return round(float(existing / nonblank), 6)


def _usable_rows(df: pd.DataFrame, transcript_col: str | None) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    mask = pd.Series([True] * len(df), index=df.index)
    if transcript_col and transcript_col in df.columns:
        mask &= df[transcript_col].fillna("").astype(str).str.strip().ne("")
    if "audio_path" in df.columns:
        mask &= df["audio_path"].fillna("").astype(str).str.strip().ne("")
    if "duration_seconds" in df.columns:
        durations = pd.to_numeric(df["duration_seconds"], errors="coerce")
        mask &= durations.isna() | ((durations >= 0.3) & (durations <= 30.0))
    return mask


def _total_hours(df: pd.DataFrame) -> float | None:
    if "duration_seconds" not in df.columns:
        return None
    durations = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()
    if durations.empty:
        return None
    return float(durations.sum() / 3600.0)


def _duplicate_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = df[column].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    return int(values.duplicated().sum())


def _blank_count(df: pd.DataFrame, column: str | None) -> int:
    if not column or column not in df.columns:
        return len(df)
    return int(df[column].fillna("").astype(str).str.strip().eq("").sum())
