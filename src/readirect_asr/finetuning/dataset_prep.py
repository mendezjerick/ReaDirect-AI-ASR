from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from readirect_asr.finetuning.split import create_splits
from readirect_asr.text.normalization import normalize_whitespace


def prepare_whisper_dataset(
    manifest_df: pd.DataFrame,
    output_dir: str | Path,
    audio_col: str = "audio_path",
    transcript_col: str = "manual_transcript",
    split_col: str = "split",
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    df = manifest_df.copy()
    if split_col not in df.columns:
        df = create_splits(df)
        split_col = "split"
    output = Path(output_dir)
    counts = {"train": 0, "validation": 0, "test": 0}
    skipped: dict[str, int] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}

    for _, row in df.iterrows():
        reason = _skip_reason(row, audio_col, transcript_col, min_duration, max_duration)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        split = str(row.get(split_col, "train") or "train")
        if split not in rows_by_split:
            split = "train"
        item = {
            "audio": str(row.get(audio_col, "")),
            "sentence": normalize_whitespace(str(row.get(transcript_col, ""))).strip(),
            "dataset_source": str(row.get("dataset_source", "")),
            "recording_id": str(row.get("recording_id", "")),
            "duration_seconds": _float_or_none(row.get("duration_seconds")),
        }
        rows_by_split[split].append(item)
        counts[split] += 1

    total_hours = sum((item.get("duration_seconds") or 0.0) for rows in rows_by_split.values() for item in rows) / 3600.0
    summary = {
        "output_dir": str(output),
        "counts": counts,
        "total_rows": sum(counts.values()),
        "total_hours": round(total_hours, 6),
        "skipped": skipped,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        _write_jsonl(output / f"{split}.jsonl", rows)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _skip_reason(row: pd.Series, audio_col: str, transcript_col: str, min_duration: float, max_duration: float) -> str | None:
    audio = str(row.get(audio_col, "") or "").strip()
    transcript = str(row.get(transcript_col, "") or "").strip()
    if not audio:
        return "missing_audio_path"
    if not Path(audio).exists():
        return "audio_file_not_found"
    if not transcript:
        return "blank_transcript"
    duration = _float_or_none(row.get("duration_seconds"))
    if duration is not None and duration < min_duration:
        return "duration_too_short"
    if duration is not None and duration > max_duration:
        return "duration_too_long"
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
