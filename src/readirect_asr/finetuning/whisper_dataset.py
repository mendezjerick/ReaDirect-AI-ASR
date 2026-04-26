from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_JSONL_FIELDS = ("audio", "sentence")


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    rows: list[dict[str, Any]] = []
    if not dataset_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}: {exc}") from exc
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def load_whisper_dataset(train_jsonl: str | Path, validation_jsonl: str | Path, test_jsonl: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    dataset = {
        "train": load_jsonl_dataset(train_jsonl),
        "validation": load_jsonl_dataset(validation_jsonl),
    }
    if test_jsonl and Path(test_jsonl).exists():
        dataset["test"] = load_jsonl_dataset(test_jsonl)
    return dataset


def validate_whisper_jsonl(
    path: str | Path,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
) -> dict[str, Any]:
    dataset_path = Path(path)
    report = {
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "issues": [],
        "warnings": [],
    }
    if not dataset_path.exists():
        report["issues"].append("file_not_found")
        return report
    rows = load_jsonl_dataset(dataset_path)
    report["total_rows"] = len(rows)
    for row in rows:
        row_issues = _validate_row(row, min_duration, max_duration)
        if row_issues:
            report["invalid_rows"] += 1
            report["issues"].extend(row_issues)
        else:
            report["valid_rows"] += 1
    report["issues"] = sorted(set(report["issues"]))
    return report


def summarize_whisper_dataset(dataset: dict[str, list[dict[str, Any]]] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(dataset, list):
        split_rows = {"dataset": dataset}
    else:
        split_rows = dataset
    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    total_rows = sum(split_counts.values())
    total_seconds = 0.0
    missing_audio = 0
    blank_sentence = 0
    for rows in split_rows.values():
        for row in rows:
            if not Path(str(row.get("audio", ""))).exists():
                missing_audio += 1
            if not str(row.get("sentence", "")).strip():
                blank_sentence += 1
            total_seconds += _float_or_zero(row.get("duration_seconds"))
    return {
        "split_counts": split_counts,
        "total_rows": total_rows,
        "total_hours": round(total_seconds / 3600.0, 6),
        "missing_audio_rows": missing_audio,
        "blank_sentence_rows": blank_sentence,
    }


def _validate_row(row: dict[str, Any], min_duration: float, max_duration: float) -> list[str]:
    issues: list[str] = []
    for field in REQUIRED_JSONL_FIELDS:
        if field not in row:
            issues.append(f"missing_{field}")
    audio = str(row.get("audio", "")).strip()
    sentence = str(row.get("sentence", "")).strip()
    if not audio:
        issues.append("blank_audio_path")
    elif not Path(audio).exists():
        issues.append("audio_file_not_found")
    if not sentence:
        issues.append("blank_sentence")
    duration = row.get("duration_seconds")
    if duration not in (None, ""):
        try:
            duration_value = float(duration)
            if duration_value < min_duration:
                issues.append("duration_too_short")
            if duration_value > max_duration:
                issues.append("duration_too_long")
        except (TypeError, ValueError):
            issues.append("invalid_duration")
    return issues


def _float_or_zero(value: Any) -> float:
    try:
        return 0.0 if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return 0.0
