from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_manifest_utils import MANIFEST_FIELDS, read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Wav2Vec2 training JSONL manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--min-duration-seconds", type=float, default=0.2)
    parser.add_argument("--max-duration-seconds", type=float, default=15.0)
    parser.add_argument("--require-audio", action="store_true", default=True)
    return parser.parse_args()


def validate_row(row: dict[str, Any], min_duration: float, max_duration: float, require_audio: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for field in MANIFEST_FIELDS:
        if field not in row:
            errors.append(f"missing_{field}")
    audio_path = str(row.get("audio_path", "")).strip()
    text = str(row.get("text", "")).strip()
    if not audio_path:
        errors.append("blank_audio_path")
    elif require_audio and not resolve_repo_path(audio_path).exists():
        errors.append("audio_file_not_found")
    if not text:
        errors.append("blank_text")
    try:
        duration = float(row.get("duration_seconds"))
        if duration < min_duration:
            errors.append("duration_too_short")
        elif duration < 1.0:
            warnings.append("duration_under_one_second")
        if duration > max_duration:
            warnings.append("duration_over_max_training_cap")
    except (TypeError, ValueError):
        warnings.append("missing_or_invalid_duration")
    try:
        sample_rate = int(row.get("sample_rate"))
        if sample_rate != 16000:
            warnings.append("sample_rate_will_be_resampled")
    except (TypeError, ValueError):
        warnings.append("missing_or_invalid_sample_rate")
    return errors, warnings


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    report: dict[str, Any] = {
        "manifest": str(args.manifest),
        "total_rows": len(rows),
        "valid_rows": 0,
        "invalid_rows": 0,
        "errors": Counter(),
        "warnings": Counter(),
        "dataset_counts": Counter(),
        "split_counts": Counter(),
    }
    for row in rows:
        errors, warnings = validate_row(row, args.min_duration_seconds, args.max_duration_seconds, args.require_audio)
        report["dataset_counts"][str(row.get("dataset", ""))] += 1
        report["split_counts"][str(row.get("split", ""))] += 1
        if errors:
            report["invalid_rows"] += 1
            report["errors"].update(errors)
        else:
            report["valid_rows"] += 1
        report["warnings"].update(warnings)

    printable = {
        **report,
        "errors": dict(report["errors"]),
        "warnings": dict(report["warnings"]),
        "dataset_counts": dict(report["dataset_counts"]),
        "split_counts": dict(report["split_counts"]),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    if report["valid_rows"] == 0:
        return 2
    if report["invalid_rows"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

