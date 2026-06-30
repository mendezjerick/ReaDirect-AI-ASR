#!/usr/bin/env python3
"""Validate the AI-ASR copy of the current ReaDirect module CSV banks."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"

COMMON_REQUIRED_COLUMNS = [
    "prompt_id",
    "source_file",
    "source_group",
    "module_key",
    "task_type",
    "activity_type",
    "prompt_text",
    "expected_text",
    "accepted_answers",
    "is_active",
    "is_mastery_item",
    "metadata",
]

MODULE_SPECS = {
    "module1_letter_sound_activities.csv": {
        "module_key": "module_1",
        "required_columns": COMMON_REQUIRED_COLUMNS,
        "counts": {
            "letter_pair_identification": 26,
            "highlighted_first_letter": 26,
            "first_letter_identification": 26,
            "missing_first_letter": 26,
            "mastery_check": 26,
        },
    },
    "module2_word_reading_activities.csv": {
        "module_key": "module_2",
        "required_columns": COMMON_REQUIRED_COLUMNS,
        "counts": {
            "display_word_reading": 50,
            "split_word_reading": 30,
            "highlighted_rhyme_word": 34,
            "highlighted_sentence_word": 25,
            "mastery_check": 25,
        },
    },
    "module3_sentence_fluency_activities.csv": {
        "module_key": "module_3",
        "required_columns": [
            *COMMON_REQUIRED_COLUMNS,
            "target_read_time_seconds",
            "min_fluent_time_seconds",
            "max_fluent_time_seconds",
            "target_wcpm",
            "min_expected_wcpm",
            "max_expected_wcpm",
            "pace_feedback_rule",
            "pace_mastery_required",
        ],
        "counts": {
            "simple_sentence_reading": 50,
            "comma_pause_reading": 50,
            "full_stop_pause_reading": 35,
            "mixed_punctuation_fluency": 35,
            "mastery_check": 36,
        },
    },
    "module_activity_selection_rules.csv": {
        "module_key": None,
        "required_columns": [
            "prompt_id",
            "source_file",
            "source_group",
            "module_key",
            "activity_type",
            "is_active",
            "metadata",
        ],
        "counts": {
            "letter_pair_identification": 1,
            "highlighted_first_letter": 1,
            "first_letter_identification": 1,
            "missing_first_letter": 1,
            "display_word_reading": 1,
            "split_word_reading": 1,
            "highlighted_rhyme_word": 1,
            "highlighted_sentence_word": 1,
            "simple_sentence_reading": 1,
            "comma_pause_reading": 1,
            "full_stop_pause_reading": 1,
            "mixed_punctuation_fluency": 1,
            "mastery_check": 3,
        },
    },
}


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def validate_file(filename: str, spec: dict[str, object], errors: list[str]) -> None:
    header, rows = load_csv(MODULES / filename)
    required_columns = spec["required_columns"]  # type: ignore[assignment]
    expected_counts = spec["counts"]  # type: ignore[assignment]
    expected_total = sum(Counter(expected_counts).values())
    expected_module_key = spec["module_key"]

    missing = [column for column in required_columns if column not in header]
    require(not missing, f"{filename} is missing required columns: {missing}", errors)
    require(len(rows) == expected_total, f"{filename} has {len(rows)} rows, expected {expected_total}", errors)

    ids = [row["prompt_id"] for row in rows if "prompt_id" in row]
    duplicate_ids = [item_id for item_id, count in Counter(ids).items() if count > 1]
    require(not duplicate_ids, f"{filename} has duplicate prompt IDs: {duplicate_ids}", errors)

    actual_counts = Counter(row.get("activity_type", "") for row in rows)
    require(
        actual_counts == Counter(expected_counts),
        f"{filename} activity counts are {dict(actual_counts)}, expected {expected_counts}",
        errors,
    )

    for row in rows:
        row_id = row.get("prompt_id", "<missing>")
        activity_type = row.get("activity_type", "")
        expected = row.get("expected_text", "").strip().casefold()
        accepted = [answer.strip().casefold() for answer in row.get("accepted_answers", "").split("|") if answer.strip()]

        if expected_module_key:
            require(row.get("module_key") == expected_module_key, f"{filename} {row_id} has wrong module_key", errors)
        require(activity_type in expected_counts, f"{filename} {row_id} has unknown activity_type", errors)
        require(activity_type or filename == "module_activity_selection_rules.csv", f"{filename} {row_id} has blank activity_type", errors)
        if filename != "module_activity_selection_rules.csv":
            require(bool(row.get("prompt_text", "").strip()), f"{filename} {row_id} has blank prompt_text", errors)
            require(bool(expected), f"{filename} {row_id} has blank expected_text", errors)
            require(expected in accepted, f"{filename} {row_id} accepted_answers omit expected_text", errors)
            require(
                truthy(row.get("is_mastery_item", "")) == (activity_type == "mastery_check"),
                f"{filename} {row_id} has incorrect is_mastery_item",
                errors,
            )


def main() -> int:
    errors: list[str] = []

    try:
        for filename, spec in MODULE_SPECS.items():
            validate_file(filename, spec, errors)
    except Exception as exc:  # noqa: BLE001 - print validation failures plainly.
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed: module CSVs match the current lesson activity keys and row counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
