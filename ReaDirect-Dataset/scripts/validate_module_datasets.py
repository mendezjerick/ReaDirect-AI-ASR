#!/usr/bin/env python3
"""Validate ReaDirect module activity CSV banks."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"

MODULE_SPECS = {
    "module1_letter_sound_activities.csv": {
        "module_key": "module_1",
        "header": [
            "id",
            "module_key",
            "activity_type",
            "sequence",
            "prompt_text",
            "expected_answer",
            "accepted_answers",
            "difficulty",
            "points",
            "is_mastery_item",
            "is_active",
        ],
        "counts": {
            "hear_and_repeat": 100,
            "see_letter_say_sound": 100,
            "match_sound_to_letter": 100,
            "sound_drill": 100,
            "mastery_check": 100,
        },
    },
    "module2_word_reading_activities.csv": {
        "module_key": "module_2",
        "header": [
            "id",
            "module_key",
            "activity_type",
            "sequence",
            "prompt_text",
            "target_word",
            "expected_answer",
            "accepted_answers",
            "word_family",
            "difficulty",
            "points",
            "is_mastery_item",
            "is_active",
        ],
        "counts": {
            "read_word": 100,
            "word_family_drill": 100,
            "minimal_pair": 100,
            "word_accuracy_challenge": 100,
            "mastery_check": 100,
        },
    },
    "module3_sentence_fluency_activities.csv": {
        "module_key": "module_3",
        "header": [
            "id",
            "module_key",
            "activity_type",
            "sequence",
            "prompt_text",
            "expected_answer",
            "accepted_answers",
            "difficulty",
            "points",
            "is_mastery_item",
            "is_active",
        ],
        "counts": {
            "read_sentence": 80,
            "read_with_coach": 80,
            "timed_sentence_reading": 80,
            "pause_practice": 80,
            "fluency_challenge": 80,
            "mastery_check": 100,
        },
    },
}


def load_csv(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"{path.relative_to(ROOT)} has columns {reader.fieldnames}, "
                f"expected {expected_header}"
            )
        return list(reader)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_file(filename: str, spec: dict[str, object], errors: list[str]) -> None:
    rows = load_csv(MODULES / filename, spec["header"])  # type: ignore[arg-type]
    expected_counts = spec["counts"]  # type: ignore[assignment]
    module_key = spec["module_key"]

    require(len(rows) == 500, f"{filename} has {len(rows)} rows, expected 500", errors)

    ids = [row["id"] for row in rows]
    duplicate_ids = [item_id for item_id, count in Counter(ids).items() if count > 1]
    require(not duplicate_ids, f"{filename} has duplicate IDs: {duplicate_ids}", errors)

    sequences = []
    for row in rows:
        try:
            sequences.append(int(row["sequence"]))
        except ValueError:
            errors.append(f"{filename} row {row['id']} has nonnumeric sequence {row['sequence']}")

    require(sorted(sequences) == list(range(1, 501)), f"{filename} sequences must be 1-500", errors)

    actual_counts = Counter(row["activity_type"] for row in rows)
    require(
        actual_counts == Counter(expected_counts),
        f"{filename} activity counts are {dict(actual_counts)}, expected {expected_counts}",
        errors,
    )

    for row in rows:
        row_id = row["id"]
        activity_type = row["activity_type"]
        accepted = [answer.strip().casefold() for answer in row["accepted_answers"].split("|")]
        expected = row["expected_answer"].strip().casefold()

        require(row["module_key"] == module_key, f"{filename} {row_id} has wrong module_key", errors)
        require(activity_type in expected_counts, f"{filename} {row_id} has unknown activity_type", errors)
        require(bool(row["prompt_text"].strip()), f"{filename} {row_id} has blank prompt_text", errors)
        require(bool(row["expected_answer"].strip()), f"{filename} {row_id} has blank expected_answer", errors)
        require(expected in accepted, f"{filename} {row_id} accepted_answers omit expected_answer", errors)
        require(row["points"] == "1", f"{filename} {row_id} points must be 1", errors)
        require(row["is_active"] == "1", f"{filename} {row_id} must be active", errors)
        require(
            row["is_mastery_item"] == ("1" if activity_type == "mastery_check" else "0"),
            f"{filename} {row_id} has incorrect is_mastery_item",
            errors,
        )

        if filename.startswith("module2"):
            require(bool(row["target_word"].strip()), f"{filename} {row_id} has blank target_word", errors)
            require(bool(row["word_family"].strip()), f"{filename} {row_id} has blank word_family", errors)
            require(
                row["target_word"].strip().casefold() == expected,
                f"{filename} {row_id} target_word must match expected_answer",
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

    print("Validation passed: each module CSV has 500 rows with valid segments and answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
