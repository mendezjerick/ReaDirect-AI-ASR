#!/usr/bin/env python3
"""Validate ReaDirect reading passage and comprehension CSV banks."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PASSAGES = ROOT / "assessment" / "reading_passages.csv"
QUESTIONS = ROOT / "assessment" / "comprehension_questions.csv"

PASSAGE_COLUMNS = [
    "id",
    "title",
    "passage_text",
    "word_count",
    "expected_reading_time_seconds",
    "max_time_seconds",
    "difficulty",
    "is_active",
]

QUESTION_COLUMNS = [
    "id",
    "passage_id",
    "sequence",
    "question_text",
    "question_type",
    "correct_answer",
    "accepted_answers",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "difficulty",
    "points",
    "is_active",
]


def load_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path.relative_to(ROOT)} has columns {reader.fieldnames}, "
                f"expected {expected_columns}"
            )
        return list(reader)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    try:
        passages = load_csv(PASSAGES, PASSAGE_COLUMNS)
        questions = load_csv(QUESTIONS, QUESTION_COLUMNS)
    except Exception as exc:  # noqa: BLE001 - report CSV/schema failures plainly.
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    passage_ids = [row["id"] for row in passages]
    question_ids = [row["id"] for row in questions]

    for label, ids in (("passage", passage_ids), ("question", question_ids)):
        counts = Counter(ids)
        for item_id, count in counts.items():
            require(count == 1, f"Duplicate {label} id: {item_id}", errors)
        require(all(item_id.strip() for item_id in ids), f"Blank {label} id found", errors)

    passage_by_id = {row["id"]: row for row in passages}
    questions_by_passage: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in passages:
        text = row["passage_text"].strip()
        require(bool(text), f"{row['id']} has a blank passage", errors)
        try:
            declared_count = int(row["word_count"])
        except ValueError:
            errors.append(f"{row['id']} has nonnumeric word_count: {row['word_count']}")
            continue
        actual_count = word_count(text)
        require(
            actual_count == declared_count,
            f"{row['id']} word_count is {declared_count}, actual is {actual_count}",
            errors,
        )

    for row in questions:
        passage_id = row["passage_id"]
        require(
            passage_id in passage_by_id,
            f"{row['id']} links to missing passage_id {passage_id}",
            errors,
        )
        questions_by_passage[passage_id].append(row)

        choices = [row[f"choice_{letter}"] for letter in "abcd"]
        normalized_choices = [choice.strip().casefold() for choice in choices]
        correct = row["correct_answer"].strip().casefold()
        require(bool(row["question_text"].strip()), f"{row['id']} has blank question_text", errors)
        require(bool(row["correct_answer"].strip()), f"{row['id']} has blank correct_answer", errors)
        require(
            normalized_choices.count(correct) == 1,
            f"{row['id']} correct_answer must match exactly one choice",
            errors,
        )
        accepted = [answer.strip().casefold() for answer in row["accepted_answers"].split("|")]
        require(correct in accepted, f"{row['id']} accepted_answers omit correct_answer", errors)

    for passage_id in passage_ids:
        linked_questions = questions_by_passage.get(passage_id, [])
        require(
            len(linked_questions) == 5,
            f"{passage_id} has {len(linked_questions)} linked questions, expected 5",
            errors,
        )
        sequences = sorted(row["sequence"] for row in linked_questions)
        require(
            sequences == ["1", "2", "3", "4", "5"],
            f"{passage_id} question sequences are {sequences}, expected 1-5",
            errors,
        )

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Validation passed: {len(passages)} passages and "
        f"{len(questions)} questions are linked and parse correctly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
