from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from readirect_asr.evaluation.model_comparison import infer_prompt_type, normalize_eval_text
from training.wav2vec2_manifest_utils import resolve_repo_path, write_jsonl


DEFAULT_INPUTS = [
    "external_datasets/manifests/readirect_valid_mixed.jsonl",
    "external_datasets/manifests/readirect_test_mixed.jsonl",
    "external_datasets/manifests/librispeech_dev_clean.jsonl",
    "external_datasets/manifests/librispeech_test_clean.jsonl",
    "external_datasets/manifests/speechocean_valid.jsonl",
    "external_datasets/manifests/speechocean_test.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real-audio letter/word evaluation manifest from existing manifests.")
    parser.add_argument("--output", default="external_datasets/manifests/readirect_letter_word_eval.jsonl", type=Path)
    parser.add_argument("--report", default="external_datasets/manifests/readirect_letter_word_eval_report.md", type=Path)
    parser.add_argument("--include-train", action="store_true", help="Include train manifests. Off by default to avoid evaluating on training rows.")
    parser.add_argument("--input", action="append", dest="inputs", default=None, help="Input JSONL manifest. Can be passed multiple times.")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return []
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_inputs(include_train: bool, inputs: list[str] | None) -> list[str]:
    if inputs:
        return inputs
    paths = list(DEFAULT_INPUTS)
    if include_train:
        paths.extend(
            [
                "external_datasets/manifests/readirect_train_mixed.jsonl",
                "external_datasets/manifests/librispeech_train_clean_100.jsonl",
                "external_datasets/manifests/speechocean_train.jsonl",
            ]
        )
    return paths


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("audio_path", "")),
        normalize_eval_text(row.get("text", "")),
        str(row.get("source_id", "")),
    )


def build_rows(inputs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    counts = Counter()
    source_counts = Counter()
    skipped_missing_audio = 0
    for input_path in inputs:
        for row in read_jsonl(input_path):
            prompt_type = str(row.get("prompt_type") or infer_prompt_type(row.get("text", "")))
            counts[f"seen_{prompt_type}"] += 1
            if prompt_type not in {"letter", "word"}:
                continue
            audio_path = str(row.get("audio_path", "")).strip()
            if not audio_path or not resolve_repo_path(audio_path).exists():
                skipped_missing_audio += 1
                continue
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            output_row = dict(row)
            output_row["prompt_type"] = prompt_type
            output_row.setdefault("metadata", {})
            if isinstance(output_row["metadata"], dict):
                output_row["metadata"]["letter_word_eval_source_manifest"] = input_path
            rows.append(output_row)
            source_counts[(prompt_type, str(row.get("dataset", "unknown")), str(row.get("split", "unknown")))] += 1
    report = {
        "input_manifests": inputs,
        "output_rows": len(rows),
        "prompt_counts": Counter(row["prompt_type"] for row in rows),
        "source_counts": source_counts,
        "seen_counts": counts,
        "skipped_missing_audio": skipped_missing_audio,
    }
    return rows, report


def write_report(path: Path, report: dict[str, Any]) -> None:
    output = resolve_repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Letter/Word Evaluation Manifest Report",
        "",
        f"- Output rows: {report['output_rows']}",
        f"- Letter rows: {report['prompt_counts'].get('letter', 0)}",
        f"- Word rows: {report['prompt_counts'].get('word', 0)}",
        f"- Skipped missing audio: {report['skipped_missing_audio']}",
        "",
        "## Input Manifests",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["input_manifests"])
    lines.extend(["", "## Source Counts", ""])
    for (prompt_type, dataset, split), count in sorted(report["source_counts"].items()):
        lines.append(f"- {prompt_type} / {dataset} / {split}: {count}")
    lines.extend(["", "## Coverage Notes", ""])
    if report["prompt_counts"].get("letter", 0) == 0:
        lines.append("- No real letter-audio rows were found in the available evaluation manifests.")
        lines.append("- Letter comparison still requires real recordings for L, Q, Z, B, X, C, G, V, T, D, M, and N.")
    if report["prompt_counts"].get("word", 0) < 25:
        lines.append("- Word coverage is very small, so word-only comparison should be treated as exploratory.")
    lines.append("- Training rows are excluded by default. Use `--include-train` only for debugging, not fair evaluation.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    inputs = selected_inputs(args.include_train, args.inputs)
    rows, report = build_rows(inputs)
    count = write_jsonl(args.output, rows)
    write_report(args.report, report)
    print(f"Wrote {count} rows to {args.output}")
    print(f"Letter rows: {report['prompt_counts'].get('letter', 0)}")
    print(f"Word rows: {report['prompt_counts'].get('word', 0)}")
    print(f"Report: {args.report}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
