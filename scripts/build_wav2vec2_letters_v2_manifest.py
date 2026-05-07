from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path, write_jsonl


READIRECT_ROOT = Path("external_datasets/readirect_letters")
SOURCES = {
    "readirect_letters": {
        "train": READIRECT_ROOT / "manifests/readirect_letters_train.jsonl",
        "valid": READIRECT_ROOT / "manifests/readirect_letters_valid.jsonl",
        "test": READIRECT_ROOT / "manifests/readirect_letters_test.jsonl",
    },
    "speechocean": {
        "train": Path("external_datasets/manifests/speechocean_train.jsonl"),
        "valid": Path("external_datasets/manifests/speechocean_valid.jsonl"),
        "test": Path("external_datasets/manifests/speechocean_test.jsonl"),
    },
    "librispeech": {
        "train": Path("external_datasets/manifests/librispeech_train_clean_100.jsonl"),
        "valid": Path("external_datasets/manifests/librispeech_dev_clean.jsonl"),
        "test": Path("external_datasets/manifests/librispeech_test_clean.jsonl"),
    },
}
OUTPUTS = {
    "train": Path("external_datasets/manifests/readirect_letters_v2_train_mixed.jsonl"),
    "valid": Path("external_datasets/manifests/readirect_letters_v2_valid_mixed.jsonl"),
    "test": Path("external_datasets/manifests/readirect_letters_v2_test_mixed.jsonl"),
}
REPORT_JSON = Path("outputs/training/wav2vec2_letters_v2_manifest_report.json")
REPORT_MD = Path("outputs/training/wav2vec2_letters_v2_manifest_report.md")
MIN_TRAIN_DURATION_SECONDS = 0.2
MAX_TRAIN_DURATION_SECONDS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 50/30/20 v2 Wav2Vec2 letter-focused manifests.")
    parser.add_argument("--letters-ratio", type=float, default=0.50)
    parser.add_argument("--speechocean-ratio", type=float, default=0.30)
    parser.add_argument("--librispeech-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total-train-rows", type=int, default=None)
    parser.add_argument("--allow-oversample", action="store_true")
    parser.add_argument("--no-oversample", dest="allow_oversample", action="store_false")
    parser.set_defaults(allow_oversample=False)
    return parser.parse_args()


def infer_prompt_type(text: str) -> str:
    normalized = text.strip()
    if re.fullmatch(r"[A-Z]", normalized):
        return "letter"
    if len(normalized.split()) == 1:
        return "word"
    return "sentence"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve_audio(row: dict[str, Any], dataset: str) -> Path:
    raw = Path(str(row.get("audio_path", "")).strip())
    if raw.is_absolute():
        return raw
    if dataset == "readirect_letters":
        candidate = resolve_repo_path(READIRECT_ROOT / raw)
        if candidate.exists():
            return candidate
    return resolve_repo_path(raw)


def normalize_row(row: dict[str, Any], dataset: str, split: str) -> tuple[dict[str, Any] | None, str | None]:
    text = str(row.get("text") or row.get("letter") or "").strip()
    if dataset == "readirect_letters":
        text = text.upper()
        if not re.fullmatch(r"[A-Z]", text):
            return None, f"Invalid ReaDirect letter label: {text!r}"
    elif not text:
        return None, f"Missing text for {dataset}"

    duration = row.get("duration_seconds")
    if split == "train" and duration is not None:
        try:
            duration_float = float(duration)
        except (TypeError, ValueError):
            duration_float = None
        if duration_float is not None and duration_float < MIN_TRAIN_DURATION_SECONDS:
            return None, f"Training row shorter than {MIN_TRAIN_DURATION_SECONDS}s: {duration_float}"
        if duration_float is not None and duration_float > MAX_TRAIN_DURATION_SECONDS:
            return None, f"Training row longer than {MAX_TRAIN_DURATION_SECONDS}s: {duration_float}"

    audio = resolve_audio(row, dataset)
    if not audio.exists():
        return None, f"Missing audio file: {row.get('audio_path')}"

    metadata = dict(row.get("metadata") or {})
    if dataset == "readirect_letters":
        for key in (
            "speaker_code",
            "voice_group",
            "letter",
            "repetition",
            "original_audio_path",
            "rms",
            "rms_dbfs",
            "peak_amplitude",
            "clipped_sample_ratio",
            "silence_ratio",
            "speech_ratio",
            "quality_flags",
            "quality_status",
            "notes",
        ):
            if key in row and key not in metadata:
                metadata[key] = row[key]

    normalized = {
        "audio_path": repo_relative(audio),
        "text": text,
        "dataset": dataset,
        "split": split,
        "prompt_type": str(row.get("prompt_type") or infer_prompt_type(text)),
        "speaker_id": str(row.get("speaker_id", "")),
        "source_id": str(row.get("source_id") or row.get("original_audio_path") or row.get("audio_path") or ""),
        "duration_seconds": duration,
        "metadata": metadata,
    }
    if "sample_rate" in row:
        normalized["sample_rate"] = row.get("sample_rate")
    return normalized, None


def load_normalized(dataset: str, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    path = SOURCES[dataset][split]
    rows = read_jsonl(path)
    normalized_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in rows:
        normalized, reason = normalize_row(row, dataset, split)
        if normalized is None:
            skipped.append(f"{path}: {reason}")
            continue
        normalized_rows.append(normalized)
    return normalized_rows, skipped


def sample_without_replacement(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, count)


def sample_with_replacement(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    if count <= len(rows):
        return rng.sample(rows, count)
    sampled = list(rows)
    while len(sampled) < count:
        item = dict(rng.choice(rows))
        metadata = dict(item.get("metadata") or {})
        metadata["oversampled"] = True
        item["metadata"] = metadata
        sampled.append(item)
    rng.shuffle(sampled)
    return sampled[:count]


def target_counts(source_counts: dict[str, int], ratios: dict[str, float], max_total: int | None, allow_oversample: bool) -> tuple[dict[str, int], list[str]]:
    warnings: list[str] = []
    total_ratio = sum(ratios.values())
    ratios = {key: value / total_ratio for key, value in ratios.items()}
    if allow_oversample and max_total is not None:
        total = max_total
    else:
        total = int(min(source_counts[key] / ratios[key] for key in ratios if ratios[key] > 0))
        if max_total is not None:
            total = min(total, max_total)
        if allow_oversample and max_total is None:
            warnings.append("allow-oversample was set without --max-total-train-rows; using non-oversampled maximum.")

    counts = {key: int(round(total * ratios[key])) for key in ratios}
    diff = total - sum(counts.values())
    if diff:
        largest = max(ratios, key=ratios.get)
        counts[largest] += diff
    if not allow_oversample:
        for key, count in list(counts.items()):
            if count > source_counts[key]:
                warnings.append(f"Requested {count} {key} rows but only {source_counts[key]} are available; reducing.")
                counts[key] = source_counts[key]
    return counts, warnings


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def metadata_distribution(rows: list[dict[str, Any]], key: str, dataset: str | None = None) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if dataset and row.get("dataset") != dataset:
            continue
        metadata = row.get("metadata") or {}
        counter[str(metadata.get(key, "unknown"))] += 1
    return dict(sorted(counter.items()))


def ratio_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row.get("dataset", "unknown")) for row in rows)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: round(value / total, 6) for key, value in sorted(counts.items())}


def write_reports(report: dict[str, Any]) -> None:
    json_path = resolve_repo_path(REPORT_JSON)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Wav2Vec2 Letters v2 Manifest Report",
        "",
        "This report describes manifest construction only. It does not include training or evaluation metrics.",
        "",
        "## Source Manifests",
    ]
    for dataset, splits in report["source_manifests"].items():
        for split, path in splits.items():
            lines.append(f"- {dataset} {split}: `{path}`")
    lines.extend(
        [
            "",
            "## Row Counts",
            "",
            f"- Train rows: {report['final_counts']['train']}",
            f"- Valid rows: {report['final_counts']['valid']}",
            f"- Test rows: {report['final_counts']['test']}",
            "",
            "## Actual Train Ratios",
        ]
    )
    for dataset, ratio in report["final_train_ratios"].items():
        lines.append(f"- {dataset}: {ratio:.4f}")
    lines.extend(["", "## Train Dataset Distribution"])
    for dataset, count in report["train_dataset_distribution"].items():
        lines.append(f"- {dataset}: {count}")
    lines.extend(["", "## Train Prompt Type Distribution"])
    for prompt_type, count in report["train_prompt_type_distribution"].items():
        lines.append(f"- {prompt_type}: {count}")
    lines.extend(["", "## ReaDirect Voice Group Distribution"])
    for voice_group, count in report["readirect_letters_voice_group_distribution"].items():
        lines.append(f"- {voice_group}: {count}")
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    if report["skipped_rows"]:
        lines.extend(["", "## Skipped Rows"])
        for skipped in report["skipped_rows"][:100]:
            lines.append(f"- {skipped}")
        if len(report["skipped_rows"]) > 100:
            lines.append(f"- ... {len(report['skipped_rows']) - 100} additional skipped rows")

    md_path = resolve_repo_path(REPORT_MD)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ratios = {
        "readirect_letters": args.letters_ratio,
        "speechocean": args.speechocean_ratio,
        "librispeech": args.librispeech_ratio,
    }
    warnings: list[str] = []
    skipped_rows: list[str] = []

    train_sources: dict[str, list[dict[str, Any]]] = {}
    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    for dataset in SOURCES:
        source_counts[dataset] = {}
        for split in ("train", "valid", "test"):
            rows, skipped = load_normalized(dataset, split)
            source_counts[dataset][split] = len(rows)
            skipped_rows.extend(skipped)
            if split == "train":
                train_sources[dataset] = rows
            elif split == "valid":
                valid_rows.extend(rows)
            else:
                test_rows.extend(rows)

    counts, count_warnings = target_counts(
        {dataset: len(rows) for dataset, rows in train_sources.items()},
        ratios,
        args.max_total_train_rows,
        args.allow_oversample,
    )
    warnings.extend(count_warnings)

    train_rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(("readirect_letters", "speechocean", "librispeech")):
        source_rows = train_sources[dataset]
        count = counts[dataset]
        if args.allow_oversample:
            selected = sample_with_replacement(source_rows, count, args.seed + index)
        else:
            selected = sample_without_replacement(source_rows, count, args.seed + index)
        train_rows.extend(selected)

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    rng.shuffle(test_rows)

    train_count = write_jsonl(OUTPUTS["train"], train_rows)
    valid_count = write_jsonl(OUTPUTS["valid"], valid_rows)
    test_count = write_jsonl(OUTPUTS["test"], test_rows)

    report = {
        "source_manifests": {
            dataset: {split: str(path).replace("\\", "/") for split, path in splits.items()}
            for dataset, splits in SOURCES.items()
        },
        "output_manifests": {split: str(path).replace("\\", "/") for split, path in OUTPUTS.items()},
        "source_row_counts": source_counts,
        "target_ratios": ratios,
        "allow_oversample": bool(args.allow_oversample),
        "max_total_train_rows": args.max_total_train_rows,
        "final_counts": {"train": train_count, "valid": valid_count, "test": test_count},
        "final_train_ratios": ratio_distribution(train_rows),
        "train_dataset_distribution": distribution(train_rows, "dataset"),
        "valid_dataset_distribution": distribution(valid_rows, "dataset"),
        "test_dataset_distribution": distribution(test_rows, "dataset"),
        "train_prompt_type_distribution": distribution(train_rows, "prompt_type"),
        "valid_prompt_type_distribution": distribution(valid_rows, "prompt_type"),
        "test_prompt_type_distribution": distribution(test_rows, "prompt_type"),
        "readirect_letters_voice_group_distribution": metadata_distribution(train_rows, "voice_group", "readirect_letters"),
        "skipped_rows": skipped_rows,
        "warnings": warnings,
    }
    write_reports(report)

    print(f"Wrote train manifest: {OUTPUTS['train']} ({train_count} rows)")
    print(f"Wrote valid manifest: {OUTPUTS['valid']} ({valid_count} rows)")
    print(f"Wrote test manifest: {OUTPUTS['test']} ({test_count} rows)")
    print(f"Actual train ratios: {report['final_train_ratios']}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if skipped_rows:
        print(f"Skipped rows: {len(skipped_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
