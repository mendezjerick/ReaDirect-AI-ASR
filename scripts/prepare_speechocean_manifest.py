from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.text_normalization import load_tokenizer_vocab, normalize_asr_text
from training.wav2vec2_manifest_utils import (
    iter_audio_files,
    make_manifest_row,
    split_train_valid,
    write_jsonl,
)


OUTPUTS = {
    "train": "external_datasets/manifests/speechocean_train.jsonl",
    "valid": "external_datasets/manifests/speechocean_valid.jsonl",
    "test": "external_datasets/manifests/speechocean_test.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SpeechOcean JSONL manifests for Wav2Vec2 fine-tuning.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("models/wav2vec2-base-960h"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--min-duration-seconds", type=float, default=0.2)
    return parser.parse_args()


def candidate_roots(args_root: Path | None) -> list[Path]:
    if args_root:
        return [args_root]
    return [
        PROJECT_ROOT / "external_datasets/speechocean/extracted",
        PROJECT_ROOT / "external_datasets/speechocean762/extracted",
    ]


def find_root(args_root: Path | None) -> Path | None:
    for root in candidate_roots(args_root):
        if root.exists():
            return root
    return None


def find_speechocean762_root(root: Path) -> Path | None:
    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if (candidate / "train.json").exists() and (candidate / "test.json").exists():
            return candidate
    return None


def rows_from_speechocean762(root: Path, vocab: set[str] | None, min_duration: float, valid_ratio: float) -> dict[str, list[dict[str, Any]]]:
    from readirect_asr.datasets.speechocean762 import Speechocean762Loader

    loader = Speechocean762Loader(root)
    df = loader.to_manifest()
    rows_by_split = {"train": [], "valid": [], "test": []}
    raw_train: list[dict[str, Any]] = []
    for record in df.fillna("").to_dict(orient="records"):
        original_text = str(record.get("manual_transcript") or record.get("expected_text") or record.get("prompt_text") or "").strip()
        text = normalize_asr_text(original_text, vocab)
        audio_path = str(record.get("audio_path", "")).strip()
        if not audio_path or not text:
            continue
        row = make_manifest_row(
            audio_path=audio_path,
            text=text,
            dataset="speechocean",
            split="test" if str(record.get("split", "")).lower() == "test" else "train",
            speaker_id=str(record.get("speaker_id_anonymized", "")),
            source_id=str(record.get("recording_id", "")),
            metadata={
                "source": "speechocean762",
                "original_text": original_text,
                "sentence_score": record.get("sentence_score", ""),
                "word_score": record.get("word_score", ""),
                "phoneme_score": record.get("phoneme_score", ""),
                "word_labels": record.get("word_labels", ""),
                "phoneme_labels": record.get("phoneme_labels", ""),
                "license_notes": record.get("license_notes", ""),
            },
        )
        duration = row.get("duration_seconds")
        if duration is not None and float(duration) < min_duration:
            continue
        if row["split"] == "test":
            rows_by_split["test"].append(row)
        else:
            raw_train.append(row)
    rows_by_split["train"], rows_by_split["valid"] = split_train_valid(raw_train, valid_ratio=valid_ratio)
    for split, rows in rows_by_split.items():
        for row in rows:
            row["split"] = split
    return rows_by_split


def read_text_sidecars(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in root.rglob("*.txt"):
        if path.name.endswith(".trans.txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            continue
        if text:
            transcripts[path.stem] = text.splitlines()[0].strip()
    return transcripts


def read_csv_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    transcript_keys = ("text", "transcript", "sentence", "prompt", "expected_text")
    id_keys = ("id", "recording_id", "audio_id", "filename", "file", "audio")
    for path in root.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    text = next((str(row.get(key, "")).strip() for key in transcript_keys if row.get(key)), "")
                    raw_id = next((str(row.get(key, "")).strip() for key in id_keys if row.get(key)), "")
                    if text and raw_id:
                        transcripts[Path(raw_id).stem] = text
        except Exception:
            continue
    return transcripts


def rows_from_generic(root: Path, vocab: set[str] | None, min_duration: float, valid_ratio: float) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    report: list[str] = []
    transcripts = read_csv_transcripts(root)
    transcripts.update(read_text_sidecars(root))
    audio_files = list(iter_audio_files(root))
    rows: list[dict[str, Any]] = []
    for audio_path in audio_files:
        original_text = transcripts.get(audio_path.stem, "")
        text = normalize_asr_text(original_text, vocab)
        if not text:
            continue
        row = make_manifest_row(
            audio_path=audio_path,
            text=text,
            dataset="speechocean",
            split="train",
            speaker_id=audio_path.parent.name,
            source_id=audio_path.stem,
            metadata={"source": "generic_speechocean_scan", "original_text": original_text},
        )
        duration = row.get("duration_seconds")
        if duration is not None and float(duration) < min_duration:
            continue
        rows.append(row)
    train, valid = split_train_valid(rows, valid_ratio=valid_ratio)
    for row in train:
        row["split"] = "train"
    for row in valid:
        row["split"] = "valid"
    report.append(f"Generic scan found {len(audio_files)} audio files and {len(transcripts)} transcript candidates.")
    if not rows:
        report.append("No generic SpeechOcean rows could be mapped. Manual transcript mapping is required.")
    return {"train": train, "valid": valid, "test": []}, report


def write_report(lines: list[str]) -> None:
    output = PROJECT_ROOT / "outputs/training/speechocean_manifest_report.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote SpeechOcean report to {output.relative_to(PROJECT_ROOT)}")


def main() -> int:
    args = parse_args()
    root = find_root(args.root)
    report: list[str] = []
    if not root:
        report.append("SpeechOcean extracted folder was not found. LibriSpeech-only training can still proceed.")
        write_report(report)
        return 0
    vocab = load_tokenizer_vocab(PROJECT_ROOT / args.model_path)
    speechocean762_root = find_speechocean762_root(root)
    if speechocean762_root:
        report.append(f"Detected Speechocean762 layout at {speechocean762_root}")
        rows_by_split = rows_from_speechocean762(speechocean762_root, vocab, args.min_duration_seconds, args.valid_ratio)
    else:
        report.append(f"Known SpeechOcean layout not detected under {root}; using generic transcript scan.")
        rows_by_split, generic_report = rows_from_generic(root, vocab, args.min_duration_seconds, args.valid_ratio)
        report.extend(generic_report)

    for split, output_path in OUTPUTS.items():
        count = write_jsonl(output_path, rows_by_split.get(split, []))
        print(f"Wrote {count} {split} rows to {output_path}")
    report.append(json.dumps({split: len(rows) for split, rows in rows_by_split.items()}, indent=2))
    report.append("Phoneme/pronunciation metadata is preserved when present, but phoneme model fine-tuning is not enabled by this script.")
    write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
