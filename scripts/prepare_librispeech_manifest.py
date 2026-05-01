from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.text_normalization import load_tokenizer_vocab, normalize_asr_text
from training.wav2vec2_manifest_utils import iter_audio_files, make_manifest_row, write_jsonl


SPLITS = {
    "train-clean-100": ("train", "external_datasets/manifests/librispeech_train_clean_100.jsonl"),
    "dev-clean": ("valid", "external_datasets/manifests/librispeech_dev_clean.jsonl"),
    "test-clean": ("test", "external_datasets/manifests/librispeech_test_clean.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech JSONL manifests for Wav2Vec2 fine-tuning.")
    parser.add_argument("--root", type=Path, default=None, help="Path containing LibriSpeech split folders.")
    parser.add_argument("--model-path", type=Path, default=Path("models/wav2vec2-base-960h"))
    parser.add_argument("--min-duration-seconds", type=float, default=0.2)
    return parser.parse_args()


def candidate_roots(args_root: Path | None) -> list[Path]:
    if args_root:
        return [args_root]
    return [
        PROJECT_ROOT / "external_datasets/librispeech/extracted/LibriSpeech",
        PROJECT_ROOT / "external_datasets/LibriSpeech/extracted/LibriSpeech",
        PROJECT_ROOT / "external_datasets/LibriSpeech",
    ]


def find_split_root(split_name: str, args_root: Path | None) -> Path | None:
    for root in candidate_roots(args_root):
        candidate = root / split_name
        if candidate.exists():
            return candidate
    return None


def load_transcripts(split_root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for transcript_file in split_root.rglob("*.trans.txt"):
        with transcript_file.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                audio_id, _, text = stripped.partition(" ")
                if audio_id and text:
                    transcripts[audio_id] = text
    return transcripts


def build_split(split_name: str, split_label: str, args: argparse.Namespace, vocab: set[str] | None) -> list[dict[str, object]]:
    split_root = find_split_root(split_name, args.root)
    if not split_root:
        print(f"Warning: LibriSpeech split not found: {split_name}")
        return []
    transcripts = load_transcripts(split_root)
    rows: list[dict[str, object]] = []
    skipped_missing_text = 0
    skipped_too_short = 0
    for audio_path in iter_audio_files(split_root):
        if audio_path.suffix.lower() != ".flac":
            continue
        original_text = transcripts.get(audio_path.stem, "")
        text = normalize_asr_text(original_text, vocab)
        if not text:
            skipped_missing_text += 1
            continue
        row = make_manifest_row(
            audio_path=audio_path,
            text=text,
            dataset="librispeech",
            split=split_label,
            speaker_id="-".join(audio_path.stem.split("-")[:2]),
            source_id=audio_path.stem,
            metadata={"original_text": original_text, "split_name": split_name},
        )
        duration = row.get("duration_seconds")
        if duration is not None and float(duration) < args.min_duration_seconds:
            skipped_too_short += 1
            continue
        rows.append(row)
    print(f"{split_name}: {len(rows)} rows, skipped_missing_text={skipped_missing_text}, skipped_too_short={skipped_too_short}")
    return rows


def main() -> int:
    args = parse_args()
    vocab = load_tokenizer_vocab(PROJECT_ROOT / args.model_path)
    wrote_any = False
    for split_name, (split_label, output_path) in SPLITS.items():
        rows = build_split(split_name, split_label, args, vocab)
        if rows:
            count = write_jsonl(output_path, rows)
            print(f"Wrote {count} rows to {output_path}")
            wrote_any = True
    if not wrote_any:
        print("No LibriSpeech manifests were written. Check extraction paths.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

