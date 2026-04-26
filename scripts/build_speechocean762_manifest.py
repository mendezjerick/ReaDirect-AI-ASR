from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.dataset.manifest import save_manifest
from readirect_asr.dataset.validation import validate_manifest_frame
from readirect_asr.datasets.speechocean762 import Speechocean762Loader
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def build_manifest(dataset_dir: Path, cmudict_dir: Path, output: Path, limit: int | None = None):
    cmu = CMUDictLoader(
        cmudict_dir / "cmudict.dict",
        cmudict_dir / "cmudict.phones",
        cmudict_dir / "cmudict.symbols",
    ).load()
    loader = Speechocean762Loader(dataset_dir, cmu)
    df = loader.to_manifest(limit=limit)
    save_manifest(df, output)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Speechocean762 unified manifest.")
    parser.add_argument("--dataset-dir", default="external_datasets/speechocean762/extracted", type=Path)
    parser.add_argument("--cmudict-dir", default="external_datasets/cmudict", type=Path)
    parser.add_argument("--output", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(args.dataset_dir, args.cmudict_dir, args.output, args.limit)
    total = len(df)
    phoneme_count = int(df["expected_phonemes"].fillna("").astype(str).str.strip().ne("").sum()) if total else 0
    print(f"Total rows: {total}")
    print(f"Audio found: {int(df['audio_path'].fillna('').astype(str).str.strip().ne('').sum()) if total else 0}")
    print(f"Transcripts found: {int(df['manual_transcript'].fillna('').astype(str).str.strip().ne('').sum()) if total else 0}")
    print(f"Sentence scores found: {int(df['sentence_score'].fillna('').astype(str).str.strip().ne('').sum()) if total else 0}")
    print(f"Word scores found: {int(df['word_score'].fillna('').astype(str).str.strip().ne('').sum()) if total else 0}")
    print(f"Phoneme scores found: {int(df['phoneme_score'].fillna('').astype(str).str.strip().ne('').sum()) if total else 0}")
    print(f"Missing audio count: {int(df['row_status'].fillna('').astype(str).str.contains('missing_audio').sum()) if total else 0}")
    print(f"Missing transcript count: {int(df['row_status'].fillna('').astype(str).str.contains('missing_transcript').sum()) if total else 0}")
    print(f"Phoneme enrichment success rate: {round((phoneme_count / total) * 100, 2) if total else 0.0}%")
    print(f"Output path: {args.output}")
    if args.validate:
        report = validate_manifest_frame(df, ".")
        print(f"Missing columns: {len(report['missing_columns'])}")
        print(f"Missing audio files: {len(report['missing_audio_files'])}")


if __name__ == "__main__":
    main()

