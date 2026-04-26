from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.finetuning.dataset_prep import prepare_whisper_dataset
from readirect_asr.finetuning.readiness import check_finetuning_readiness
from readirect_asr.finetuning.split import create_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Whisper fine-tuning JSONL files from a manifest.")
    parser.add_argument("--manifest", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--baseline", default=None, type=Path)
    parser.add_argument("--output-dir", default="data/processed/whisper_finetune", type=Path)
    parser.add_argument("--transcript-col", default="manual_transcript")
    parser.add_argument("--audio-col", default="audio_path")
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    baseline = pd.read_csv(args.baseline) if args.baseline and args.baseline.exists() else None
    readiness = check_finetuning_readiness(manifest, baseline)
    split_df = create_splits(manifest, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
    summary = prepare_whisper_dataset(
        split_df,
        args.output_dir,
        audio_col=args.audio_col,
        transcript_col=args.transcript_col,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dry_run=args.dry_run,
    )
    print(f"Readiness status: {readiness['status']}")
    print(f"Train rows: {summary['counts']['train']}")
    print(f"Validation rows: {summary['counts']['validation']}")
    print(f"Test rows: {summary['counts']['test']}")
    print(f"Total hours: {summary['total_hours']}")
    print(f"Skipped rows: {summary['skipped']}")
    print(f"Output dir: {summary['output_dir']}")
    if args.dry_run:
        print("Dry run only; no JSONL files were written.")


if __name__ == "__main__":
    main()
