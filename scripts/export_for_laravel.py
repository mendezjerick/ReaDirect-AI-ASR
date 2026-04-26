from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.dataset.manifest import load_manifest


EXPORT_COLUMNS = [
    "recording_id",
    "prompt_id",
    "expected_text",
    "asr_transcript",
    "human_correct",
    "error_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export non-private ASR result fields for Laravel review.")
    parser.add_argument("--manifest", default="data/manifests/dataset_manifest.csv", type=Path)
    parser.add_argument("--output", default="reports/laravel_asr_export.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_manifest(args.manifest)
    available_columns = [column for column in EXPORT_COLUMNS if column in df.columns]
    export_df = df[available_columns].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(args.output, index=False)
    print(f"Wrote {len(export_df)} rows to {args.output}")
    print("Review exports for privacy before sharing or committing.")


if __name__ == "__main__":
    main()
