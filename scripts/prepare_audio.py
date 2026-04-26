from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.audio.preprocessing import describe_preprocessing_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview non-destructive audio preprocessing.")
    parser.add_argument("--input-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = describe_preprocessing_plan(args.input_dir, args.output_dir)
    if not plan:
        print(f"No supported audio files found in {args.input_dir}")
        return

    print("Phase AI-1 preview only. No audio files will be modified.")
    for line in plan:
        print(line)
    print("TODO: Add resampling, channel conversion, loudness normalization, and duration checks.")


if __name__ == "__main__":
    main()
