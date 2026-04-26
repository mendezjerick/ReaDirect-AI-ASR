from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.audio.preprocessing import list_audio_files
from readirect_asr.dataset.manifest import load_manifest
from readirect_asr.dataset.validation import missing_audio_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a ReaDirect ASR dataset manifest.")
    parser.add_argument("--manifest", default="data/manifests/dataset_manifest.csv", type=Path)
    parser.add_argument("--audio-base-path", default="data/raw", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        return

    df = load_manifest(args.manifest)
    audio_files = list_audio_files(args.audio_base_path)
    missing_audio = missing_audio_paths(df, args.audio_base_path)

    print(f"Manifest rows: {len(df)}")
    print(f"Audio files under {args.audio_base_path}: {len(audio_files)}")
    print(f"Missing audio paths: {len(missing_audio)}")

    if "prompt_type" in df.columns:
        print("\nPrompt type counts:")
        print(df["prompt_type"].fillna("").value_counts().to_string())

    if "error_type" in df.columns:
        print("\nError type counts:")
        print(df["error_type"].fillna("").value_counts().to_string())


if __name__ == "__main__":
    main()
