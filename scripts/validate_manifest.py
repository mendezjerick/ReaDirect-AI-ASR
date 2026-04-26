from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.dataset.manifest import load_manifest
from readirect_asr.dataset.validation import validate_manifest_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ReaDirect ASR manifest.")
    parser.add_argument("--manifest", default="data/manifests/dataset_manifest.csv", type=Path)
    parser.add_argument("--content-index", default=None, type=Path)
    parser.add_argument("--content-bank", default="content_bank", type=Path)
    parser.add_argument("--audio-base", "--audio-base-path", dest="audio_base", default="data/raw", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    df = load_manifest(args.manifest)
    content_index = pd.read_csv(args.content_index) if args.content_index and args.content_index.exists() else None
    report = validate_manifest_frame(df, args.audio_base, content_index)

    print("Manifest validation report")
    print(f"Rows: {report['summary']['total_recordings']}")
    print(f"Missing columns: {len(report['missing_columns'])}")
    for column in report["missing_columns"]:
        print(f"- missing column: {column}")
    print(f"Missing audio files: {len(report['missing_audio_files'])}")
    print(f"Unsupported audio files: {len(report['unsupported_audio_files'])}")
    print(f"Missing prompt IDs: {len(report['missing_prompt_ids'])}")
    print(f"Prompt IDs not found: {len(report['prompt_ids_not_found'])}")
    print(f"Rows missing expected_text: {report['summary']['rows_missing_expected_text']}")
    print(f"Rows missing manual_transcript: {report['summary']['rows_missing_manual_transcript']}")
    print(f"Total duration seconds: {report['summary']['total_duration_seconds']}")

    if report["missing_columns"] or report["prompt_ids_not_found"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
