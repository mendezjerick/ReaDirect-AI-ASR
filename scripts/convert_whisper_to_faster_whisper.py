from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face Whisper model to CTranslate2 format.")
    parser.add_argument("--model-dir", default="model_artifacts/readirect-whisper-base-en-v1-hf", type=Path)
    parser.add_argument("--output-dir", default="model_artifacts/readirect-whisper-base-en-v1-ct2", type=Path)
    parser.add_argument("--quantization", default="int8_float16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        "ct2-transformers-converter",
        "--model",
        str(args.model_dir),
        "--output_dir",
        str(args.output_dir),
        "--quantization",
        args.quantization,
    ]
    print("Conversion command:")
    print(" ".join(command))
    converter = shutil.which("ct2-transformers-converter")
    if args.dry_run:
        if not args.model_dir.exists():
            print(f"Model directory not found yet: {args.model_dir}")
        if not converter:
            print("ct2-transformers-converter is not installed; dry-run only is still valid.")
        print("Dry run complete. No conversion started.")
        return 0
    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}")
        return 2
    if not converter:
        print("ct2-transformers-converter not found. Install ctranslate2 before conversion.")
        return 2
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    print(f"Converted faster-whisper model path: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
