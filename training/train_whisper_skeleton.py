from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a future Whisper fine-tuning run.")
    parser.add_argument("--config", default="configs/whisper_finetune_config.yaml", type=Path)
    parser.add_argument("--run", action="store_true", help="Reserved for a future phase. Does not train yet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config)
    print("Whisper fine-tuning configuration preview")
    print(f"Config path: {args.config}")
    for key in sorted(config):
        print(f"{key}: {config[key]}")
    missing = [
        path
        for path in (config.get("train_jsonl"), config.get("validation_jsonl"))
        if path and not Path(path).exists()
    ]
    if missing:
        print(f"Missing dataset files: {missing}")
    if not args.run:
        print("No training started. Pass --run in a future implemented phase only.")
        return
    print("Training execution is not implemented in Phase 9. This guard prevents accidental GPU jobs.")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    main()
