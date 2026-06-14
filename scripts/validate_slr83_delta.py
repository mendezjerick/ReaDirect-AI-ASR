from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_delta_data import (
    load_delta_config,
    load_slr83_dataset,
    validate_normalized_slr83,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SLR83 parsing for Delta.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_delta.yaml"))
    parser.add_argument("--decode-sample", action="store_true")
    args = parser.parse_args()
    configure_windows_ffmpeg()
    config = load_delta_config(args.config)

    from transformers import Wav2Vec2Processor

    beta_path = resolve_repo_path(config["model"]["beta_checkpoint_path"])
    processor = Wav2Vec2Processor.from_pretrained(str(beta_path), local_files_only=True)
    summary = validate_normalized_slr83(config, set(processor.tokenizer.get_vocab()))
    if args.decode_sample:
        dataset = load_slr83_dataset(config)
        audio = dataset[0]["audio"]
        summary["decoded_sample"] = {
            "samples": len(audio["array"]),
            "sample_rate": audio["sampling_rate"],
        }
    print(json.dumps(summary, indent=2))
    print("SLR83 parsing validation passed. Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

