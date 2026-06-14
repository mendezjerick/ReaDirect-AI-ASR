from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_epsilon_data import (
    load_epsilon_config,
    plan_epsilon_dataset_mix,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/check Epsilon's 45/30/15/10 mix.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_epsilon.yaml"))
    args = parser.parse_args()
    config = load_epsilon_config(args.config)

    from transformers import Wav2Vec2Processor

    delta_path = resolve_repo_path(config["model"]["delta_checkpoint_path"])
    processor = Wav2Vec2Processor.from_pretrained(str(delta_path), local_files_only=True)
    summary = plan_epsilon_dataset_mix(
        config,
        set(processor.tokenizer.get_vocab()),
        write_summary=True,
    )
    print(json.dumps(summary, indent=2))
    print("Epsilon dataset mix is ready. Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
