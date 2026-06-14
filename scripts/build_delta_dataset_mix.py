from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_delta_data import load_delta_config, plan_delta_dataset_mix


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify Delta's effective dataset mix.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_delta.yaml"))
    args = parser.parse_args()
    configure_windows_ffmpeg()
    config = load_delta_config(args.config)
    summary = plan_delta_dataset_mix(config, write_summary=True)
    print(json.dumps(summary, indent=2))
    print("Delta dataset mix is ready. Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
