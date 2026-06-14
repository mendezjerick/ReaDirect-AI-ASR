from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_shared_benchmark import (
    build_shared_benchmark,
    load_benchmark_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed five-source ASR benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wav2vec2_shared_benchmark.yaml"),
    )
    args = parser.parse_args()
    configure_windows_ffmpeg()
    config = load_benchmark_config(args.config)
    _, summary = build_shared_benchmark(config)
    printable = dict(summary)
    printable["sources"] = {
        name: {key: value for key, value in details.items() if key != "source_ids"}
        for name, details in summary["sources"].items()
    }
    print(json.dumps(printable, indent=2))
    print("Shared benchmark built. No model evaluation was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

