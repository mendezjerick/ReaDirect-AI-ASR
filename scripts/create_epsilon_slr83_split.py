from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_epsilon_data import load_epsilon_config, split_slr83_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Epsilon's deterministic SLR83 split.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_epsilon.yaml"))
    args = parser.parse_args()
    config = load_epsilon_config(args.config)
    _, _, summary = split_slr83_rows(config, write_manifest=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print("Saved reports/asr/epsilon/slr83_split_manifest.json. Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
