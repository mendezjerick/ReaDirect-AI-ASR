from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the active public dataset manifest.")
    parser.add_argument("--speechocean-manifest", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--output", default="data/manifests/unified_public_dataset_manifest.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    included = []
    if args.speechocean_manifest.exists():
        frames.append(pd.read_csv(args.speechocean_manifest))
        included.append("speechocean762")
    output_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Active datasets included: {', '.join(included) if included else 'none'}")
    print("Research-only datasets excluded: l2_arctic")
    print("Future optional datasets excluded: pf_star")
    print(f"Total rows: {len(output_df)}")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()

