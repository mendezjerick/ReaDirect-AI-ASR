from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_asr_baseline import evaluate_file
from scripts.run_asr_baseline import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Phase 4 ASR baseline sample.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model-size", default="base.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--manifest", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--output", default="data/manifests/speechocean762_asr_baseline_sample.csv", type=Path)
    parser.add_argument("--provider", default="faster_whisper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        print("Build it first with scripts/build_speechocean762_manifest.py")
        return
    run_baseline(
        manifest=args.manifest,
        output=args.output,
        provider_name=args.provider,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        limit=args.limit,
        save_every=max(1, args.limit),
    )
    evaluate_file(
        input_path=args.output,
        output_path=Path("reports/asr_baseline_sample_summary.md"),
        metrics_csv=Path("reports/asr_baseline_sample_metrics.csv"),
        reference_col="auto",
        hypothesis_col="normalized_transcript",
        group_by=["dataset_source", "speaker_type", "age_group", "prompt_type"],
        max_examples=10,
    )


if __name__ == "__main__":
    main()

