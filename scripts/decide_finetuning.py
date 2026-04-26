from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.evaluation.asr_metrics import evaluate_asr_dataframe
from readirect_asr.evaluation.readirect_metrics import evaluate_short_words
from readirect_asr.finetuning.decision_rules import decide_finetuning_need
from readirect_asr.finetuning.readiness import check_finetuning_readiness
from readirect_asr.finetuning.report import generate_decision_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether Whisper fine-tuning is justified.")
    parser.add_argument("--manifest", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--baseline", default="data/manifests/speechocean762_asr_baseline.csv", type=Path)
    parser.add_argument("--metrics-csv", default=None, type=Path)
    parser.add_argument("--output", default="reports/finetuning_decision.md", type=Path)
    parser.add_argument("--config", default="configs/finetuning_decision.yaml", type=Path)
    parser.add_argument("--reference-col", default="auto")
    parser.add_argument("--hypothesis-col", default="normalized_transcript")
    parser.add_argument("--cmudict-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    manifest = pd.read_csv(args.manifest) if args.manifest.exists() else pd.DataFrame()
    baseline = pd.read_csv(args.baseline) if args.baseline.exists() else None
    readiness = check_finetuning_readiness(
        manifest,
        baseline,
        min_total_hours=float(cfg.get("min_total_hours", 2.0)),
        min_rows=int(cfg.get("min_rows", 500)),
        min_transcript_coverage=float(cfg.get("min_transcript_coverage", 0.9)),
    )
    metrics = None
    short_metrics = None
    common = []
    if baseline is not None and not baseline.empty:
        ref_col = choose_reference_column(baseline, args.reference_col)
        hyp_col = args.hypothesis_col if args.hypothesis_col in baseline.columns else "asr_transcript"
        metrics = evaluate_asr_dataframe(baseline, ref_col, hyp_col)
        short_metrics = evaluate_short_words(baseline, ref_col, hyp_col)
        common = short_metrics.get("common_confusions", [])
    decision = decide_finetuning_need(metrics, readiness, short_metrics, cfg)
    report = generate_decision_report(readiness, metrics, short_metrics, decision, common)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Decision: {decision['decision']}")
    print(f"Confidence: {decision['confidence']}")
    print(f"Report path: {args.output}")


def choose_reference_column(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    if "manual_transcript" in df.columns and df["manual_transcript"].fillna("").astype(str).str.strip().ne("").any():
        return "manual_transcript"
    return "expected_text"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    main()
