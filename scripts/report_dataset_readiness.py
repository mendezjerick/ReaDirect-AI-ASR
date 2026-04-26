from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _availability(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().ne("").sum())


def build_report(df: pd.DataFrame, dataset_name: str) -> str:
    total = len(df)
    duration = pd.to_numeric(df.get("duration_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    missing_audio = int(df.get("row_status", pd.Series(dtype=str)).fillna("").astype(str).str.contains("missing_audio").sum())
    parsing_warnings = int(df.get("row_status", pd.Series(dtype=str)).fillna("").astype(str).str.contains("parse_warning").sum())
    phoneme_coverage = _availability(df, "expected_phonemes")

    lines = [
        f"# {dataset_name} Readiness Report",
        "",
        f"- Total recordings: {total}",
        f"- Total duration seconds: {round(float(duration), 3)}",
        f"- Transcript availability: {_availability(df, 'manual_transcript')} / {total}",
        f"- Sentence score availability: {_availability(df, 'sentence_score')} / {total}",
        f"- Word score availability: {_availability(df, 'word_score')} / {total}",
        f"- Phoneme score availability: {_availability(df, 'phoneme_score')} / {total}",
        f"- CMUdict phoneme coverage: {phoneme_coverage} / {total}",
        f"- Missing audio count: {missing_audio}",
        f"- Parsing warnings: {parsing_warnings}",
        "",
        "## Speaker Type Distribution",
        df.get("speaker_type", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_string(),
        "",
        "## Age Group Distribution",
        df.get("age_group", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_string(),
        "",
        "## Recommended Next Steps",
        "- Review license notes before any deployable training or evaluation.",
        "- Run baseline ASR in AI Phase 4.",
        "- Compute WER/CER and pronunciation-relevant score correlations.",
        "- Inspect rows with missing transcripts, missing audio, or parse warnings.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dataset readiness report.")
    parser.add_argument("--manifest", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--output", default="reports/speechocean762_readiness.md", type=Path)
    parser.add_argument("--dataset-name", default="Speechocean762")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    df = pd.read_csv(args.manifest)
    report = build_report(df, args.dataset_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote readiness report to {args.output}")


if __name__ == "__main__":
    main()

