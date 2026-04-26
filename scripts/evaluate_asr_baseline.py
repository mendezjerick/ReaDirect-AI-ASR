from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer, evaluate_asr_dataframe, exact_match, token_accuracy
from readirect_asr.evaluation.error_analysis import (
    categorize_asr_error,
    summarize_by_age_group,
    summarize_by_score_bucket,
    summarize_by_speaker_type,
    summarize_by_text_length,
    summarize_common_substitutions,
    summarize_short_word_accuracy,
)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "None."
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(rows)


def choose_reference_column(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    if "manual_transcript" in df.columns and df["manual_transcript"].fillna("").astype(str).str.strip().ne("").any():
        return "manual_transcript"
    return "expected_text"


def evaluate_file(
    input_path: Path,
    output_path: Path,
    metrics_csv: Path,
    reference_col: str = "auto",
    hypothesis_col: str = "normalized_transcript",
    group_by: list[str] | None = None,
    max_examples: int = 25,
) -> dict[str, object]:
    df = pd.read_csv(input_path)
    ref_col = choose_reference_column(df, reference_col)
    metrics = evaluate_asr_dataframe(df, ref_col, hypothesis_col)

    per_row = []
    for _, row in df.iterrows():
        ref = str(row.get(ref_col, ""))
        hyp = str(row.get(hypothesis_col, ""))
        per_row.append(
            {
                "recording_id": row.get("recording_id", ""),
                "reference": ref,
                "hypothesis": hyp,
                "wer": compute_wer(ref, hyp),
                "cer": compute_cer(ref, hyp),
                "exact_match": exact_match(ref, hyp),
                "token_accuracy": token_accuracy(ref, hyp),
                "error_category": categorize_asr_error(ref, hyp),
            }
        )
    per_row_df = pd.DataFrame(per_row)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    per_row_df.to_csv(metrics_csv, index=False)

    group_lines: list[str] = []
    for column in group_by or []:
        if column not in df.columns:
            continue
        group_lines.append(f"### {column}")
        for value, group in df.groupby(column, dropna=False):
            group_metrics = evaluate_asr_dataframe(group, ref_col, hypothesis_col)
            group_lines.append(f"- `{value}`: WER={group_metrics['wer']}, CER={group_metrics['cer']}, exact={group_metrics['exact_match_rate']}, rows={group_metrics['evaluated_rows']}")

    exact_examples = per_row_df[per_row_df["exact_match"]].head(max_examples)
    high_cer = per_row_df.sort_values("cer", ascending=False).head(max_examples)
    blanks = per_row_df[per_row_df["hypothesis"].fillna("").astype(str).str.strip().eq("")].head(max_examples)

    report = [
        "# ASR Baseline Summary",
        "",
        f"- Reference column: `{ref_col}`",
        f"- Hypothesis column: `{hypothesis_col}`",
        f"- Total rows: {metrics['total_rows']}",
        f"- Evaluated rows: {metrics['evaluated_rows']}",
        f"- Skipped rows: {metrics['skipped_rows']}",
        f"- WER: {metrics['wer']}",
        f"- CER: {metrics['cer']}",
        f"- Exact match rate: {metrics['exact_match_rate']}",
        f"- Average token accuracy: {metrics['average_token_accuracy']}",
        f"- Blank references: {metrics['blank_reference_count']}",
        f"- Blank hypotheses: {metrics['blank_hypothesis_count']}",
        "",
        "## Pronunciation-Relevant Summaries",
        f"- Short-word summary: {summarize_short_word_accuracy(df, ref_col, hypothesis_col)}",
        f"- Text-length summary: {summarize_by_text_length(df, ref_col, hypothesis_col)}",
        f"- Speaker type distribution: {summarize_by_speaker_type(df)}",
        f"- Age group distribution: {summarize_by_age_group(df)}",
        f"- Score buckets: {summarize_by_score_bucket(df)}",
        f"- Common substitutions: {summarize_common_substitutions(df, ref_col, hypothesis_col, 15)}",
        "",
        "## Group Metrics",
        *(group_lines or ["No group-by columns requested or available."]),
        "",
        "## Exact Match Examples",
        markdown_table(exact_examples, ["recording_id", "reference", "hypothesis"]),
        "",
        "## High CER Examples",
        markdown_table(high_cer, ["recording_id", "reference", "hypothesis", "cer", "error_category"]),
        "",
        "## Blank ASR Outputs",
        markdown_table(blanks, ["recording_id", "reference", "hypothesis"]),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"WER: {metrics['wer']}")
    print(f"CER: {metrics['cer']}")
    print(f"Exact match rate: {metrics['exact_match_rate']}")
    print(f"Report path: {output_path}")
    print(f"Metrics CSV: {metrics_csv}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ASR baseline CSV.")
    parser.add_argument("--input", default="data/manifests/speechocean762_asr_baseline.csv", type=Path)
    parser.add_argument("--output", default="reports/asr_baseline_summary.md", type=Path)
    parser.add_argument("--metrics-csv", default="reports/asr_baseline_metrics.csv", type=Path)
    parser.add_argument("--reference-col", default="auto")
    parser.add_argument("--hypothesis-col", default="normalized_transcript")
    parser.add_argument("--group-by", default="dataset_source,speaker_type,age_group,prompt_type")
    parser.add_argument("--max-examples", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_by = [item.strip() for item in args.group_by.split(",") if item.strip()] if args.group_by else []
    evaluate_file(args.input, args.output, args.metrics_csv, args.reference_col, args.hypothesis_col, group_by, args.max_examples)


if __name__ == "__main__":
    main()
