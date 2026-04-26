from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _value_counts(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "Not available."
    return df[column].fillna("").astype(str).value_counts().to_string()


def _examples(df: pd.DataFrame, error_type: str, limit: int = 5) -> str:
    if "analysis_error_type" not in df.columns:
        return "Not available."
    subset = df[df["analysis_error_type"] == error_type].head(limit)
    if subset.empty:
        return "None."
    columns = [column for column in ["recording_id", "expected_text", "manual_transcript", "asr_transcript", "normalized_transcript", "analysis_error_type"] if column in subset.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in subset[columns].iterrows():
        rows.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(rows)


def build_report(df: pd.DataFrame) -> str:
    correct = int(df.get("analysis_is_correct", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    total = len(df)
    target_counts = _value_counts(df, "analysis_target_phoneme")
    report = [
        "# Reading Analysis Summary",
        "",
        f"- Total rows: {total}",
        f"- Correct count: {correct}",
        f"- Incorrect count: {total - correct}",
        "",
        "## Similarity Label Distribution",
        _value_counts(df, "analysis_similarity_label"),
        "",
        "## Error Type Distribution",
        _value_counts(df, "analysis_error_type"),
        "",
        "## Skill Signal Distribution",
        _value_counts(df, "analysis_skill_signal"),
        "",
        "## Most Common Target Phonemes",
        target_counts,
        "",
        "## Examples By Error Type",
    ]
    for error_type in ["final_sound_error", "initial_sound_error", "vowel_error", "skipped_word", "partial_sentence", "far_answer", "blank"]:
        report.extend([f"### {error_type}", _examples(df, error_type), ""])
    very_close = df[(df.get("analysis_similarity_label", "") == "very_close") & (~df.get("analysis_is_correct", False).astype(bool))] if "analysis_similarity_label" in df.columns else pd.DataFrame()
    blank = df[df.get("analysis_error_type", "") == "blank"] if "analysis_error_type" in df.columns else pd.DataFrame()
    report.extend(
        [
            "## Very Close But Not Exact",
            _examples(very_close, very_close["analysis_error_type"].iloc[0]) if not very_close.empty else "None.",
            "",
            "## Blank Or Unclear",
            _examples(blank, "blank") if not blank.empty else "None.",
            "",
            "## Limitations",
            "- Actual phonemes are derived from the ASR transcript, not acoustic phoneme recognition.",
            "- ASR mistakes can affect error detection.",
            "- This is a heuristic analysis layer, not a trained pronunciation model.",
        ]
    )
    return "\n".join(report) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report ReaDirect reading-analysis outputs.")
    parser.add_argument("--input", default="data/manifests/speechocean762_reading_analysis.csv", type=Path)
    parser.add_argument("--output", default="reports/reading_analysis_summary.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    df = pd.read_csv(args.input)
    report = build_report(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote reading analysis report to {args.output}")


if __name__ == "__main__":
    main()

