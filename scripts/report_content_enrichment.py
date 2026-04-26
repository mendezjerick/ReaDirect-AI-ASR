from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _counts(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "Not available."
    return df[column].fillna("").astype(str).value_counts().head(30).to_string()


def build_report(df: pd.DataFrame) -> str:
    total = len(df)
    match_rate = df["has_cmudict_match"].fillna(False).astype(str).str.lower().isin({"1", "true"}).mean() if "has_cmudict_match" in df.columns and total else 0
    needs_review = int(df["needs_manual_review"].fillna(False).astype(str).str.lower().isin({"1", "true", "yes"}).sum()) if "needs_manual_review" in df.columns else 0
    return "\n".join(
        [
            "# Content Enrichment Report",
            "",
            f"- Total items: {total}",
            f"- CMUdict match rate: {round(float(match_rate) * 100, 2)}%",
            f"- Items needing manual review: {needs_review}",
            "",
            "## By Source File",
            _counts(df, "source_file"),
            "",
            "## By Module",
            _counts(df, "module_key"),
            "",
            "## By Activity Type",
            _counts(df, "activity_type"),
            "",
            "## By Skill Group",
            _counts(df, "skill_group"),
            "",
            "## By Error Focus",
            _counts(df, "error_focus"),
            "",
            "## By Difficulty Level",
            _counts(df, "difficulty_level"),
            "",
            "## Top Word Families",
            _counts(df, "word_family"),
            "",
            "## Top Target Phonemes",
            _counts(df, "target_phoneme"),
            "",
            "## Adaptive Buckets",
            _counts(df, "adaptive_bucket"),
            "",
            "## Recommended Next Actions",
            "- Review rows where `needs_manual_review` is true.",
            "- Review missing CMUdict words and add manual tags where needed.",
            "- Spot-check difficulty levels with educators before importing into Laravel.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a content enrichment report.")
    parser.add_argument("--enriched-index", default="content_bank_enriched/enriched_content_index.csv", type=Path)
    parser.add_argument("--output", default="content_bank_enriched/reports/content_enrichment_report.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.enriched_index.exists():
        raise SystemExit(f"Input not found: {args.enriched_index}")
    report = build_report(pd.read_csv(args.enriched_index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

