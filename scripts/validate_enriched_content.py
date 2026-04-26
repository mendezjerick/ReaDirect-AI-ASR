from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.content.enrichment_validation import validate_enriched_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate enriched ReaDirect content.")
    parser.add_argument("--input", default="content_bank_enriched/enriched_content_index.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    report = validate_enriched_dataframe(pd.read_csv(args.input))
    print("Enriched content validation report")
    for key, value in report.items():
        print(f"{key}: {value}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

