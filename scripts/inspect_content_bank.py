from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.content.loader import build_content_index
from readirect_asr.content.validation import resolve_content_bank_root, validate_content_bank
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ReaDirect content-bank CSVs.")
    parser.add_argument("--content-bank", default="content_bank", type=Path)
    parser.add_argument("--cmudict", default="external_datasets/cmudict", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_content_bank_root(args.content_bank)
    cmu = CMUDictLoader(args.cmudict / "cmudict.dict", args.cmudict / "cmudict.phones", args.cmudict / "cmudict.symbols").load()
    report = validate_content_bank(root)

    print(f"Content bank: {root}")
    csv_files = sorted(root.rglob("*.csv")) if root.exists() else []
    print(f"Detected CSV files: {len(csv_files)}")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            active = int(df["is_active"].fillna(0).astype(str).str.lower().isin(["1", "true", "yes"]).sum()) if "is_active" in df.columns else 0
            inactive = len(df) - active if "is_active" in df.columns else 0
            print(f"- {csv_path.relative_to(root).as_posix()}: {len(df)} rows, active={active}, inactive={inactive}")
        except Exception as exc:
            print(f"- {csv_path.relative_to(root).as_posix()}: read error: {exc}")

    print(f"Missing required files: {len(report['missing_required_files'])}")
    print(f"Missing optional files: {len(report['missing_optional_files'])}")
    print(f"Column errors: {len(report['column_errors'])}")

    index = build_content_index(root, cmu, enrich_phonemes=True)
    df = index.to_dataframe()
    print(f"Indexed items: {len(df)}")
    print(f"Duplicate prompt IDs: {index.duplicate_prompt_ids()}")
    if not df.empty:
        print(f"Sample prompt IDs: {', '.join(df['prompt_id'].head(5).astype(str))}")
        print(f"Missing expected_text count: {int(df['expected_text'].fillna('').astype(str).str.strip().eq('').sum())}")
        cmu_matches = int(df["expected_phonemes"].fillna("").astype(str).str.strip().ne("").sum())
        print(f"CMUdict match count: {cmu_matches}")
        print(f"CMUdict missing word/content count: {len(df) - cmu_matches}")
        print(f"Phoneme enrichment success rate: {round((cmu_matches / len(df)) * 100, 2)}%")
        if "module_key" in df.columns:
            print("Module activity counts:")
            print(df.groupby(["module_key", "activity_type"]).size().to_string())
        if "task_type" in df.columns:
            print("Assessment/content counts by task type:")
            print(df["task_type"].fillna("").value_counts().to_string())


if __name__ == "__main__":
    main()

