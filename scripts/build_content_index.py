from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.content.loader import build_content_index
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an enriched ReaDirect content index.")
    parser.add_argument("--content-bank", default="content_bank", type=Path)
    parser.add_argument("--cmudict-dir", default="external_datasets/cmudict", type=Path)
    parser.add_argument("--output", default="data/manifests/content_index.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = CMUDictLoader(
        args.cmudict_dir / "cmudict.dict",
        args.cmudict_dir / "cmudict.phones",
        args.cmudict_dir / "cmudict.symbols",
    ).load()
    index = build_content_index(args.content_bank, loader, enrich_phonemes=True)
    index.save_csv(args.output)
    df = index.to_dataframe()
    phoneme_success = int(df["expected_phonemes"].fillna("").astype(str).str.strip().ne("").sum()) if not df.empty else 0
    print(f"Wrote {len(df)} content items to {args.output}")
    print(f"Duplicate prompt IDs: {len(index.duplicate_prompt_ids())}")
    print(f"Phoneme-enriched items: {phoneme_success}")


if __name__ == "__main__":
    main()

