from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.content.enricher import ContentEnricher
from readirect_asr.content.loader import build_content_index
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def load_content(content_bank: Path, content_index: Path | None) -> pd.DataFrame:
    if content_index and content_index.exists():
        return pd.read_csv(content_index)
    return build_content_index(content_bank, enrich_phonemes=False).to_dataframe()


def enrich_content_bank(
    content_bank: Path,
    cmudict_dir: Path,
    output_dir: Path,
    content_index: Path | None = None,
    include_assessment: bool = True,
    include_modules: bool = True,
    write_import_ready: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    df = load_content(content_bank, content_index)
    groups = []
    if include_assessment:
        groups.append("assessment")
    if include_modules:
        groups.append("modules")
    if groups and "source_group" in df.columns:
        df = df[df["source_group"].isin(groups)].copy()
    if limit is not None:
        df = df.head(limit).copy()

    loader = CMUDictLoader(
        cmudict_dir / "cmudict.dict",
        cmudict_dir / "cmudict.phones",
        cmudict_dir / "cmudict.symbols",
    ).load()
    enriched = ContentEnricher(loader).enrich_dataframe(df)

    print(f"Total rows: {len(enriched)}")
    print(f"Rows enriched: {int(enriched['enrichment_status'].fillna('').astype(str).ne('').sum()) if not enriched.empty else 0}")
    if not enriched.empty:
        match_rate = enriched["has_cmudict_match"].fillna(False).astype(str).str.lower().isin({"1", "true"}).mean()
        print(f"CMUdict match rate: {round(float(match_rate) * 100, 2)}%")
        print(f"Rows needing manual review: {int(enriched['needs_manual_review'].fillna(False).astype(str).str.lower().isin({'1','true','yes'}).sum())}")
        print("Skill group distribution:")
        print(enriched["skill_group"].fillna("").astype(str).value_counts().to_string())
        print("Error focus distribution:")
        print(enriched["error_focus"].fillna("").astype(str).value_counts().to_string())
        print("Difficulty distribution:")
        print(enriched["difficulty_level"].fillna("").astype(str).value_counts().to_string())

    if dry_run:
        print("Dry run: no files written.")
        return enriched

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "enriched_content_index.csv"
    enriched.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")

    for (source_group, source_file), group_df in enriched.groupby(["source_group", "source_file"], dropna=False):
        group_name = str(source_group or "unknown")
        source_name = Path(str(source_file or "content")).stem
        target_dir = output_dir / group_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{source_name}_enriched.csv"
        group_df.to_csv(target, index=False)
        print(f"Wrote {target}")
        if write_import_ready:
            ready_dir = output_dir / "import_ready" / group_name
            ready_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, ready_dir / target.name)

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich ReaDirect content-bank items with phoneme and adaptive metadata.")
    parser.add_argument("--content-bank", default="content_bank", type=Path)
    parser.add_argument("--content-index", default=None, type=Path)
    parser.add_argument("--cmudict-dir", default="external_datasets/cmudict", type=Path)
    parser.add_argument("--output-dir", default="content_bank_enriched", type=Path)
    parser.add_argument("--include-assessment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-import-ready", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enrich_content_bank(
        content_bank=args.content_bank,
        content_index=args.content_index,
        cmudict_dir=args.cmudict_dir,
        output_dir=args.output_dir,
        include_assessment=args.include_assessment,
        include_modules=args.include_modules,
        write_import_ready=args.write_import_ready,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

