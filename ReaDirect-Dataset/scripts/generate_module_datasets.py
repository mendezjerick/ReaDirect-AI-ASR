#!/usr/bin/env python3
"""Sync current ReaDirect learner module CSV banks into the AI-ASR dataset copy."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"


MODULE_SOURCES = {
    "module1_letter_sound_activities_adaptive_v2.csv": "module1_letter_sound_activities.csv",
    "module2_word_reading_activities_adaptive_v2.csv": "module2_word_reading_activities.csv",
    "module3_sentence_fluency_activities_adaptive_v2.csv": "module3_sentence_fluency_activities.csv",
    "module_activity_selection_rules.csv": "module_activity_selection_rules.csv",
}


def default_seed_data_dir() -> Path:
    workspace_root = ROOT.parents[1]
    return workspace_root / "ReaDirect" / "database" / "seed-data" / "readirect"


def resolve_seed_data_dir() -> Path:
    configured = os.environ.get("READIRECT_SEED_DATA_DIR", "").strip()
    return Path(configured) if configured else default_seed_data_dir()


def sync_module_csvs(seed_data_dir: Path) -> list[Path]:
    if not seed_data_dir.exists():
        raise FileNotFoundError(f"Seed data directory not found: {seed_data_dir}")

    MODULES.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source_name, target_name in MODULE_SOURCES.items():
        source = seed_data_dir / source_name
        if not source.exists():
            raise FileNotFoundError(f"Canonical module CSV not found: {source}")
        target = MODULES / target_name
        temp_target = target.with_name(f"{target.name}.tmp")
        if temp_target.exists():
            temp_target.unlink()
        shutil.copy2(source, temp_target)
        temp_target.replace(target)
        written.append(target)
    return written


def main() -> int:
    written = sync_module_csvs(resolve_seed_data_dir())
    for path in written:
        print(f"Synced {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
