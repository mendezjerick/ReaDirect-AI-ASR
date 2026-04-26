from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from readirect_asr.dataset.manifest import REQUIRED_MANIFEST_COLUMNS


def blank_manifest_row() -> dict[str, object]:
    return {column: "" for column in REQUIRED_MANIFEST_COLUMNS}


def json_dumps(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def find_dataset_root(dataset_dir: str | Path, marker_files: tuple[str, ...]) -> Path:
    root = Path(dataset_dir)
    if any((root / marker).exists() for marker in marker_files):
        return root
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and any((child / marker).exists() for marker in marker_files):
            return child
    return root


def age_to_group(age: object) -> str:
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return ""
    if age_int < 13:
        return "child"
    if age_int < 18:
        return "teen"
    return "adult"


def speaker_type_from_age(age: object) -> str:
    group = age_to_group(age)
    if group in {"child", "teen"}:
        return "child"
    if group == "adult":
        return "adult"
    return ""

