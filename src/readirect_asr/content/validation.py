from __future__ import annotations

from pathlib import Path

import pandas as pd

from readirect_asr.content.schemas import KNOWN_SCHEMAS, OPTIONAL_CONTENT_FILES, REQUIRED_CONTENT_FILES


def validate_csv_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    file_name: str,
) -> dict[str, object]:
    missing = [column for column in required_columns if column not in df.columns]
    return {
        "file": file_name,
        "missing_columns": missing,
        "ok": not missing,
    }


def resolve_content_bank_root(content_bank_path: str | Path) -> Path:
    root = Path(content_bank_path)
    nested = root / "readirect-content-bank"
    if nested.exists():
        return nested
    return root


def validate_required_files(content_bank_path: str | Path) -> dict[str, list[str]]:
    root = resolve_content_bank_root(content_bank_path)
    missing_required = [path for path in REQUIRED_CONTENT_FILES if not (root / path).exists()]
    missing_optional = [path for path in OPTIONAL_CONTENT_FILES if not (root / path).exists()]
    return {
        "missing_required_files": missing_required,
        "missing_optional_files": missing_optional,
    }


def validate_content_bank(content_bank_path: str | Path) -> dict[str, object]:
    root = resolve_content_bank_root(content_bank_path)
    file_report = validate_required_files(root)
    column_errors: list[dict[str, object]] = []
    warnings: list[str] = []

    for csv_path in root.rglob("*.csv"):
        required_columns = KNOWN_SCHEMAS.get(csv_path.name)
        if not required_columns:
            warnings.append(f"No schema registered for {csv_path.relative_to(root).as_posix()}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            column_errors.append(
                {
                    "file": csv_path.relative_to(root).as_posix(),
                    "missing_columns": [],
                    "ok": False,
                    "error": str(exc),
                }
            )
            continue
        result = validate_csv_columns(df, required_columns, csv_path.relative_to(root).as_posix())
        if not result["ok"]:
            column_errors.append(result)

    return {
        "content_bank_path": str(root),
        **file_report,
        "column_errors": column_errors,
        "warnings": warnings,
        "ok": not file_report["missing_required_files"] and not column_errors,
    }

