import pandas as pd

from readirect_asr.dataset.manifest import REQUIRED_MANIFEST_COLUMNS, validate_manifest_columns


def test_validate_manifest_columns_passes_for_required_columns() -> None:
    df = pd.DataFrame(columns=REQUIRED_MANIFEST_COLUMNS)
    assert validate_manifest_columns(df, REQUIRED_MANIFEST_COLUMNS) == []


def test_validate_manifest_columns_reports_missing_columns() -> None:
    df = pd.DataFrame(columns=["recording_id"])
    missing = validate_manifest_columns(df, ["recording_id", "audio_path"])
    assert missing == ["audio_path"]

