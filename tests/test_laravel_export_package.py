import subprocess
import sys
import zipfile
from pathlib import Path


def test_export_dry_run_lists_expected_output(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/export_laravel_integration_package.py", "--output-dir", str(tmp_path / "export"), "--zip-output", str(tmp_path / "export.zip"), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Would include docs" in result.stdout


def test_export_excludes_model_and_datasets_by_default(tmp_path):
    out = tmp_path / "export"
    zip_path = tmp_path / "export.zip"
    result = subprocess.run(
        [sys.executable, "scripts/export_laravel_integration_package.py", "--output-dir", str(out), "--zip-output", str(zip_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert (out / "EXPORT_MANIFEST.md").exists()
    manifest = (out / "EXPORT_MANIFEST.md").read_text(encoding="utf-8")
    assert "model artifacts by default" in manifest
    assert "External datasets included: `false`" in manifest
    assert not (out / "model_artifacts").exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = "\n".join(zf.namelist())
    assert "Speechocean762" not in names


def test_export_includes_enriched_content_if_present(tmp_path):
    out = tmp_path / "export"
    result = subprocess.run(
        [sys.executable, "scripts/export_laravel_integration_package.py", "--output-dir", str(out), "--zip-output", str(tmp_path / "export.zip")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    if Path("content_bank_enriched/enriched_content_index.csv").exists():
        assert (out / "content" / "enriched_content_index.csv").exists()
