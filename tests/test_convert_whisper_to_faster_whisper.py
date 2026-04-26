import subprocess
import sys
from pathlib import Path


def test_conversion_dry_run_prints_command_without_dependency(tmp_path: Path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "ct2"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_whisper_to_faster_whisper.py",
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "ct2-transformers-converter" in result.stdout
    assert "Dry run complete" in result.stdout


def test_conversion_missing_model_dir_non_dry_run_fails_safely(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "scripts/convert_whisper_to_faster_whisper.py", "--model-dir", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Model directory not found" in result.stdout
