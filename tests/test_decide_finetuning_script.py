import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_decide_finetuning_script_runs_on_fake_data(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    baseline = tmp_path / "baseline.csv"
    output = tmp_path / "decision.md"
    config = tmp_path / "config.yaml"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    pd.DataFrame(
        {
            "audio_path": [str(audio), str(audio)],
            "manual_transcript": ["cat", "dog"],
            "duration_seconds": [1800, 1800],
        }
    ).to_csv(manifest, index=False)
    pd.DataFrame(
        {
            "manual_transcript": ["cat", "dog"],
            "normalized_transcript": ["cap", ""],
            "audio_path": [str(audio), str(audio)],
            "duration_seconds": [1800, 1800],
        }
    ).to_csv(baseline, index=False)
    config.write_text("min_rows: 2\nmin_total_hours: 0.5\nmin_transcript_coverage: 0.9\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/decide_finetuning.py",
            "--manifest",
            str(manifest),
            "--baseline",
            str(baseline),
            "--output",
            str(output),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert output.exists()
    assert "Decision:" in result.stdout


def test_decide_finetuning_script_handles_missing_baseline(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "decision.md"
    pd.DataFrame({"audio_path": [""], "manual_transcript": ["cat"], "duration_seconds": [1.0]}).to_csv(manifest, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/decide_finetuning.py",
            "--manifest",
            str(manifest),
            "--baseline",
            str(tmp_path / "missing.csv"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "baseline_missing" in result.stdout
