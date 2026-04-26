from pathlib import Path

import pandas as pd

from readirect_asr.finetuning.readiness import check_finetuning_readiness


def test_ready_when_enough_rows_transcripts_and_audio(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    df = pd.DataFrame(
        {
            "audio_path": [str(audio)] * 3,
            "manual_transcript": ["cat", "dog", "sun"],
            "duration_seconds": [1200, 1200, 1200],
        }
    )
    result = check_finetuning_readiness(df, min_rows=3, min_total_hours=1.0)
    assert result["ready"] is True
    assert result["transcript_coverage"] == 1.0
    assert result["audio_available_rate"] == 1.0


def test_not_ready_when_too_few_rows(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    df = pd.DataFrame({"audio_path": [str(audio)], "manual_transcript": ["cat"], "duration_seconds": [1.0]})
    result = check_finetuning_readiness(df, min_rows=5, min_total_hours=0.0)
    assert result["ready"] is False
    assert "too_few_rows" in result["issues"]


def test_warns_on_missing_transcripts_and_audio():
    df = pd.DataFrame(
        {
            "audio_path": ["missing.wav", ""],
            "manual_transcript": ["", "cat"],
            "duration_seconds": [1.0, 1.0],
        }
    )
    result = check_finetuning_readiness(df, min_rows=1, min_total_hours=0.0)
    assert "low_transcript_coverage" in result["issues"]
    assert "some_audio_paths_missing" in result["warnings"]
    assert "blank_reference_transcripts_detected" in result["warnings"]
