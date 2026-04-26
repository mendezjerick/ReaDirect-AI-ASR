from pathlib import Path

import pandas as pd

from readirect_asr.finetuning.dataset_prep import prepare_whisper_dataset


def test_jsonl_files_created_and_rows_skipped(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    df = pd.DataFrame(
        {
            "audio_path": [str(audio), str(audio), "missing.wav"],
            "manual_transcript": ["cat", "", "dog"],
            "duration_seconds": [1.0, 1.0, 1.0],
            "split": ["train", "validation", "test"],
            "dataset_source": ["fake", "fake", "fake"],
            "recording_id": ["1", "2", "3"],
        }
    )
    summary = prepare_whisper_dataset(df, tmp_path / "out")
    assert summary["counts"]["train"] == 1
    assert summary["skipped"]["blank_transcript"] == 1
    assert summary["skipped"]["audio_file_not_found"] == 1
    assert (tmp_path / "out" / "train.jsonl").exists()
    assert (tmp_path / "out" / "dataset_summary.json").exists()


def test_dry_run_writes_nothing(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    df = pd.DataFrame({"audio_path": [str(audio)], "manual_transcript": ["cat"], "duration_seconds": [1.0]})
    summary = prepare_whisper_dataset(df, tmp_path / "out", dry_run=True)
    assert summary["total_rows"] == 1
    assert not (tmp_path / "out").exists()
