import json
from pathlib import Path

from readirect_asr.finetuning.whisper_dataset import (
    load_jsonl_dataset,
    summarize_whisper_dataset,
    validate_whisper_jsonl,
)


def test_validates_jsonl_structure(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"audio": str(audio), "sentence": "cat", "duration_seconds": 1.0}) + "\n", encoding="utf-8")
    report = validate_whisper_jsonl(path)
    assert report["valid_rows"] == 1
    assert report["invalid_rows"] == 0
    rows = load_jsonl_dataset(path)
    assert rows[0]["sentence"] == "cat"


def test_rejects_missing_audio_and_blank_sentence(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"audio": str(tmp_path / "missing.wav"), "sentence": ""}) + "\n",
        encoding="utf-8",
    )
    report = validate_whisper_jsonl(path)
    assert "audio_file_not_found" in report["issues"]
    assert "blank_sentence" in report["issues"]


def test_summarizes_fake_dataset(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    summary = summarize_whisper_dataset({"train": [{"audio": str(audio), "sentence": "cat", "duration_seconds": 1.0}]})
    assert summary["split_counts"]["train"] == 1
    assert summary["missing_audio_rows"] == 0
