import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_config(tmp_path: Path, train_path: Path, val_path: Path) -> Path:
    config = {
        "model": {"name_or_path": "openai/whisper-tiny.en", "language": "English", "task": "transcribe", "forced_decoder_ids": None, "suppress_tokens": []},
        "data": {
            "train_jsonl": str(train_path),
            "validation_jsonl": str(val_path),
            "test_jsonl": str(val_path),
            "audio_column": "audio",
            "text_column": "sentence",
            "sampling_rate": 16000,
            "min_duration_seconds": 0.3,
            "max_duration_seconds": 30.0,
        },
        "training": {
            "output_dir": str(tmp_path / "model_artifacts" / "out"),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "max_steps": 1,
            "fp16": True,
            "gradient_checkpointing": True,
        },
        "runtime": {"require_cuda": True, "allow_cpu_training": False, "dry_run_default": True},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_training_script_dry_run_works_on_fake_config(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    row = json.dumps({"audio": str(audio), "sentence": "cat", "duration_seconds": 1.0}) + "\n"
    train.write_text(row, encoding="utf-8")
    val.write_text(row, encoding="utf-8")
    config = _write_config(tmp_path, train, val)
    result = subprocess.run([sys.executable, "training/train_whisper.py", "--config", str(config), "--dry-run"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "No model download or training started" in result.stdout
    assert "Training-time evaluation: False" in result.stdout


def test_training_script_missing_dataset_gives_clear_error(tmp_path: Path):
    config = _write_config(tmp_path, tmp_path / "missing_train.jsonl", tmp_path / "missing_val.jsonl")
    result = subprocess.run([sys.executable, "training/train_whisper.py", "--config", str(config), "--dry-run"], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "Training input errors" in result.stdout
