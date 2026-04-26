import os
import subprocess
import sys


def test_startup_validation_passes_with_mock_provider(monkeypatch):
    env = os.environ.copy()
    env["ASR_PROVIDER"] = "mock"
    env["ASR_DEVICE"] = "cpu"
    result = subprocess.run([sys.executable, "scripts/validate_ai_service_startup.py"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "Speechocean762" in result.stdout


def test_startup_validation_warns_or_fails_when_local_model_missing(tmp_path):
    env = os.environ.copy()
    env["ASR_PROVIDER"] = "hf_whisper_local"
    env["ASR_HF_MODEL_PATH"] = str(tmp_path / "missing")
    env["ASR_DEVICE"] = "cpu"
    result = subprocess.run([sys.executable, "scripts/validate_ai_service_startup.py"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "HF model path missing" in result.stdout
