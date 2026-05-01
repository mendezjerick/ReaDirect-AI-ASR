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


def test_startup_validation_fails_when_wav2vec2_model_missing(tmp_path):
    env = os.environ.copy()
    env["ASR_PROVIDER"] = "wav2vec2_only"
    env["WAV2VEC2_ASR_MODEL_PATH"] = str(tmp_path / "missing")
    env["WAV2VEC2_PHONEME_MODEL_PATH"] = str(tmp_path / "missing_phoneme")
    env["ALLOW_WAV2VEC2_BASE_FALLBACK"] = "false"
    env["ASR_DEVICE"] = "cpu"
    result = subprocess.run([sys.executable, "scripts/validate_ai_service_startup.py"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "Wav2Vec2 ASR model path missing" in result.stdout
    assert "Whisper is not required" in result.stdout
