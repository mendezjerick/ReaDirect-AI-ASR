from unittest.mock import patch

from readirect_asr.asr.hf_whisper_asr import HFWhisperLocalASR


def test_hf_whisper_config_parses_values():
    provider = HFWhisperLocalASR("model-path", device="cuda", use_fp16=True, language="en")
    assert provider.model_path == "model-path"
    assert provider.device == "cuda"
    assert provider.use_fp16 is True


def test_missing_model_directory_returns_clear_error(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    provider = HFWhisperLocalASR(str(tmp_path / "missing"))
    result = provider.transcribe(str(audio))
    assert result.error
    assert "model path not found" in result.error


def test_missing_audio_returns_clear_error(tmp_path):
    provider = HFWhisperLocalASR(str(tmp_path))
    result = provider.transcribe(str(tmp_path / "missing.wav"))
    assert "audio file not found" in result.error


def test_librosa_loader_can_be_mocked(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    provider = HFWhisperLocalASR(str(tmp_path))
    with patch.object(provider, "_load_model", side_effect=RuntimeError("mock model stop")):
        result = provider.transcribe(str(audio))
    assert result.error == "mock model stop"
