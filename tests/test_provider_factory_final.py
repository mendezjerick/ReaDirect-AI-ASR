from readirect_asr.asr.faster_whisper_asr import FasterWhisperASR
from readirect_asr.asr.hf_whisper_asr import HFWhisperLocalASR
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.asr.provider_factory import create_asr_provider


def test_mock_provider_loads_without_heavy_model():
    provider = create_asr_provider({"provider": "mock"})
    assert isinstance(provider, MockASR)


def test_hf_whisper_local_provider_is_lazy_with_missing_path(tmp_path):
    provider = create_asr_provider({"provider": "hf_whisper_local", "hf_model_path": str(tmp_path / "missing")})
    assert isinstance(provider, HFWhisperLocalASR)
    assert provider.is_available() is False
    assert provider._model is None


def test_faster_whisper_local_provider_uses_ct2_path(tmp_path):
    provider = create_asr_provider({"provider": "faster_whisper_local", "ct2_model_path": str(tmp_path / "ct2")})
    assert isinstance(provider, FasterWhisperASR)
    assert provider.model_size == str(tmp_path / "ct2")
    assert provider._model is None


def test_faster_whisper_pretrained_provider():
    provider = create_asr_provider({"provider": "faster_whisper_pretrained", "pretrained_model_size": "base.en"})
    assert isinstance(provider, FasterWhisperASR)
    assert provider.model_size == "base.en"
