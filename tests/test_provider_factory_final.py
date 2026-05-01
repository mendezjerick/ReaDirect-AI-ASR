from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.asr.provider_factory import create_asr_provider
from readirect_asr.asr.wav2vec2_asr import Wav2Vec2OnlyASR


def test_mock_provider_loads_without_heavy_model():
    provider = create_asr_provider({"provider": "mock"})
    assert isinstance(provider, MockASR)


def test_wav2vec2_provider_is_lazy_with_missing_path(tmp_path):
    provider = create_asr_provider({"provider": "wav2vec2_only", "wav2vec2_asr_model_path": str(tmp_path / "missing")})
    assert isinstance(provider, Wav2Vec2OnlyASR)
    assert provider.is_available() is False
    assert provider._model is None


def test_wav2vec2_provider_uses_configured_paths(tmp_path):
    provider = create_asr_provider(
        {
            "provider": "hf_wav2vec2_local",
            "wav2vec2_asr_model_path": str(tmp_path / "asr"),
            "wav2vec2_phoneme_model_path": str(tmp_path / "phoneme"),
            "wav2vec2_base_asr_model_path": str(tmp_path / "base"),
            "allow_wav2vec2_base_fallback": True,
        }
    )
    assert isinstance(provider, Wav2Vec2OnlyASR)
    assert provider.model_path == str(tmp_path / "asr")
    assert provider.phoneme_model_path == str(tmp_path / "phoneme")
    assert provider.base_model_path == str(tmp_path / "base")
    assert provider.allow_base_fallback is True
