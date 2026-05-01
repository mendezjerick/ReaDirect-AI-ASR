from __future__ import annotations

from typing import Any

from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.asr.wav2vec2_asr import Wav2Vec2OnlyASR


def create_asr_provider(config: dict[str, Any]):
    provider = str(config.get("provider", "wav2vec2_only"))
    if provider in {"wav2vec2_only", "wav2vec2", "hf_wav2vec2_local"}:
        return Wav2Vec2OnlyASR(
            model_path=str(config.get("wav2vec2_asr_model_path", "models/wav2vec2-readirect-asr")),
            phoneme_model_path=str(config.get("wav2vec2_phoneme_model_path", "models/wav2vec2-phoneme")),
            base_model_path=str(config.get("wav2vec2_base_asr_model_path", "models/wav2vec2-base-960h")),
            allow_base_fallback=_as_bool(config.get("allow_wav2vec2_base_fallback", False)),
            device=str(config.get("device", "cpu")),
        )
    return MockASR()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
