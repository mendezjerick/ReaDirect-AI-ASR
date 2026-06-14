from __future__ import annotations

from typing import Any

from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.asr.wav2vec2_asr import Wav2Vec2OnlyASR


def create_asr_provider(config: dict[str, Any]):
    provider = str(config.get("provider", "wav2vec2_only"))
    if provider in {"wav2vec2_only", "wav2vec2", "hf_wav2vec2_local"}:
        return Wav2Vec2OnlyASR(
            model_path=str(config.get("wav2vec2_asr_model_path", "models/asr/epsilon")),
            model_name=str(config.get("model_name", "epsilon")),
            phoneme_model_path=str(config.get("wav2vec2_phoneme_model_path", "models/wav2vec2-phoneme")),
            base_model_path=str(config.get("wav2vec2_base_asr_model_path", "models/wav2vec2-readirect-asr")),
            allow_base_fallback=_as_bool(config.get("allow_wav2vec2_base_fallback", False)),
            device=str(config.get("device", "cpu")),
            decode_mode=str(config.get("decode_mode", "beam_lm")),
            beam_width=int(config.get("beam_width", 100)),
            lm_path=str(config.get("lm_path", "")) or None,
            alpha=float(config.get("alpha", 0.5)),
            beta=float(config.get("beta", 1.0)),
            hotwords=tuple(config.get("hotwords", []) or []),
            hotword_weight=float(config.get("hotword_weight", 5.0)),
            allow_no_lm_fallback=_as_bool(config.get("allow_no_lm_fallback", False)),
        )
    return MockASR()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
