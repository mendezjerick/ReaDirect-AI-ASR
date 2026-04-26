from __future__ import annotations

from typing import Any

from readirect_asr.asr.faster_whisper_asr import FasterWhisperASR
from readirect_asr.asr.hf_whisper_asr import HFWhisperLocalASR
from readirect_asr.asr.mock_asr import MockASR


def create_asr_provider(config: dict[str, Any]):
    provider = str(config.get("provider", "mock"))
    if provider in {"faster_whisper", "faster-whisper", "faster_whisper_pretrained"}:
        return FasterWhisperASR(
            model_size=str(config.get("pretrained_model_size", config.get("model_size", "base.en"))),
            device=str(config.get("device", "cpu")),
            compute_type=str(config.get("compute_type", "int8")),
            language=str(config.get("language", "en")),
            beam_size=int(config.get("beam_size", 1)),
        )
    if provider == "faster_whisper_local":
        return FasterWhisperASR(
            model_size=str(config.get("ct2_model_path", "model_artifacts/readirect-whisper-base-en-v1-ct2")),
            device=str(config.get("device", "cuda")),
            compute_type=str(config.get("compute_type", "float16")),
            language=str(config.get("language", "en")),
            beam_size=int(config.get("beam_size", 1)),
        )
    if provider == "hf_whisper_local":
        return HFWhisperLocalASR(
            model_path=str(config.get("hf_model_path", "model_artifacts/readirect-whisper-base-en-v1-hf")),
            device=str(config.get("device", "cuda")),
            use_fp16=_as_bool(config.get("use_fp16", True)),
            language=str(config.get("language", "en")),
            task=str(config.get("task", "transcribe")),
        )
    return MockASR()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
