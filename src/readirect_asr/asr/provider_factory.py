from __future__ import annotations

from typing import Any

from readirect_asr.asr.faster_whisper_asr import FasterWhisperASR
from readirect_asr.asr.mock_asr import MockASR


def create_asr_provider(config: dict[str, Any]):
    provider = str(config.get("provider", "mock"))
    if provider in {"faster_whisper", "faster-whisper"}:
        return FasterWhisperASR(
            model_size=str(config.get("model_size", "base.en")),
            device=str(config.get("device", "cpu")),
            compute_type=str(config.get("compute_type", "int8")),
            language=str(config.get("language", "en")),
            beam_size=int(config.get("beam_size", 1)),
        )
    return MockASR()

