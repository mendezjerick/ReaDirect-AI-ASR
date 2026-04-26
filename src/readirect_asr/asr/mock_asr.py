from __future__ import annotations

from typing import Any

from readirect_asr.asr.base import ASRProvider


class MockASR(ASRProvider):
    provider = "mock"

    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        transcript = kwargs.get("expected_text") or kwargs.get("transcript") or "mock transcript"
        return {
            "transcript": str(transcript),
            "confidence": None,
            "provider": self.provider,
            "audio_path": audio_path,
        }

