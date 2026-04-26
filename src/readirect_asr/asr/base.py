from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        """Transcribe an audio file and return provider-specific metadata."""

