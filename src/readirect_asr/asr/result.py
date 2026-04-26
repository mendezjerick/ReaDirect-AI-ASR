from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ASRSegment:
    start: float | None
    end: float | None
    text: str


@dataclass
class ASRWord:
    word: str
    start: float | None = None
    end: float | None = None
    probability: float | None = None


@dataclass
class ASRResult:
    transcript: str = ""
    normalized_transcript: str = ""
    language: str = ""
    confidence: float | None = None
    segments: list[ASRSegment] = field(default_factory=list)
    words: list[ASRWord] | None = None
    duration_seconds: float | None = None
    provider: str = ""
    model_size: str = ""
    processing_seconds: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

