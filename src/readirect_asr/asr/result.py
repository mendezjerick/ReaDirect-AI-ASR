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
    raw_transcript_original: str = ""
    wav2vec2_transcript: str = ""
    asr_route: str = ""
    model_family: str = ""
    model_used: str = ""
    language: str = ""
    confidence: float | None = None
    segments: list[ASRSegment] = field(default_factory=list)
    words: list[ASRWord] | None = None
    duration_seconds: float | None = None
    audio_sample_rate: int | None = None
    provider: str = ""
    model_size: str = ""
    processing_seconds: float | None = None
    inference_time_ms: float | None = None
    observed_phonemes: list[str] = field(default_factory=list)
    phoneme_model_used: str = ""
    phoneme_inference_time_ms: float | None = None
    phoneme_error: str | None = None
    decoded_acoustic_phonemes: list[str] = field(default_factory=list)
    acoustic_frame_count: int | None = None
    debug_metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    trace_notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
