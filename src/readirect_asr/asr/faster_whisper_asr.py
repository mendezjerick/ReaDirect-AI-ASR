from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from readirect_asr.asr.base import ASRProvider
from readirect_asr.asr.result import ASRResult, ASRSegment, ASRWord
from readirect_asr.audio.preprocessing import get_audio_duration_seconds
from readirect_asr.text.normalization import normalize_transcript


class FasterWhisperASR(ASRProvider):
    provider = "faster_whisper"

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = False,
        return_word_timestamps: bool = True,
        temperature: float = 0.0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.return_word_timestamps = return_word_timestamps
        self.temperature = temperature
        self._model: Any | None = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def _load_model(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it before real ASR runs: "
                "pip install faster-whisper"
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str, **kwargs: Any) -> ASRResult:
        started = time.perf_counter()
        path = Path(audio_path)
        if not path.exists():
            return ASRResult(
                provider=self.provider,
                model_size=self.model_size,
                language=self.language,
                error=f"audio file not found: {audio_path}",
                processing_seconds=round(time.perf_counter() - started, 3),
            )

        try:
            model = self._load_model()
            segments_iter, info = model.transcribe(
                str(path),
                language=kwargs.get("language", self.language),
                beam_size=int(kwargs.get("beam_size", self.beam_size)),
                vad_filter=bool(kwargs.get("vad_filter", self.vad_filter)),
                word_timestamps=bool(kwargs.get("word_timestamps", self.return_word_timestamps)),
                temperature=float(kwargs.get("temperature", self.temperature)),
            )
            segments_raw = list(segments_iter)
            transcript = " ".join(str(segment.text).strip() for segment in segments_raw).strip()
            segments = [
                ASRSegment(
                    start=getattr(segment, "start", None),
                    end=getattr(segment, "end", None),
                    text=str(getattr(segment, "text", "")).strip(),
                )
                for segment in segments_raw
            ]
            words: list[ASRWord] = []
            for segment in segments_raw:
                for word in getattr(segment, "words", None) or []:
                    words.append(
                        ASRWord(
                            word=str(getattr(word, "word", "")).strip(),
                            start=getattr(word, "start", None),
                            end=getattr(word, "end", None),
                            probability=getattr(word, "probability", None),
                        )
                    )
            return ASRResult(
                transcript=transcript,
                normalized_transcript=normalize_transcript(transcript),
                language=str(getattr(info, "language", self.language) or self.language),
                confidence=None,
                segments=segments,
                words=words or None,
                duration_seconds=get_audio_duration_seconds(path),
                provider=self.provider,
                model_size=self.model_size,
                processing_seconds=round(time.perf_counter() - started, 3),
                error=None,
            )
        except Exception as exc:
            return ASRResult(
                provider=self.provider,
                model_size=self.model_size,
                language=self.language,
                duration_seconds=get_audio_duration_seconds(path),
                processing_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )

