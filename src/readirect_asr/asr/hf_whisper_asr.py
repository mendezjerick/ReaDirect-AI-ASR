from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from readirect_asr.asr.base import ASRProvider
from readirect_asr.asr.result import ASRResult
from readirect_asr.finetuning.whisper_audio import load_audio_array
from readirect_asr.finetuning.whisper_generation_config import prepare_whisper_generation_config
from readirect_asr.text.normalization import normalize_transcript


class HFWhisperLocalASR(ASRProvider):
    provider = "hf_whisper_local"

    def __init__(
        self,
        model_path: str = "model_artifacts/readirect-whisper-base-en-v1-hf",
        device: str = "cuda",
        use_fp16: bool = True,
        language: str = "en",
        task: str = "transcribe",
        sampling_rate: int = 16000,
    ) -> None:
        self.model_path = model_path
        self.model_size = model_path
        self.device = device
        self.use_fp16 = use_fp16
        self.language = language
        self.task = task
        self.sampling_rate = sampling_rate
        self._model: Any | None = None
        self._processor: Any | None = None

    def is_available(self) -> bool:
        return Path(self.model_path).exists()

    def _load_model(self) -> tuple[Any, Any, Any, str]:
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise RuntimeError(f"local Hugging Face Whisper model path not found: {model_dir}")
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for hf_whisper_local ASR") from exc
        actual_device = self.device
        if actual_device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"
        if self._model is None or self._processor is None:
            self._processor = WhisperProcessor.from_pretrained(model_dir)
            self._model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(actual_device)
            self._model.eval()
            prepare_whisper_generation_config(self._model, self._processor, language=self.language, task=self.task, verbose=False)
        return self._model, self._processor, torch, actual_device

    def transcribe(self, audio_path: str, **kwargs: Any) -> ASRResult:
        started = time.perf_counter()
        path = Path(audio_path)
        if not path.exists():
            return ASRResult(provider=self.provider, model_size=self.model_path, language=self.language, error=f"audio file not found: {audio_path}")
        try:
            model, processor, torch, actual_device = self._load_model()
            audio, sr = load_audio_array(path, sampling_rate=self.sampling_rate, backend="librosa")
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(actual_device)
            if actual_device == "cuda" and self.use_fp16:
                inputs = inputs.half()
                model = model.half()
            with torch.no_grad():
                predicted_ids = model.generate(inputs)
            transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            return ASRResult(
                transcript=transcript,
                normalized_transcript=normalize_transcript(transcript),
                language=self.language,
                provider=self.provider,
                model_size=self.model_path,
                processing_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            return ASRResult(
                provider=self.provider,
                model_size=self.model_path,
                language=self.language,
                processing_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
