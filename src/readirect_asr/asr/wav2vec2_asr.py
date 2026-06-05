from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from readirect_asr.asr.base import ASRProvider
from readirect_asr.asr.result import ASRResult
from readirect_asr.text.normalization import normalize_transcript


class Wav2Vec2OnlyASR(ASRProvider):
    provider = "wav2vec2_only"

    def __init__(
        self,
        model_path: str = "models/wav2vec2-readirect-asr-letters-v2",
        phoneme_model_path: str = "models/wav2vec2-phoneme",
        base_model_path: str = "models/wav2vec2-readirect-asr",
        allow_base_fallback: bool = False,
        device: str = "cpu",
        sampling_rate: int = 16000,
    ) -> None:
        self.model_path = model_path
        self.phoneme_model_path = phoneme_model_path
        self.base_model_path = base_model_path
        self.allow_base_fallback = allow_base_fallback
        self.device = device
        self.sampling_rate = sampling_rate
        self.model_size = model_path
        self.model_family = "wav2vec2"
        self.asr_route = "wav2vec2_only"
        self._model: Any | None = None
        self._processor: Any | None = None
        self._phoneme_model: Any | None = None
        self._phoneme_processor: Any | None = None
        self._torch: Any | None = None
        self._actual_device: str | None = None
        self._active_model_path: str | None = None
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    @property
    def active_model_path(self) -> str:
        return self._active_model_path or self.model_path

    def is_available(self) -> bool:
        return self._select_asr_model_path(strict=False) is not None and Path(self.phoneme_model_path).exists()

    def status(self) -> dict[str, Any]:
        asr_path = self._select_asr_model_path(strict=False)
        missing_paths = []
        if asr_path is None:
            missing_paths.append(self.model_path)
        if not Path(self.phoneme_model_path).exists():
            missing_paths.append(self.phoneme_model_path)
        active_model = (asr_path or Path(self.model_path)).as_posix()
        metadata = _model_metadata(Path(active_model))
        return {
            "asr_architecture": self.asr_route,
            "active_asr_model": "wav2vec2",
            "active_asr_model_path": active_model,
            "wav2vec2_asr_available": asr_path is not None,
            "wav2vec2_asr_model_name": active_model,
            "wav2vec2_phoneme_available": Path(self.phoneme_model_path).exists(),
            "wav2vec2_phoneme_model_name": self.phoneme_model_path,
            "base_fallback_enabled": self.allow_base_fallback,
            "using_base_fallback": asr_path == Path(self.base_model_path),
            "missing_model_paths": missing_paths,
            **metadata,
        }

    def _select_asr_model_path(self, strict: bool = True) -> Path | None:
        primary = Path(self.model_path)
        if primary.exists():
            self._active_model_path = str(primary)
            return primary
        if self.allow_base_fallback:
            fallback = Path(self.base_model_path)
            if fallback.exists():
                self._active_model_path = str(fallback)
                return fallback
        if strict:
            if self.allow_base_fallback:
                raise RuntimeError(f"Wav2Vec2 ASR model missing: {primary}; base fallback also missing: {self.base_model_path}")
            raise RuntimeError(f"Wav2Vec2 ASR model missing: {primary}; ALLOW_WAV2VEC2_BASE_FALLBACK is false")
        return None

    def _load_model(self) -> tuple[Any, Any, Any, str]:
        model_dir = self._select_asr_model_path(strict=True)
        assert model_dir is not None
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for Wav2Vec2 ASR") from exc
        actual_device = self.device
        if actual_device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"
        if self._model is None or self._processor is None:
            self._processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
            self._model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True).to(actual_device)
            self._model.eval()
        self._torch = torch
        self._actual_device = actual_device
        return self._model, self._processor, torch, actual_device

    def _load_phoneme_model(self) -> tuple[Any, Any, Any, str]:
        model_dir = Path(self.phoneme_model_path)
        if not model_dir.exists():
            raise RuntimeError(f"Wav2Vec2 phoneme model path not found: {model_dir}")
        try:
            import torch
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for Wav2Vec2 phoneme evidence") from exc
        actual_device = self._actual_device or self.device
        if actual_device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"
        if self._phoneme_model is None or self._phoneme_processor is None:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            self._phoneme_processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self._phoneme_model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True).to(actual_device)
            self._phoneme_model.eval()
        return self._phoneme_model, self._phoneme_processor, torch, actual_device

    def transcribe(self, audio_path: str, **kwargs: Any) -> ASRResult:
        del kwargs
        started = time.perf_counter()
        path = Path(audio_path)
        if not path.exists():
            return ASRResult(provider=self.provider, model_size=self.active_model_path, model_family=self.model_family, asr_route=self.asr_route, error=f"audio file not found: {audio_path}")
        try:
            audio, sr = _load_audio(path, self.sampling_rate)
            duration = round(len(audio) / float(sr), 6) if sr else None
            model, processor, torch, actual_device = self._load_model()
            inference_started = time.perf_counter()
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(actual_device)
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            decoded = processor.batch_decode(predicted_ids)[0].strip()
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 3)
            observed_phonemes, phoneme_ms, phoneme_error, phoneme_frame_count = self._phoneme_evidence(audio, sr)
            normalized = normalize_transcript(decoded)
            return ASRResult(
                transcript=normalized,
                normalized_transcript=normalized,
                raw_transcript_original=decoded,
                wav2vec2_transcript=normalized,
                asr_route=self.asr_route,
                model_family=self.model_family,
                model_used=self.active_model_path,
                language="en",
                duration_seconds=duration,
                audio_sample_rate=sr,
                provider=self.provider,
                model_size=self.active_model_path,
                processing_seconds=round(time.perf_counter() - started, 3),
                inference_time_ms=inference_ms,
                observed_phonemes=observed_phonemes,
                decoded_acoustic_phonemes=observed_phonemes,
                acoustic_frame_count=phoneme_frame_count,
                phoneme_model_used=self.phoneme_model_path,
                phoneme_inference_time_ms=phoneme_ms,
                phoneme_error=phoneme_error,
                debug_metadata={
                    "actual_device": actual_device,
                    "raw_transcript_original": decoded,
                    "base_fallback_used": self.active_model_path == self.base_model_path,
                },
            )
        except Exception as exc:
            return ASRResult(
                provider=self.provider,
                model_size=self.active_model_path,
                model_family=self.model_family,
                asr_route=self.asr_route,
                processing_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )

    def phoneme_frame_evidence(self, audio_path: str) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            return {"available": False, "error": f"audio file not found: {audio_path}"}
        try:
            audio, sr = _load_audio(path, self.sampling_rate)
            model, processor, torch, actual_device = self._load_phoneme_model()
            started = time.perf_counter()
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(actual_device)
            with torch.no_grad():
                logits = model(input_values).logits[0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).detach().cpu().numpy()
                predicted_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            vocabulary = _processor_vocabulary(processor)
            decoded = _ctc_decode_ids(predicted_ids, vocabulary, _blank_token_id(processor))
            return {
                "available": True,
                "model_version": "existing_wavtec_phoneme_model",
                "model_path": self.phoneme_model_path,
                "sample_rate": sr,
                "duration_seconds": round(len(audio) / float(sr), 6) if sr else None,
                "frame_count": int(log_probs.shape[0]),
                "vocabulary": vocabulary,
                "blank_token_id": _blank_token_id(processor),
                "log_probs": log_probs,
                "decoded_phonemes": decoded,
                "inference_time_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except Exception as exc:
            return {
                "available": False,
                "model_version": "existing_wavtec_phoneme_model",
                "model_path": self.phoneme_model_path,
                "error": str(exc),
            }

    def _phoneme_evidence(self, audio: Any, sr: int) -> tuple[list[str], float | None, str | None, int | None]:
        try:
            model, processor, torch, actual_device = self._load_phoneme_model()
            started = time.perf_counter()
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(actual_device)
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            decoded = processor.batch_decode(predicted_ids)[0]
            return _normalize_phoneme_output(decoded), round((time.perf_counter() - started) * 1000, 3), None, int(logits.shape[1])
        except Exception as exc:
            return [], None, str(exc), None


def _load_audio(path: Path, sampling_rate: int) -> tuple[Any, int]:
    import librosa

    audio, sr = librosa.load(str(path), sr=sampling_rate, mono=True)
    return audio, int(sr)


def _normalize_phoneme_output(text: str) -> list[str]:
    cleaned = str(text or "").replace("|", " ").replace("/", " ")
    phones = []
    for token in cleaned.split():
        phone = "".join(char for char in token.upper() if char.isalpha())
        if phone:
            phones.append(phone)
    return phones


def _processor_vocabulary(processor: Any) -> dict[int, str]:
    tokenizer = getattr(processor, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        return {}
    vocab = get_vocab()
    return {int(index): str(token) for token, index in vocab.items()}


def _blank_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    value = getattr(tokenizer, "pad_token_id", None)
    return int(value) if value is not None else None


def _ctc_decode_ids(ids: list[int], vocabulary: dict[int, str], blank_token_id: int | None) -> list[str]:
    phones: list[str] = []
    previous: int | None = None
    for token_id in ids:
        if token_id == previous:
            continue
        previous = token_id
        if blank_token_id is not None and token_id == blank_token_id:
            continue
        token = vocabulary.get(int(token_id), "")
        normalized = _normalize_phoneme_output(token)
        phones.extend(normalized)
    return phones


def _model_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "readirect_model_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    model_name = str(data.get("model_name", ""))
    model_version = str(data.get("model_version") or "")
    if not model_version and model_name.endswith("letters-v2"):
        model_version = "letters-v2"
    return {
        "model_version": model_version,
        "base_model": str(data.get("base_model_path") or data.get("base_model") or ""),
        "training_type": str(data.get("training_type") or ""),
        "training_mix": _format_training_mix(data.get("training_mix")),
    }


def _format_training_mix(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    labels = {
        "readirect_letters": "ReaDirect letters",
        "speechocean": "SpeechOcean",
        "librispeech": "LibriSpeech",
    }
    parts = []
    for key in ("readirect_letters", "speechocean", "librispeech"):
        if key not in value:
            continue
        try:
            ratio = float(value[key])
        except (TypeError, ValueError):
            continue
        percent = int(round(ratio * 100)) if ratio <= 1 else int(round(ratio))
        parts.append(f"{percent}% {labels[key]}")
    return ", ".join(parts)
