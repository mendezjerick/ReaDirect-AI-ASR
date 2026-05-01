from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


VALID_PROVIDERS = {"mock", "wav2vec2_only", "wav2vec2", "hf_wav2vec2_local"}


def main() -> int:
    checks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    service_config = _load_service_config()
    asr_config = service_config.get("asr", {})
    provider = os.getenv("ASR_PROVIDER", str(asr_config.get("provider", "mock")))
    device = os.getenv("ASR_DEVICE", str(asr_config.get("device", "cpu")))
    wav2vec2_model = Path(os.getenv("WAV2VEC2_ASR_MODEL_PATH", str(asr_config.get("wav2vec2_asr_model_path", "models/wav2vec2-readirect-asr"))))
    phoneme_model = Path(os.getenv("WAV2VEC2_PHONEME_MODEL_PATH", str(asr_config.get("wav2vec2_phoneme_model_path", "models/wav2vec2-phoneme"))))
    base_model = Path(os.getenv("WAV2VEC2_BASE_ASR_MODEL_PATH", str(asr_config.get("wav2vec2_base_asr_model_path", "models/wav2vec2-base-960h"))))
    allow_base_fallback = os.getenv("ALLOW_WAV2VEC2_BASE_FALLBACK", str(asr_config.get("allow_wav2vec2_base_fallback", False))).lower() in {"1", "true", "yes"}
    checks.append(f"Selected provider: {provider}")
    if provider not in VALID_PROVIDERS:
        errors.append(f"invalid ASR_PROVIDER: {provider}")
    try:
        import api.main  # noqa: F401

        checks.append("FastAPI app imports successfully")
    except Exception as exc:
        errors.append(f"FastAPI import failed: {exc}")
    try:
        import torch

        checks.append(f"torch installed: {torch.__version__}")
        cuda = bool(torch.cuda.is_available())
        checks.append(f"CUDA available: {cuda}")
        if device == "cuda" and not cuda:
            errors.append("ASR_DEVICE=cuda but CUDA is not available")
    except Exception as exc:
        errors.append(f"torch unavailable or failed to import: {exc}")
    for package in ("transformers", "soundfile", "librosa", "torchaudio", "rapidfuzz"):
        try:
            module = __import__(package)
            checks.append(f"{package} installed: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            errors.append(f"{package} unavailable or failed to import: {exc}")
    try:
        import g2p_en  # noqa: F401

        checks.append("g2p_en installed")
    except Exception as exc:
        warnings.append(f"g2p_en unavailable; expected phoneme generation will use CMUdict/custom letter dictionary first: {exc}")
    if provider != "mock":
        selected_asr_model = wav2vec2_model
        if not wav2vec2_model.exists():
            if allow_base_fallback and base_model.exists():
                selected_asr_model = base_model
                warnings.append(f"Fine-tuned Wav2Vec2 ASR model missing; using explicitly enabled base fallback: {base_model}")
            else:
                fallback_note = f"; base fallback missing: {base_model}" if allow_base_fallback else "; ALLOW_WAV2VEC2_BASE_FALLBACK=false"
                errors.append(f"Wav2Vec2 ASR model path missing: {wav2vec2_model}{fallback_note}")
        else:
            checks.append(f"Fine-tuned Wav2Vec2 ASR model exists: {wav2vec2_model}")
        if phoneme_model.exists():
            checks.append(f"Wav2Vec2 phoneme model exists: {phoneme_model}")
        else:
            errors.append(f"Wav2Vec2 phoneme model path missing: {phoneme_model}")
        _validate_wav2vec2_processor_load(selected_asr_model, "ASR", checks, errors)
        _validate_wav2vec2_processor_load(phoneme_model, "phoneme", checks, errors)
    for path in (
        Path("external_datasets/cmudict/cmudict.dict"),
        Path("external_datasets/cmudict/cmudict.phones"),
        Path("external_datasets/cmudict/cmudict.symbols"),
    ):
        if path.exists():
            checks.append(f"CMUdict file exists: {path}")
        else:
            errors.append(f"CMUdict file missing: {path}")
    for path in (Path("content_bank_enriched/enriched_content_index.csv"), Path("data/manifests/content_index.csv")):
        if path.exists():
            checks.append(f"Content metadata exists: {path}")
        else:
            warnings.append(f"Content metadata not found: {path}")
    for path in ("configs/service_config.yaml", "configs/analysis_config.yaml", "configs/adaptive_config.yaml"):
        if Path(path).exists():
            checks.append(f"Config exists: {path}")
        else:
            errors.append(f"Config missing: {path}")
    warnings.append("Speechocean762 and training JSONL files are NOT required for production runtime.")
    warnings.append("Whisper is not required and is not validated for Wav2Vec2-only runtime.")
    print("AI service startup validation")
    print("Checks:")
    for item in checks:
        print(f"- {item}")
    print("Warnings:")
    for item in warnings:
        print(f"- {item}")
    if errors:
        print("FAIL:")
        for item in errors:
            print(f"- {item}")
        return 1
    print("PASS: AI service runtime checks passed.")
    return 0


def _load_service_config() -> dict:
    path = Path("configs/service_config.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _validate_wav2vec2_processor_load(model_path: Path, label: str, checks: list[str], errors: list[str]) -> None:
    if not model_path.exists():
        return
    try:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

        if label == "phoneme":
            Wav2Vec2FeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
            Wav2Vec2CTCTokenizer.from_pretrained(str(model_path), local_files_only=True)
        else:
            Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
        checks.append(f"{label} Wav2Vec2 processor loads locally: {model_path}")
        Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
        checks.append(f"{label} Wav2Vec2 model loads locally: {model_path}")
    except Exception as exc:
        errors.append(f"{label} Wav2Vec2 local load failed for {model_path}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
