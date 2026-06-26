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
    model_name = os.getenv("ASR_MODEL_NAME", str(asr_config.get("model_name", "epsilon")))
    wav2vec2_model = Path(os.getenv("ASR_MODEL_PATH", os.getenv("WAV2VEC2_ASR_MODEL_PATH", str(asr_config.get("wav2vec2_asr_model_path", "models/asr/epsilon")))))
    phoneme_model = Path(os.getenv("WAV2VEC2_PHONEME_MODEL_PATH", str(asr_config.get("wav2vec2_phoneme_model_path", "models/wav2vec2-phoneme"))))
    base_model = Path(os.getenv("WAV2VEC2_BASE_ASR_MODEL_PATH", str(asr_config.get("wav2vec2_base_asr_model_path", "models/wav2vec2-readirect-asr"))))
    allow_base_fallback = os.getenv("ALLOW_WAV2VEC2_BASE_FALLBACK", str(asr_config.get("allow_wav2vec2_base_fallback", False))).lower() in {"1", "true", "yes"}
    decode_mode = os.getenv("ASR_DECODE_MODE", str(asr_config.get("decode_mode", "beam_lm")))
    lm_path = Path(os.getenv("ASR_LM_PATH", str(asr_config.get("lm_path", "external_datasets/language_models/3-gram.pruned.1e-7.arpa"))))
    allow_no_lm_fallback = os.getenv("ASR_ALLOW_NO_LM_FALLBACK", str(asr_config.get("allow_no_lm_fallback", False))).lower() in {"1", "true", "yes"}
    checks.append(f"Selected ASR model: {model_name}")
    checks.append(f"Selected decode mode: {decode_mode}")
    checks.append(f"Selected provider: {provider}")
    if provider not in VALID_PROVIDERS:
        errors.append(f"invalid ASR_PROVIDER: {provider}")
    try:
        import api.main as api_main

        getattr(api_main, "__name__", None)
        checks.append("FastAPI app imports successfully")
    except Exception as exc:
        errors.append(f"FastAPI import failed: {exc}")
    try:
        import torch

        checks.append(f"torch installed: {torch.__version__}")
        cuda = bool(torch.cuda.is_available())
        checks.append(f"CUDA available: {cuda}")
        if cuda:
            checks.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
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
        import g2p_en

        getattr(g2p_en, "__name__", None)
        checks.append("g2p_en installed")
    except Exception as exc:
        warnings.append(f"g2p_en unavailable; expected phoneme generation will use CMUdict/custom letter dictionary first: {exc}")
    if provider != "mock":
        if model_name.lower() != "epsilon":
            errors.append(f"ASR_MODEL_NAME must be epsilon for deployment, got {model_name}")
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
        _validate_huggingface_files(selected_asr_model, "ASR", checks, errors)
        _validate_huggingface_files(phoneme_model, "phoneme", checks, errors)
        _validate_wav2vec2_processor_load(selected_asr_model, "ASR", checks, errors)
        _validate_wav2vec2_processor_load(phoneme_model, "phoneme", checks, errors)
        if decode_mode == "beam_lm":
            if lm_path.exists():
                checks.append(f"KenLM language model exists: {lm_path}")
            elif allow_no_lm_fallback:
                warnings.append(f"KenLM language model missing; explicit no-LM fallback is enabled: {lm_path}")
            else:
                errors.append(f"ASR_DECODE_MODE=beam_lm but language model is missing: {lm_path}")
        try:
            from api.dependencies import get_asr_provider

            get_asr_provider().warmup()
            checks.append("Epsilon model, processor, and decoder loaded successfully")
        except Exception as exc:
            errors.append(f"ASR startup warmup failed: {exc}")
        _validate_health_status(wav2vec2_model, phoneme_model, checks, errors)
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


def _validate_huggingface_files(model_path: Path, label: str, checks: list[str], errors: list[str]) -> None:
    if not model_path.exists():
        return
    required = ["config.json"]
    for filename in required:
        if (model_path / filename).exists():
            checks.append(f"{label} model file exists: {model_path / filename}")
        else:
            errors.append(f"{label} model file missing: {model_path / filename}")
    if any((model_path / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")):
        checks.append(f"{label} model weights exist: {model_path}")
    else:
        errors.append(f"{label} model weights missing: expected model.safetensors or pytorch_model.bin in {model_path}")
    if any((model_path / filename).exists() for filename in ("preprocessor_config.json", "processor_config.json")):
        checks.append(f"{label} processor/preprocessor config exists: {model_path}")
    else:
        errors.append(f"{label} processor/preprocessor config missing in {model_path}")
    if (model_path / "vocab.json").exists() or (model_path / "tokenizer.json").exists():
        checks.append(f"{label} tokenizer files exist: {model_path}")
    else:
        errors.append(f"{label} tokenizer files missing: expected vocab.json or tokenizer.json in {model_path}")
    if (model_path / "readirect_model_metadata.json").exists():
        checks.append(f"{label} ReaDirect model metadata exists: {model_path / 'readirect_model_metadata.json'}")


def _validate_health_status(wav2vec2_model: Path, phoneme_model: Path, checks: list[str], errors: list[str]) -> None:
    try:
        from fastapi.testclient import TestClient

        import api.main as api_main

        response = TestClient(api_main.app).get("/health")
        if response.status_code != 200:
            errors.append(f"health/status endpoint returned HTTP {response.status_code}")
            return
        body = response.json()
    except Exception as exc:
        errors.append(f"health/status endpoint check failed: {exc}")
        return
    expected = {
        "asr_architecture": "wav2vec2_only",
        "active_asr_model": "wav2vec2",
        "wav2vec2_asr_available": True,
        "wav2vec2_asr_model_name": wav2vec2_model.as_posix(),
        "wav2vec2_phoneme_available": True,
        "wav2vec2_phoneme_model_name": phoneme_model.as_posix(),
        "whisper_removed": True,
        "correction_layer_enabled": True,
        "expected_centric_scoring_enabled": True,
        "phoneme_evidence_enabled": True,
        "reinforcement_corrections_enabled": True,
        "audio_quality_validation_enabled": True,
        "pause_detection_enabled": True,
        "uncertainty_decision_enabled": True,
    }
    for key, expected_value in expected.items():
        actual = body.get(key)
        if actual == expected_value:
            checks.append(f"health/status {key}: {actual}")
        else:
            errors.append(f"health/status {key} expected {expected_value!r}, got {actual!r}")
    if body.get("model_version"):
        checks.append(f"health/status model_version: {body['model_version']}")
    if body.get("training_type"):
        checks.append(f"health/status training_type: {body['training_type']}")
    if body.get("training_mix"):
        checks.append(f"health/status training_mix: {body['training_mix']}")
    decoder_expected = {
        "asr_model_name": "epsilon",
        "asr_model_loaded": True,
        "processor_loaded": True,
        "decode_mode": os.getenv("ASR_DECODE_MODE", "beam_lm"),
        "beam_search_enabled": True,
    }
    for key, expected_value in decoder_expected.items():
        if body.get(key) == expected_value:
            checks.append(f"health/status {key}: {body[key]}")
        else:
            errors.append(f"health/status {key} expected {expected_value!r}, got {body.get(key)!r}")
    if decoder_expected["decode_mode"] == "beam_lm" and body.get("language_model_loaded") is not True:
        errors.append("health/status language_model_loaded must be true for beam_lm")


if __name__ == "__main__":
    raise SystemExit(main())
