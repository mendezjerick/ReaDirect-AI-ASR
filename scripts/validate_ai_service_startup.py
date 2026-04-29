from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


VALID_PROVIDERS = {"mock", "faster_whisper_pretrained", "faster_whisper", "faster-whisper", "faster_whisper_local", "hf_whisper_local"}


def main() -> int:
    checks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    service_config = _load_service_config()
    asr_config = service_config.get("asr", {})
    provider = os.getenv("ASR_PROVIDER", str(asr_config.get("provider", "mock")))
    device = os.getenv("ASR_DEVICE", str(asr_config.get("device", "cpu")))
    hf_model = Path(os.getenv("ASR_HF_MODEL_PATH", str(asr_config.get("hf_model_path", "model_artifacts/readirect-whisper-base-en-v1-hf"))))
    ct2_model = Path(os.getenv("ASR_CT2_MODEL_PATH", str(asr_config.get("ct2_model_path", "model_artifacts/readirect-whisper-base-en-v1-ct2"))))
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
        warnings.append(f"torch unavailable or failed to import: {exc}")
    if provider == "hf_whisper_local" and not hf_model.exists():
        errors.append(f"HF model path missing: {hf_model}")
    if provider == "faster_whisper_local" and not ct2_model.exists():
        errors.append(f"CTranslate2 model path missing: {ct2_model}")
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


if __name__ == "__main__":
    raise SystemExit(main())
