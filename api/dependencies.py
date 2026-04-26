from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from api.service import AIAnalysisService
from readirect_asr.asr.provider_factory import create_asr_provider
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    config = _load_yaml("configs/service_config.yaml")
    config.setdefault("api", {})
    config.setdefault("analysis", {})
    config.setdefault("asr", {})
    adaptive_config_path = os.getenv("ADAPTIVE_CONFIG_PATH", "configs/adaptive_config.yaml")
    config["adaptive"] = {**_load_yaml(adaptive_config_path), **config.get("adaptive", {})}
    config["api"]["debug"] = os.getenv("API_DEBUG", str(config["api"].get("debug", True))).lower() in {"1", "true", "yes"}
    config["api"]["auth_enabled"] = os.getenv("API_AUTH_ENABLED", str(config["api"].get("auth_enabled", False))).lower() in {"1", "true", "yes"}
    origins = os.getenv("CORS_ALLOW_ORIGINS")
    if origins:
        config["api"]["cors_allow_origins"] = [origin.strip() for origin in origins.split(",") if origin.strip()]
    config["analysis"]["content_index_path"] = os.getenv("CONTENT_INDEX_PATH", config["analysis"].get("content_index_path", "data/manifests/content_index.csv"))
    config["analysis"]["enriched_content_index_path"] = os.getenv("ENRICHED_CONTENT_INDEX_PATH", config["analysis"].get("enriched_content_index_path", "content_bank_enriched/enriched_content_index.csv"))
    config["asr"]["provider"] = os.getenv("ASR_PROVIDER", config["asr"].get("provider", "mock"))
    config["asr"]["model_size"] = os.getenv("ASR_MODEL_SIZE", config["asr"].get("model_size", "base.en"))
    config["asr"]["pretrained_model_size"] = os.getenv("ASR_PRETRAINED_MODEL_SIZE", config["asr"].get("pretrained_model_size", config["asr"].get("model_size", "base.en")))
    config["asr"]["hf_model_path"] = os.getenv("ASR_HF_MODEL_PATH", config["asr"].get("hf_model_path", "model_artifacts/readirect-whisper-base-en-v1-hf"))
    config["asr"]["ct2_model_path"] = os.getenv("ASR_CT2_MODEL_PATH", config["asr"].get("ct2_model_path", "model_artifacts/readirect-whisper-base-en-v1-ct2"))
    config["asr"]["device"] = os.getenv("ASR_DEVICE", config["asr"].get("device", "cpu"))
    config["asr"]["compute_type"] = os.getenv("ASR_COMPUTE_TYPE", config["asr"].get("compute_type", "int8"))
    config["asr"]["use_fp16"] = os.getenv("ASR_USE_FP16", str(config["asr"].get("use_fp16", False))).lower() in {"1", "true", "yes"}
    config["asr"]["language"] = os.getenv("ASR_LANGUAGE", config["asr"].get("language", "en"))
    config["asr"]["task"] = os.getenv("ASR_TASK", config["asr"].get("task", "transcribe"))
    config["asr"]["beam_size"] = int(os.getenv("ASR_BEAM_SIZE", str(config["asr"].get("beam_size", 1))))
    return config


@lru_cache(maxsize=1)
def get_cmudict_loader() -> CMUDictLoader:
    cmudict_dir = Path(os.getenv("CMUDICT_DIR", "external_datasets/cmudict"))
    return CMUDictLoader(
        cmudict_dir / "cmudict.dict",
        cmudict_dir / "cmudict.phones",
        cmudict_dir / "cmudict.symbols",
    ).load()


@lru_cache(maxsize=1)
def get_content_repository() -> ContentRepository:
    config = get_config()
    analysis = config.get("analysis", {})
    return ContentRepository(
        content_index_path=analysis.get("content_index_path", "data/manifests/content_index.csv"),
        enriched_content_index_path=analysis.get("enriched_content_index_path", "content_bank_enriched/enriched_content_index.csv"),
        prefer_enriched_content=bool(analysis.get("prefer_enriched_content", True)),
    ).load()


@lru_cache(maxsize=1)
def get_asr_provider():
    return create_asr_provider(get_config().get("asr", {}))


@lru_cache(maxsize=1)
def get_service() -> AIAnalysisService:
    return AIAnalysisService(
        asr_provider=get_asr_provider(),
        cmudict_loader=get_cmudict_loader(),
        content_repository=get_content_repository(),
        config=get_config(),
    )
