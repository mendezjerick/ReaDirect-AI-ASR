from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.dependencies import get_config, get_service
from api.schemas import (
    AnalysisResponse,
    AnalyzeAudioRequest,
    AnalyzeTextRequest,
    ContentItemRequest,
    ContentItemResponse,
    HealthResponse,
    RecommendNextRequest,
    RecommendNextResponse,
    ReinforcementCorrectionRequest,
    ReinforcementCorrectionResponse,
    VersionResponse,
)
from api.security import validate_api_token
from readirect_asr.audio.preprocessing import audio_quality_config
from readirect_asr.text.reinforcement_corrections import append_developer_correction, reinforcement_status_from_config

SERVICE_NAME = "ReaDirect AI/ASR Service"
SERVICE_VERSION = "0.1.0"

logging.basicConfig(level=logging.INFO)

config = get_config()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_allow_origins", ["http://127.0.0.1:8000", "http://localhost:8000"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    provider_status = service.asr_provider.status() if hasattr(service.asr_provider, "status") else {}
    missing_paths = list(provider_status.get("missing_model_paths", []) or [])
    architecture = str(provider_status.get("asr_architecture", "wav2vec2_only" if service.provider_name != "mock" else "mock"))
    service_status = "ok" if service.provider_name == "mock" or not missing_paths else "degraded"
    reinforcement_status = reinforcement_status_from_config(config.get("transcript_normalization", {}))
    qa_config = audio_quality_config(config.get("audio_quality", {}))
    return HealthResponse(
        status=service_status,
        service_status=service_status,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        asr_provider=service.provider_name,
        content_index_loaded=service.content_repository.is_loaded(),
        cmudict_loaded=not bool(service.cmudict_loader.missing_files()),
        asr_architecture=architecture,
        active_asr_model=str(provider_status.get("active_asr_model", "wav2vec2" if service.provider_name != "mock" else "mock")),
        active_asr_model_path=str(provider_status.get("active_asr_model_path", "")),
        wav2vec2_asr_available=bool(provider_status.get("wav2vec2_asr_available", service.provider_name == "mock")),
        wav2vec2_asr_model_name=str(provider_status.get("wav2vec2_asr_model_name", config.get("asr", {}).get("wav2vec2_asr_model_path", ""))),
        wav2vec2_phoneme_available=bool(provider_status.get("wav2vec2_phoneme_available", service.provider_name == "mock")),
        wav2vec2_phoneme_model_name=str(provider_status.get("wav2vec2_phoneme_model_name", config.get("asr", {}).get("wav2vec2_phoneme_model_path", ""))),
        whisper_available=False,
        whisper_removed=True,
        model_version=str(provider_status.get("model_version", "")),
        base_model=str(provider_status.get("base_model", "")),
        training_type=str(provider_status.get("training_type", "")),
        training_mix=str(provider_status.get("training_mix", "")),
        thresholds={
            **config.get("transcript_normalization", {}),
            "gop": config.get("gop", {}),
            "dynamic_expected_correction": config.get("dynamic_expected_correction", {}),
        },
        local_model_paths_loaded=not missing_paths,
        missing_model_paths=missing_paths,
        audio_quality_validation_enabled=True,
        audio_quality_thresholds={
            "min_duration_seconds": qa_config["min_duration_seconds"],
            "low_volume_dbfs": qa_config["low_volume_dbfs"],
            "mostly_silent_ratio": qa_config["mostly_silent_ratio"],
            "clipping_threshold": qa_config["clipping_threshold"],
            "clipped_ratio_threshold": qa_config["clipped_ratio_threshold"],
            "long_pause_seconds": qa_config["long_pause_seconds"],
        },
        pause_detection_enabled=True,
        uncertainty_decision_enabled=True,
        **reinforcement_status,
    )


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    service = get_service()
    return VersionResponse(
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        asr_provider=service.provider_name,
        config={
            "api": {
                "debug": config.get("api", {}).get("debug"),
                "auth_enabled": config.get("api", {}).get("auth_enabled"),
            },
            "asr": {
                "provider": service.provider_name,
                "model_size": service.model_size,
                "asr_architecture": "wav2vec2_only",
                "model_family": "wav2vec2",
                "wav2vec2_asr_model_path": config.get("asr", {}).get("wav2vec2_asr_model_path"),
                "wav2vec2_phoneme_model_path": config.get("asr", {}).get("wav2vec2_phoneme_model_path"),
                "allow_wav2vec2_base_fallback": config.get("asr", {}).get("allow_wav2vec2_base_fallback"),
                "whisper_removed": True,
            },
            "analysis": {
                "content_index_loaded": service.content_repository.is_loaded(),
                "content_index_path": str(service.content_repository.loaded_path) if service.content_repository.loaded_path else "",
            },
            "transcript_normalization": config.get("transcript_normalization", {}),
            "gop": config.get("gop", {}),
            "dynamic_expected_correction": config.get("dynamic_expected_correction", {}),
        },
    )


@app.post("/analyze-text", response_model=AnalysisResponse, dependencies=[Depends(validate_api_token)])
def analyze_text(request: AnalyzeTextRequest) -> AnalysisResponse:
    return get_service().analyze_text(request)


@app.post("/analyze-audio", response_model=AnalysisResponse, dependencies=[Depends(validate_api_token)])
def analyze_audio(request: AnalyzeAudioRequest) -> AnalysisResponse:
    return get_service().analyze_audio(request)


@app.post("/reinforcement/corrections", response_model=ReinforcementCorrectionResponse, dependencies=[Depends(validate_api_token)])
def reinforcement_correction(request: ReinforcementCorrectionRequest) -> ReinforcementCorrectionResponse:
    active_config = config.get("transcript_normalization", {})
    result = append_developer_correction(
        expected_text=request.expected_text,
        raw_transcript=request.raw_transcript,
        prompt_type=request.prompt_type,
        accepted=request.accepted,
        retry_required=request.retry_required,
        uncertain=request.uncertain,
        correction_strategy_used=request.correction_strategy_used,
        developer_reinforcement_enabled=request.developer_reinforcement_enabled,
        developer_user_role=request.developer_user_role or request.created_by,
        created_by=request.created_by,
        source=request.source,
        notes=request.notes,
        corrections_dir=active_config.get("reinforcement_corrections_dir", "reinforcement-learning"),
        letter_file=active_config.get("letter_reinforcement_file", "letter-reinforcement.csv"),
        word_file=active_config.get("word_reinforcement_file", "word-reinforcement.csv"),
    )
    return ReinforcementCorrectionResponse(**result)


@app.post("/content-item", response_model=ContentItemResponse, dependencies=[Depends(validate_api_token)])
def content_item(request: ContentItemRequest) -> ContentItemResponse:
    return get_service().get_content_item(request)


@app.post("/analyze-content-item", response_model=ContentItemResponse, dependencies=[Depends(validate_api_token)])
def analyze_content_item(request: ContentItemRequest) -> ContentItemResponse:
    return get_service().get_content_item(request)


@app.post("/recommend-next", response_model=RecommendNextResponse, dependencies=[Depends(validate_api_token)])
def recommend_next(request: RecommendNextRequest) -> RecommendNextResponse:
    return get_service().recommend_next(request)
