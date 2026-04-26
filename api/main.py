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
    VersionResponse,
)
from api.security import validate_api_token

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
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        asr_provider=service.provider_name,
        content_index_loaded=service.content_repository.is_loaded(),
        cmudict_loaded=not bool(service.cmudict_loader.missing_files()),
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
            },
            "analysis": {
                "content_index_loaded": service.content_repository.is_loaded(),
                "content_index_path": str(service.content_repository.loaded_path) if service.content_repository.loaded_path else "",
            },
        },
    )


@app.post("/analyze-text", response_model=AnalysisResponse, dependencies=[Depends(validate_api_token)])
def analyze_text(request: AnalyzeTextRequest) -> AnalysisResponse:
    return get_service().analyze_text(request)


@app.post("/analyze-audio", response_model=AnalysisResponse, dependencies=[Depends(validate_api_token)])
def analyze_audio(request: AnalyzeAudioRequest) -> AnalysisResponse:
    return get_service().analyze_audio(request)


@app.post("/content-item", response_model=ContentItemResponse, dependencies=[Depends(validate_api_token)])
def content_item(request: ContentItemRequest) -> ContentItemResponse:
    return get_service().get_content_item(request)


@app.post("/analyze-content-item", response_model=ContentItemResponse, dependencies=[Depends(validate_api_token)])
def analyze_content_item(request: ContentItemRequest) -> ContentItemResponse:
    return get_service().get_content_item(request)


@app.post("/recommend-next", response_model=RecommendNextResponse, dependencies=[Depends(validate_api_token)])
def recommend_next(request: RecommendNextRequest) -> RecommendNextResponse:
    return get_service().recommend_next(request)
