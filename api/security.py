from __future__ import annotations

import os

from fastapi import Header, HTTPException


def auth_enabled() -> bool:
    return os.getenv("API_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}


def validate_api_token(x_redirect_ai_token: str | None = Header(default=None, alias="X-ReaDirect-AI-Token")) -> None:
    if not auth_enabled():
        return
    expected = os.getenv("READIRECT_AI_API_TOKEN", "")
    if not expected or x_redirect_ai_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing ReaDirect AI API token.")
