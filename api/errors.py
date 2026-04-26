from __future__ import annotations

import os
import traceback
import uuid
from typing import Any


def new_request_id() -> str:
    return str(uuid.uuid4())


def api_debug_enabled() -> bool:
    return os.getenv("API_DEBUG", "true").lower() in {"1", "true", "yes"}


def safe_error_response(
    code: str,
    message: str,
    request_id: str | None = None,
    debug_error: Exception | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": False,
        "request_id": request_id or new_request_id(),
        "error": code,
        "warnings": [message],
    }
    if debug_error and api_debug_enabled():
        response["debug_info"] = {
            "exception": str(debug_error),
            "traceback": traceback.format_exc(),
        }
    return response

