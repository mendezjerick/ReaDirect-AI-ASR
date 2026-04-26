from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.security import validate_api_token


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(validate_api_token)])
    def protected():
        return {"ok": True}

    return app


def test_auth_disabled_allows_request(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_ENABLED", "false")
    response = TestClient(_app()).get("/protected")
    assert response.status_code == 200


def test_auth_enabled_rejects_missing_token(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("READIRECT_AI_API_TOKEN", "secret")
    response = TestClient(_app()).get("/protected")
    assert response.status_code == 401


def test_auth_enabled_accepts_valid_token(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("READIRECT_AI_API_TOKEN", "secret")
    response = TestClient(_app()).get("/protected", headers={"X-ReaDirect-AI-Token": "secret"})
    assert response.status_code == 200

