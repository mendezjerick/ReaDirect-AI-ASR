import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["ASR_PROVIDER"] = "mock"

from api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ReaDirect AI/ASR Service"
    assert "asr_provider" in body


def test_analyze_audio_endpoint_uses_mock_asr(tmp_path: Path) -> None:
    audio_path = tmp_path / "example.wav"
    audio_path.write_bytes(b"fake")
    response = client.post(
        "/analyze-audio",
        json={
            "audio_path": str(audio_path),
            "expected_text": "cat",
            "accepted_answers": ["cat"],
            "prompt_id": "M2-001",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["transcript"] == "cat"
    assert body["normalized_transcript"] == "cat"
    assert body["expected_text"] == "cat"
    assert body["is_correct"] is True
    assert body["similarity_label"] == "exact"
    assert body["error_type"] == "correct"
    assert body["confidence"] is None
    assert body["provider"] == "mock"
