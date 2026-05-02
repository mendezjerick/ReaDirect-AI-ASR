import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["ASR_PROVIDER"] = "mock"

from api.main import app
from api.service import AIAnalysisService
from readirect_asr.asr.mock_asr import MockASR


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ReaDirect AI/ASR Service"
    assert "asr_provider" in body
    assert body["reinforcement_corrections_enabled"] is True
    assert "letter-reinforcement.csv" in body["reinforcement_files_loaded"]
    assert body["reinforcement_letter_rules_count"] == 20
    assert body["audio_quality_validation_enabled"] is True
    assert body["pause_detection_enabled"] is True
    assert body["uncertainty_decision_enabled"] is True
    assert body["audio_quality_thresholds"]["min_duration_seconds"] == 1.0


def test_analyze_audio_endpoint_uses_mock_asr(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("api.main.get_service", lambda: AIAnalysisService(asr_provider=MockASR()))
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
