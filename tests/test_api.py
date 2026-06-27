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
    assert body["status"] == "healthy"
    assert body["service_status"] == "healthy"
    assert body["service"] == "ReaDirect AI/ASR Service"
    assert "asr_provider" in body
    assert body["reinforcement_corrections_enabled"] is True
    assert "letter-reinforcement.csv" in body["reinforcement_files_loaded"]
    assert body["reinforcement_letter_rules_count"] == 20
    assert body["audio_quality_validation_enabled"] is True
    assert body["pause_detection_enabled"] is True
    assert body["uncertainty_decision_enabled"] is True
    assert body["audio_quality_thresholds"]["min_duration_seconds"] == 1.0


def test_live_endpoint_is_lightweight_and_does_not_resolve_service(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("/live must not resolve the ASR service")

    monkeypatch.setattr("api.main.get_service", fail_if_called)

    response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "service": "ReaDirect AI/ASR Service",
    }


def test_ready_endpoint_reports_mock_provider_ready() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["service"] == "ReaDirect AI/ASR Service"
    assert body["provider"] == "mock"
    assert body["model_loaded"] is True
    assert body["processor_loaded"] is True
    assert body["local_model_paths_loaded"] is True
    assert body["missing_model_paths_count"] == 0


def test_ready_endpoint_does_not_call_inference(monkeypatch) -> None:
    class TrackingASR(MockASR):
        transcribe_called = False

        def transcribe(self, audio_path: str, **kwargs):
            self.transcribe_called = True
            raise AssertionError("/ready must not call ASR inference")

    class StubService:
        def __init__(self, asr_provider):
            self.asr_provider = asr_provider

        @property
        def provider_name(self) -> str:
            return self.asr_provider.provider

    provider = TrackingASR()
    monkeypatch.setattr("api.main.get_service", lambda: StubService(provider))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert provider.transcribe_called is False


def test_ready_endpoint_returns_503_without_exposing_internal_paths(monkeypatch) -> None:
    class NotReadyProvider:
        provider = "wav2vec2_only"

        def status(self):
            return {
                "asr_model_loaded": False,
                "processor_loaded": False,
                "missing_model_paths": ["C:/private/readirect/models/asr/epsilon"],
                "warnings": ["model is not loaded"],
            }

    class StubService:
        asr_provider = NotReadyProvider()
        provider_name = "wav2vec2_only"

    monkeypatch.setattr("api.main.get_service", lambda: StubService())

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["provider"] == "wav2vec2_only"
    assert body["model_loaded"] is False
    assert body["processor_loaded"] is False
    assert body["local_model_paths_loaded"] is False
    assert body["missing_model_paths_count"] == 1
    assert "missing_model_paths" not in body
    assert "C:/private/readirect/models/asr/epsilon" not in response.text


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
