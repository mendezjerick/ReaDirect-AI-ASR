from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_version_and_content_item_endpoints() -> None:
    version = client.get("/version")
    content = client.post("/content-item", json={"expected_text": "cat", "module_key": "module_2", "activity_type": "display_word_reading"})
    assert version.status_code == 200
    assert version.json()["service"] == "ReaDirect AI/ASR Service"
    assert content.status_code == 200
    assert content.json()["ok"] is True


def test_analyze_text_endpoint_contract() -> None:
    response = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "cap"})
    assert response.status_code == 200
    assert response.json()["mode"] == "text"


def test_analyze_audio_missing_file_safe() -> None:
    response = client.post("/analyze-audio", json={"audio_path": "missing.wav", "expected_text": "cat"})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["error"] == "audio_file_not_found"


def test_reinforcement_corrections_endpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("api.main.config", {"transcript_normalization": {"reinforcement_corrections_dir": str(tmp_path)}})

    response = client.post(
        "/reinforcement/corrections",
        json={
            "expected_text": "Leo",
            "raw_transcript": "Layo",
            "prompt_type": "word",
            "accepted": False,
            "created_by": "admin",
            "source": "developer_auto",
        },
    )
    duplicate = client.post(
        "/reinforcement/corrections",
        json={
            "expected_text": "Leo",
            "raw_transcript": "Layo",
            "prompt_type": "word",
            "accepted": False,
            "created_by": "admin",
            "source": "developer_auto",
        },
    )

    assert response.status_code == 200
    assert response.json()["saved"] is True
    assert response.json()["target_file"] == "word-reinforcement.csv"
    assert duplicate.json()["saved"] is False
    assert duplicate.json()["duplicate"] is True
