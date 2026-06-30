from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_analyze_content_item_returns_phoneme_metadata_for_cat() -> None:
    response = client.post(
        "/analyze-content-item",
        json={"prompt_id": "M2-001", "expected_text": "cat", "activity_type": "display_word_reading", "module_key": "module_2"},
    )
    assert response.status_code == 200
    enrichment = response.json()["enrichment_metadata"]
    assert enrichment["initial_phoneme"] == "K"
    assert enrichment["phoneme_pattern"] == "CVC"


def test_analyze_content_item_handles_missing_word_safely() -> None:
    response = client.post(
        "/analyze-content-item",
        json={"prompt_id": "M2-999", "expected_text": "zzzznotaword", "activity_type": "display_word_reading", "module_key": "module_2"},
    )
    assert response.status_code == 200
    assert response.json()["enrichment_metadata"]["needs_manual_review"] is True
