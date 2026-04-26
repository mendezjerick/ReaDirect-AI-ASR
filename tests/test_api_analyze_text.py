from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_analyze_text_returns_structured_result() -> None:
    response = client.post(
        "/analyze-text",
        json={"expected_text": "cat", "actual_text": "cap", "accepted_answers": ["cat"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["expected_text"] == "cat"
    assert body["transcript"] == "cap"
    assert body["is_correct"] is False
    assert body["similarity_label"] == "very_close"


def test_analyze_text_accepted_answer_and_blank() -> None:
    accepted = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "kitty", "accepted_answers": ["kitty"]})
    blank = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "", "accepted_answers": []})
    assert accepted.json()["error_type"] == "accepted_variant"
    assert blank.json()["error_type"] == "blank"
