from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_analyze_text_still_works_without_history():
    response = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "cap"})
    assert response.status_code == 200
    data = response.json()
    assert data["error_type"] == "final_sound_error"
    assert data["adaptive_recommendation"] is None


def test_analyze_text_with_history_includes_adaptive_recommendation():
    response = client.post(
        "/analyze-text",
        json={
            "expected_text": "cat",
            "actual_text": "cap",
            "learner_history": [
                {
                    "is_correct": False,
                    "error_type": "final_sound_error",
                    "skill_signal": "final_consonant",
                    "difficulty_level": "easy",
                }
            ],
            "candidate_items": [
                {
                    "prompt_id": "M2-014",
                    "expected_text": "hat",
                    "error_focus": "final_consonant",
                    "target_phoneme": "T",
                    "difficulty_level": "easy",
                    "is_active": True,
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["adaptive_recommendation"]["primary_focus"] == "final_consonant"
    assert data["learner_summary"]["total_attempts"] == 2
    assert "error_type" in data
