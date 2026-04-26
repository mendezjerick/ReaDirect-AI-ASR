from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_recommend_next_with_candidate_items():
    response = client.post(
        "/recommend-next",
        json={
            "learner_history": [
                {
                    "prompt_id": "M2-001",
                    "expected_text": "cat",
                    "actual_text": "cap",
                    "is_correct": False,
                    "error_type": "final_sound_error",
                    "skill_signal": "final_consonant",
                    "target_phoneme": "T",
                    "difficulty_level": "easy",
                }
            ],
            "candidate_items": [
                {
                    "prompt_id": "M2-014",
                    "module_key": "module_2",
                    "activity_type": "read_word",
                    "prompt_text": "Read the word.",
                    "expected_text": "hat",
                    "error_focus": "final_consonant",
                    "target_phoneme": "T",
                    "difficulty_level": "easy",
                    "is_active": True,
                    "needs_manual_review": False,
                }
            ],
            "top_k": 5,
            "debug": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["selected_item"]["prompt_id"] == "M2-014"
    assert data["recommendation"]["primary_focus"] == "final_consonant"
    assert data["explanation"]["learner_safe_summary"]


def test_recommend_next_empty_history_baseline():
    response = client.post(
        "/recommend-next",
        json={
            "learner_history": [],
            "candidate_items": [
                {
                    "prompt_id": "BASE",
                    "expected_text": "cat",
                    "error_focus": "vowel_sound",
                    "difficulty_level": "easy",
                    "is_active": True,
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["learner_summary"]["total_attempts"] == 0


def test_recommend_next_uses_repository_when_available():
    response = client.post(
        "/recommend-next",
        json={
            "learner_history": [
                {
                    "is_correct": False,
                    "error_type": "final_sound_error",
                    "skill_signal": "final_consonant",
                    "difficulty_level": "easy",
                }
            ],
            "current_context": {"module_key": "module_2"},
        },
    )
    assert response.status_code == 200
    assert "selected_item" in response.json()
