import json

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_analyze_text_response_has_required_laravel_fields():
    response = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "cap", "accepted_answers": ["cat"]})
    assert response.status_code == 200
    data = response.json()
    required = {"ok", "request_id", "transcript", "normalized_transcript", "raw_transcript", "corrected_transcript", "displayed_transcript", "raw_wer", "corrected_wer", "phonetic_similarity_score", "normalization_applied", "normalization_reason", "correction_strategy_used", "accepted_by_phonetic_threshold", "threshold_used", "confidence_or_threshold_used", "provider", "expected_text", "is_correct", "similarity_label", "character_similarity", "token_similarity", "expected_phonemes", "actual_phonemes", "phoneme_similarity", "error_type", "feedback_hint", "coach_hint_key", "learner_safe_summary", "skill_signal", "recommended_practice_focus", "warnings", "error"}
    assert required.issubset(data.keys())
    json.dumps(data)


def test_recommend_next_response_has_required_laravel_fields():
    response = client.post(
        "/recommend-next",
        json={"learner_history": [{"is_correct": False, "error_type": "final_sound_error", "skill_signal": "final_consonant"}], "candidate_items": [{"prompt_id": "M2-014", "expected_text": "hat", "error_focus": "final_consonant", "difficulty_level": "easy", "is_active": True}]},
    )
    assert response.status_code == 200
    data = response.json()
    required = {"ok", "selected_item", "ranked_candidates", "learner_summary", "recommendation", "explanation", "warnings"}
    assert required.issubset(data.keys())
    json.dumps(data)
