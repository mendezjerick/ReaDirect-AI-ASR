import json

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_analyze_text_response_has_required_laravel_fields():
    response = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "cap", "accepted_answers": ["cat"]})
    assert response.status_code == 200
    data = response.json()
    required = {"ok", "request_id", "transcript", "normalized_transcript", "raw_transcript", "corrected_transcript", "displayed_transcript", "raw_wer", "corrected_wer", "phonetic_similarity_score", "normalization_applied", "normalization_reason", "correction_strategy_used", "accepted_by_phonetic_threshold", "accepted_by_reinforcement_match", "reinforcement_source_file", "reinforcement_expected_label", "reinforcement_matched_transcript", "threshold_used", "confidence_or_threshold_used", "provider", "expected_text", "is_correct", "similarity_label", "character_similarity", "token_similarity", "expected_phonemes", "actual_phonemes", "phoneme_similarity", "error_type", "feedback_hint", "coach_hint_key", "learner_safe_summary", "skill_signal", "recommended_practice_focus", "audio_quality", "pause_metrics", "uncertain", "retry_required", "uncertainty_reasons", "quality_gate_failed", "learner_retry_message", "gop_enabled", "gop_available", "gop_score", "gop_decision", "gop_threshold", "gop_expected_phonemes", "gop_observed_phonemes", "gop_word_scores", "gop_correction_applied", "gop_error", "dynamic_correction_enabled", "dynamic_correction_applied", "dynamic_correction_strategy", "dynamic_correction_sub_strategy", "dynamic_correction_confidence", "dynamic_correction_threshold", "dynamic_spelling_similarity", "dynamic_phoneme_similarity", "dynamic_gop_score", "dynamic_homophone_match", "dynamic_context_score", "dynamic_correction_reason", "word_alignment", "debug_metadata", "warnings", "error"}
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
