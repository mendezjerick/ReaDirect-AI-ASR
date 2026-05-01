import json

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


EXPECTED_FIELDS = {
    "ok",
    "request_id",
    "mode",
    "provider",
    "model_size",
    "prompt_id",
    "expected_text",
    "accepted_answers",
    "transcript",
    "normalized_transcript",
    "raw_transcript",
    "corrected_transcript",
    "displayed_transcript",
    "raw_wer",
    "corrected_wer",
    "phonetic_similarity_score",
    "normalization_applied",
    "normalization_reason",
    "correction_strategy_used",
    "accepted_by_phonetic_threshold",
    "accepted_by_known_confusion",
    "accepted_by_letter_lattice",
    "accepted_by_letter_normalization",
    "accepted_by_exact_match",
    "threshold_used",
    "confidence_or_threshold_used",
    "confidence",
    "is_correct",
    "is_exact",
    "is_accepted",
    "character_similarity",
    "token_similarity",
    "similarity_label",
    "expected_phonemes",
    "actual_phonemes",
    "phoneme_similarity",
    "error_type",
    "error_position",
    "feedback_hint",
    "coach_hint_key",
    "learner_safe_summary",
    "skill_signal",
    "target_phoneme",
    "target_position",
    "recommended_practice_focus",
    "recommended_action",
    "content_metadata",
    "enrichment_metadata",
    "analysis_source",
    "warnings",
    "debug_info",
    "processing_seconds",
    "error",
}


def test_laravel_contract_fields_and_json_serializable() -> None:
    response = client.post("/analyze-text", json={"expected_text": "cat", "actual_text": "cap", "debug": True})
    body = response.json()
    assert EXPECTED_FIELDS.issubset(body.keys())
    json.dumps(body)
    assert body["debug_info"] is not None
