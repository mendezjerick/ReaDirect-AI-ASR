from readirect_asr.content.adaptive_tags import generate_adaptive_metadata


def test_error_focus_maps_to_recommended_error_type() -> None:
    assert generate_adaptive_metadata({}, {"error_focus": "final_consonant", "difficulty_level": "easy"})["recommended_for_error_type"] == "final_sound_error"
    assert generate_adaptive_metadata({}, {"error_focus": "vowel_sound", "difficulty_level": "easy"})["recommended_for_error_type"] == "vowel_error"
    assert generate_adaptive_metadata({}, {"error_focus": "sentence_tracking", "difficulty_level": "easy"})["recommended_for_error_type"] == "skipped_word"


def test_mastery_candidate_rules() -> None:
    result = generate_adaptive_metadata({"activity_type": "mastery_check"}, {"error_focus": "final_consonant", "difficulty_level": "easy", "needs_manual_review": False})
    assert result["mastery_candidate"] is True

