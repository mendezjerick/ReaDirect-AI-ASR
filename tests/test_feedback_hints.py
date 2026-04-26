from readirect_asr.scoring.feedback_hints import generate_feedback_hint


def test_error_types_map_to_safe_feedback() -> None:
    for error_type in ["correct", "accepted_variant", "blank", "initial_sound_error", "final_sound_error", "vowel_error", "skipped_word", "partial_sentence", "far_answer", "unclear_asr", "incorrect_general"]:
        result = generate_feedback_hint(error_type)
        assert result["coach_hint_key"]
        assert "bad" not in str(result["learner_safe_summary"]).lower()
        assert "wrong" not in str(result["learner_safe_summary"]).lower()

