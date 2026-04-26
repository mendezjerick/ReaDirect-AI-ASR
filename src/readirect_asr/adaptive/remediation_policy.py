from __future__ import annotations


ERROR_FOCUS_MAP = {
    "final_sound_error": {
        "primary_focus": "final_consonant",
        "target_position": "final",
        "preferred_error_focus": ["final_consonant"],
        "preferred_activity_types": ["read_word", "word_accuracy_challenge"],
        "recommended_action": "remediate",
        "reason_code": "target_final_sound_error",
    },
    "initial_sound_error": {
        "primary_focus": "initial_consonant",
        "target_position": "initial",
        "preferred_error_focus": ["initial_consonant"],
        "preferred_activity_types": ["read_word", "letter_sound"],
        "recommended_action": "remediate",
        "reason_code": "target_initial_sound_error",
    },
    "vowel_error": {
        "primary_focus": "vowel_sound",
        "target_position": "vowel",
        "preferred_error_focus": ["vowel_sound"],
        "preferred_activity_types": ["read_word", "word_family_practice"],
        "recommended_action": "remediate",
        "reason_code": "target_vowel_error",
    },
    "consonant_error": {
        "primary_focus": "consonant_accuracy",
        "target_position": "consonant",
        "preferred_error_focus": ["initial_consonant", "final_consonant", "consonant_blend"],
        "preferred_activity_types": ["read_word"],
        "recommended_action": "remediate",
        "reason_code": "target_consonant_error",
    },
    "skipped_word": {
        "primary_focus": "sentence_tracking",
        "target_position": "sentence",
        "preferred_error_focus": ["sentence_tracking"],
        "preferred_activity_types": ["read_sentence", "sentence_fluency"],
        "recommended_action": "practice",
        "reason_code": "target_skipped_word",
    },
    "partial_sentence": {
        "primary_focus": "fluency_completion",
        "target_position": "sentence",
        "preferred_error_focus": ["fluency_pacing", "sentence_tracking"],
        "preferred_activity_types": ["read_sentence", "sentence_fluency"],
        "recommended_action": "practice",
        "reason_code": "target_partial_sentence",
    },
    "word_order_error": {
        "primary_focus": "sentence_order",
        "target_position": "sentence",
        "preferred_error_focus": ["sentence_tracking"],
        "preferred_activity_types": ["read_sentence"],
        "recommended_action": "practice",
        "reason_code": "target_word_order",
    },
    "blank": {
        "primary_focus": "retry_or_easier",
        "target_position": "",
        "preferred_error_focus": ["initial_consonant", "vowel_sound"],
        "preferred_activity_types": ["read_word", "letter_sound"],
        "recommended_action": "retry_or_easier",
        "reason_code": "blank_response",
    },
    "unclear_asr": {
        "primary_focus": "retry_recording",
        "target_position": "",
        "preferred_error_focus": [],
        "preferred_activity_types": [],
        "recommended_action": "retry_recording",
        "reason_code": "unclear_asr_retry",
    },
    "correct": {
        "primary_focus": "continue_or_increase",
        "target_position": "",
        "preferred_error_focus": [],
        "preferred_activity_types": [],
        "recommended_action": "continue",
        "reason_code": "correct_continue",
    },
    "accepted_variant": {
        "primary_focus": "continue",
        "target_position": "",
        "preferred_error_focus": [],
        "preferred_activity_types": [],
        "recommended_action": "continue",
        "reason_code": "accepted_variant_continue",
    },
    "far_answer": {
        "primary_focus": "easier_review",
        "target_position": "",
        "preferred_error_focus": ["word_reading", "sentence_tracking"],
        "preferred_activity_types": ["read_word", "review"],
        "recommended_action": "easier_review",
        "reason_code": "far_answer_review",
    },
}


SKILL_FALLBACKS = {
    "final_consonant": "final_sound_error",
    "initial_consonant": "initial_sound_error",
    "vowel_sound": "vowel_error",
    "sentence_tracking": "skipped_word",
    "fluency_completion": "partial_sentence",
}


def map_error_to_focus(error_type: str, skill_signal: str | None = None) -> dict[str, object]:
    normalized = (error_type or "").strip() or "incorrect_general"
    if normalized not in ERROR_FOCUS_MAP and skill_signal in SKILL_FALLBACKS:
        normalized = SKILL_FALLBACKS[str(skill_signal)]
    base = ERROR_FOCUS_MAP.get(
        normalized,
        {
            "primary_focus": "general_review",
            "target_position": "",
            "preferred_error_focus": ["word_reading"],
            "preferred_activity_types": ["read_word", "review"],
            "recommended_action": "practice",
            "reason_code": "general_review",
        },
    )
    result = {**base, "avoid_activity_types": []}
    if skill_signal and skill_signal not in result["preferred_error_focus"]:
        result["preferred_error_focus"] = [skill_signal, *list(result["preferred_error_focus"])]
    return result
