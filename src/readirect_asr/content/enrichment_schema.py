from __future__ import annotations


CORE_IDENTITY_COLUMNS = [
    "prompt_id",
    "source_file",
    "source_group",
    "module_key",
    "task_type",
    "activity_type",
    "prompt_text",
    "expected_text",
    "accepted_answers",
]

PHONEME_METADATA_COLUMNS = [
    "expected_phonemes",
    "initial_phoneme",
    "vowel_phonemes",
    "final_phoneme",
    "phoneme_pattern",
    "phoneme_count",
    "syllable_estimate",
    "has_cmudict_match",
    "cmudict_missing_words",
]

SKILL_METADATA_COLUMNS = [
    "skill_tag",
    "skill_group",
    "error_focus",
    "target_position",
    "target_phoneme",
    "target_grapheme",
    "word_family",
    "rime_unit",
    "onset_unit",
]

ADAPTIVE_METADATA_COLUMNS = [
    "adaptive_bucket",
    "recommended_for_error_type",
    "remediation_priority",
    "difficulty_level",
    "difficulty_score",
    "practice_role",
    "mastery_candidate",
    "review_candidate",
    "min_required_attempts",
    "cooldown_group",
]

QUALITY_METADATA_COLUMNS = [
    "enrichment_status",
    "enrichment_warnings",
    "needs_manual_review",
]

ENRICHMENT_COLUMNS = (
    CORE_IDENTITY_COLUMNS
    + PHONEME_METADATA_COLUMNS
    + SKILL_METADATA_COLUMNS
    + ADAPTIVE_METADATA_COLUMNS
    + QUALITY_METADATA_COLUMNS
)

VALID_SKILL_GROUPS = {
    "letter_sound",
    "phonemic_awareness",
    "word_reading",
    "sentence_reading",
    "fluency",
    "comprehension",
    "unknown",
}

VALID_ERROR_FOCUS = {
    "initial_consonant",
    "final_consonant",
    "vowel_sound",
    "consonant_blend",
    "digraph",
    "word_family",
    "sentence_tracking",
    "fluency_pacing",
    "comprehension_detail",
    "comprehension_inference",
    "sequencing",
    "vocabulary",
    "unknown",
}

VALID_PRACTICE_ROLES = {"practice", "mastery_check", "review", "remediation", "assessment", "unknown"}
VALID_DIFFICULTY_LEVELS = {"very_easy", "easy", "medium", "hard", "very_hard"}

