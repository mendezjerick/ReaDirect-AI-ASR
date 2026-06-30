from __future__ import annotations


ASSESSMENT_SCHEMAS: dict[str, list[str]] = {
    "task1_letter_pronunciation.csv": ["id", "prompt_text", "expected_answer", "accepted_answers", "is_active"],
    "task2a_rhyming_words.csv": ["id", "prompt_text", "accepted_answers", "is_active"],
    "task2b_word_in_sentence.csv": ["id", "sentence_text", "target_word", "expected_answer", "accepted_answers", "is_active"],
    "reading_passages.csv": ["id", "passage_text", "difficulty", "is_active"],
    "comprehension_questions.csv": ["id", "passage_id", "question_text", "correct_answer", "accepted_answers", "is_active"],
}

MODULE_SCHEMAS: dict[str, list[str]] = {
    "module1_letter_sound_activities.csv": ["prompt_id", "module_key", "activity_type", "prompt_text", "expected_text", "accepted_answers", "is_active"],
    "module2_word_reading_activities.csv": ["prompt_id", "module_key", "activity_type", "prompt_text", "expected_text", "accepted_answers", "is_active"],
    "module3_sentence_fluency_activities.csv": ["prompt_id", "module_key", "activity_type", "prompt_text", "expected_text", "accepted_answers", "is_active"],
    "module_activity_selection_rules.csv": ["prompt_id", "module_key", "activity_type", "is_active"],
    "module_feedback_templates.csv": ["id", "module_key", "activity_type", "error_type", "feedback_text", "is_active"],
    "mastery_thresholds.csv": ["id"],
}

RULE_SCHEMAS: dict[str, list[str]] = {
    "reading_classification_rules.csv": ["id", "min_score", "max_score", "classification", "is_active"],
    "module_placement_rules.csv": ["id", "crla_level", "reading_classification", "assigned_module", "is_active"],
    "crla_classification_rules.csv": ["id"],
    "module_mastery_rules.csv": ["id"],
}

KNOWN_SCHEMAS: dict[str, list[str]] = {
    **ASSESSMENT_SCHEMAS,
    **MODULE_SCHEMAS,
    **RULE_SCHEMAS,
}

REQUIRED_CONTENT_FILES = [
    "assessment/task1_letter_pronunciation.csv",
    "assessment/task2a_rhyming_words.csv",
    "assessment/task2b_word_in_sentence.csv",
    "modules/module1_letter_sound_activities.csv",
    "modules/module2_word_reading_activities.csv",
    "modules/module3_sentence_fluency_activities.csv",
]

OPTIONAL_CONTENT_FILES = [
    "assessment/reading_passages.csv",
    "assessment/comprehension_questions.csv",
    "modules/module_activity_selection_rules.csv",
    "modules/module_feedback_templates.csv",
    "modules/mastery_thresholds.csv",
    "rules/reading_classification_rules.csv",
    "rules/module_placement_rules.csv",
    "rules/crla_classification_rules.csv",
    "rules/module_mastery_rules.csv",
]
