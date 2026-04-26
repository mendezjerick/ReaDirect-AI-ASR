from __future__ import annotations


MAPPINGS = {
    "correct": ("praise_continue", "praise_continue", "Good reading. Keep going.", "continue"),
    "accepted_variant": ("praise_continue", "praise_continue", "Good reading. Keep going.", "continue"),
    "blank": ("try_answer_first", "try_answer_first", "Try saying the answer first.", "retry"),
    "initial_sound_error": ("beginning_sound", "listen_to_first_sound", "Listen to the first sound and try again.", "retry"),
    "final_sound_error": ("ending_sound", "listen_to_final_sound", "Listen to the ending sound and try again.", "retry"),
    "vowel_error": ("middle_sound", "listen_to_middle_sound", "Listen to the middle sound and try again.", "retry"),
    "consonant_error": ("consonant_sound", "listen_to_consonant_sound", "Listen to the consonant sounds and try again.", "retry"),
    "skipped_word": ("read_each_word", "read_each_word", "Read each word carefully.", "retry"),
    "partial_sentence": ("read_whole_sentence", "read_whole_sentence", "Try reading the whole sentence.", "retry"),
    "word_order_error": ("word_order", "read_in_order", "Read the words in order.", "retry"),
    "far_answer": ("listen_again", "listen_again", "Listen again and try once more.", "retry"),
    "unclear_asr": ("try_recording_again", "try_recording_again", "Try recording again clearly.", "retry"),
    "incorrect_general": ("try_again_slowly", "try_again_slowly", "Try again slowly.", "retry"),
}


def generate_feedback_hint(error_type: str, skill_signal: str | None = None) -> dict[str, object]:
    feedback_hint, coach_hint_key, summary, action = MAPPINGS.get(error_type, MAPPINGS["incorrect_general"])
    return {
        "feedback_hint": feedback_hint,
        "coach_hint_key": coach_hint_key,
        "learner_safe_summary": summary,
        "recommended_action": action,
        "skill_signal": skill_signal,
    }

