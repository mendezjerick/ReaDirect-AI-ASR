from __future__ import annotations

from typing import Any


def infer_skill_signal(
    error_type: str,
    expected_text: str,
    expected_phonemes: list[str] | None = None,
    content_metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    phonemes = expected_phonemes or []
    metadata = content_metadata or {}
    skill_signal = str(metadata.get("skill_signal", "") or "")
    target_position = ""
    target_phoneme = ""
    difficulty_adjustment = "same"

    if error_type == "initial_sound_error":
        skill_signal = skill_signal or "initial_consonant"
        target_position = "initial"
        target_phoneme = phonemes[0] if phonemes else ""
    elif error_type == "final_sound_error":
        skill_signal = skill_signal or "final_consonant"
        target_position = "final"
        target_phoneme = phonemes[-1] if phonemes else ""
    elif error_type == "vowel_error":
        skill_signal = skill_signal or "vowel_sound"
        target_position = "medial"
        target_phoneme = next((phone for phone in phonemes if phone in {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}), "")
    elif error_type == "skipped_word":
        skill_signal = skill_signal or "sentence_tracking"
    elif error_type == "partial_sentence":
        skill_signal = skill_signal or "fluency_completion"
    elif error_type == "word_order_error":
        skill_signal = skill_signal or "sentence_order"
    elif error_type in {"correct", "accepted_variant"}:
        skill_signal = skill_signal or "ready_to_advance"
        difficulty_adjustment = "increase_if_repeated"
    elif error_type == "far_answer":
        skill_signal = skill_signal or "listening_comprehension"
        difficulty_adjustment = "easier"
    else:
        skill_signal = skill_signal or "general_accuracy"

    focus_suffix = f"_{target_phoneme.lower()}_words" if target_phoneme else ""
    return {
        "skill_signal": skill_signal,
        "target_phoneme": target_phoneme,
        "target_position": target_position,
        "recommended_practice_focus": f"{skill_signal}{focus_suffix}",
        "difficulty_adjustment": difficulty_adjustment,
    }

