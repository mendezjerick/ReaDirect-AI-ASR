from __future__ import annotations

from typing import Any


ERROR_BY_FOCUS = {
    "final_consonant": "final_sound_error",
    "initial_consonant": "initial_sound_error",
    "vowel_sound": "vowel_error",
    "sentence_tracking": "skipped_word",
    "fluency_pacing": "partial_sentence",
    "comprehension_detail": "comprehension_detail_error",
    "comprehension_inference": "comprehension_inference_error",
    "word_family": "word_family_error",
}


def generate_adaptive_metadata(row: dict[str, Any], enrichment: dict[str, Any]) -> dict[str, object]:
    error_focus = str(enrichment.get("error_focus") or "unknown")
    difficulty_level = str(enrichment.get("difficulty_level") or "medium")
    needs_manual_review = _truthy(enrichment.get("needs_manual_review"))
    practice_role = _practice_role(row, enrichment)
    recommended = ERROR_BY_FOCUS.get(error_focus, "incorrect_general")
    focused = error_focus in {"final_consonant", "initial_consonant", "vowel_sound", "sentence_tracking", "fluency_pacing"}
    mastery_candidate = practice_role == "mastery_check" and not needs_manual_review and difficulty_level not in {"hard", "very_hard"}
    review_candidate = difficulty_level in {"very_easy", "easy", "medium"} and not needs_manual_review
    return {
        "adaptive_bucket": f"{error_focus}_{difficulty_level}",
        "recommended_for_error_type": recommended,
        "remediation_priority": "high" if focused else "medium" if error_focus != "unknown" else "low",
        "practice_role": practice_role,
        "mastery_candidate": mastery_candidate,
        "review_candidate": review_candidate,
        "min_required_attempts": 3 if focused else 1,
        "cooldown_group": f"{error_focus}:{enrichment.get('target_phoneme') or enrichment.get('word_family') or 'general'}",
    }


def _practice_role(row: dict[str, Any], enrichment: dict[str, Any]) -> str:
    activity = str(row.get("activity_type", "")).lower()
    source_group = str(row.get("source_group", "")).lower()
    mastery = str(row.get("is_mastery_item", "")).lower() in {"1", "true", "yes"} or "mastery" in activity
    if mastery:
        return "mastery_check"
    if source_group == "assessment":
        return "assessment"
    if enrichment.get("error_focus") not in {"unknown", "", None}:
        return "practice"
    return "unknown"


def _truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}

