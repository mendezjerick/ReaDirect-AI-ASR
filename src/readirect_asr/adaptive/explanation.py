from __future__ import annotations

from typing import Any


LEARNER_SUMMARIES = {
    "final_consonant": "Let's practice the ending sound again.",
    "initial_consonant": "Let's practice the first sound again.",
    "vowel_sound": "Let's practice the middle sound again.",
    "sentence_tracking": "Let's practice reading each word.",
    "fluency_completion": "Let's practice reading the whole sentence.",
    "sentence_order": "Let's practice the word order.",
    "retry_recording": "Let's try recording that again.",
    "continue_or_increase": "Nice work. Let's keep going.",
    "easier_review": "Let's review with an easier item.",
    "general_review": "Let's practice another item.",
}


def generate_recommendation_explanation(recommendation: dict[str, Any], learner_summary: dict[str, Any]) -> dict[str, Any]:
    focus = str(recommendation.get("primary_focus") or "general_review")
    adjustment = str(recommendation.get("difficulty_adjustment") or "same")
    reason_code = str(recommendation.get("reason_code") or "general_review")
    learner_safe = LEARNER_SUMMARIES.get(focus, LEARNER_SUMMARIES["general_review"])
    repeated = "repeated " if _has_repeated_focus(focus, learner_summary) else ""
    teacher = f"The learner has {repeated}{focus.replace('_', ' ')} needs, so the next item targets that focus at {adjustment} difficulty."
    developer = "Selected by heuristic candidate scoring using focus match, difficulty fit, activity/module context, recency, active status, and review safety."
    return {
        "teacher_explanation": teacher,
        "developer_explanation": developer,
        "learner_safe_summary": learner_safe,
        "reason_codes": [reason_code, f"{adjustment}_difficulty"],
    }


def _has_repeated_focus(focus: str, learner_summary: dict[str, Any]) -> bool:
    counts = learner_summary.get("skill_signal_counts", {}) or {}
    return int(counts.get(focus, 0) or 0) >= 2
