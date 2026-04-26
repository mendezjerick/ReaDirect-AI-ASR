from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_history(history: list[dict[str, Any]], recent_window: int = 5) -> dict[str, Any]:
    clean = [item for item in history or [] if isinstance(item, dict)]
    total = len(clean)
    correct = sum(1 for item in clean if item.get("is_correct") is True)
    incorrect = sum(1 for item in clean if item.get("is_correct") is False)
    recent = clean[-recent_window:] if recent_window > 0 else clean
    recent_labeled = [item for item in recent if item.get("is_correct") is not None]
    recent_correct = sum(1 for item in recent_labeled if item.get("is_correct") is True)
    accuracy = round(correct / total, 3) if total else 0.0
    recent_accuracy = round(recent_correct / len(recent_labeled), 3) if recent_labeled else 0.0
    error_counts = count_error_types(clean)
    skill_counts = count_skill_signals(clean)
    phoneme_counts = _count_values(clean, "target_phoneme")
    weak_skills = infer_weak_skill_signals(clean)
    strong_skills = _infer_strong_skill_signals(clean)
    focus = infer_recommended_focus(clean)
    notes: list[str] = []
    if not clean:
        notes.append("no_history_baseline_recommendation")
    return {
        "total_attempts": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy": accuracy,
        "recent_accuracy": recent_accuracy,
        "correct_streak": compute_correct_streak(clean),
        "incorrect_streak": compute_incorrect_streak(clean),
        "error_type_counts": error_counts,
        "skill_signal_counts": skill_counts,
        "target_phoneme_counts": phoneme_counts,
        "weak_skill_signals": weak_skills,
        "strong_skill_signals": strong_skills,
        "recommended_focus": focus,
        "difficulty_adjustment": _difficulty_hint(recent_accuracy, compute_correct_streak(clean), compute_incorrect_streak(clean), total),
        "notes": notes,
    }


def compute_correct_streak(history: list[dict[str, Any]]) -> int:
    streak = 0
    for item in reversed(history or []):
        if item.get("is_correct") is True:
            streak += 1
        else:
            break
    return streak


def compute_incorrect_streak(history: list[dict[str, Any]]) -> int:
    streak = 0
    for item in reversed(history or []):
        if item.get("is_correct") is False:
            streak += 1
        else:
            break
    return streak


def count_error_types(history: list[dict[str, Any]]) -> dict[str, int]:
    return _count_values(history, "error_type", ignored={"", "correct", "accepted_variant"})


def count_skill_signals(history: list[dict[str, Any]]) -> dict[str, int]:
    return _count_values(history, "skill_signal")


def infer_weak_skill_signals(history: list[dict[str, Any]]) -> list[str]:
    errors = [
        str(item.get("skill_signal") or "")
        for item in (history or [])[-10:]
        if item.get("is_correct") is False and str(item.get("skill_signal") or "")
    ]
    counts = Counter(errors)
    return [skill for skill, count in counts.most_common() if count >= 1]


def infer_recommended_focus(history: list[dict[str, Any]]) -> list[str]:
    recent = (history or [])[-5:]
    focus_values = [
        str(item.get("skill_signal") or item.get("error_type") or "")
        for item in recent
        if item.get("is_correct") is False
    ]
    counts = Counter(value for value in focus_values if value)
    return [focus for focus, _ in counts.most_common(3)]


def _infer_strong_skill_signals(history: list[dict[str, Any]]) -> list[str]:
    correct_skills = [
        str(item.get("skill_signal") or "")
        for item in (history or [])[-10:]
        if item.get("is_correct") is True and str(item.get("skill_signal") or "")
    ]
    error_skills = set(infer_weak_skill_signals(history))
    counts = Counter(correct_skills)
    return [skill for skill, count in counts.most_common() if count >= 2 and skill not in error_skills]


def _count_values(history: list[dict[str, Any]], key: str, ignored: set[str] | None = None) -> dict[str, int]:
    ignored = ignored or {""}
    values = [str(item.get(key) or "") for item in history or []]
    counts = Counter(value for value in values if value not in ignored)
    return dict(counts.most_common())


def _difficulty_hint(recent_accuracy: float, correct_streak: int, incorrect_streak: int, total: int) -> str:
    if total == 0:
        return "baseline"
    if correct_streak >= 3 or recent_accuracy >= 0.8:
        return "increase"
    if incorrect_streak >= 2 or recent_accuracy < 0.5:
        return "decrease"
    return "same"
