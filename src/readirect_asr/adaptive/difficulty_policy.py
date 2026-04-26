from __future__ import annotations

from typing import Any


DIFFICULTY_ORDER = ["very_easy", "easy", "medium", "hard", "very_hard"]


def recommend_difficulty(history_summary: dict[str, Any], current_difficulty: str | None = None) -> dict[str, Any]:
    total = int(history_summary.get("total_attempts", 0) or 0)
    correct_streak = int(history_summary.get("correct_streak", 0) or 0)
    incorrect_streak = int(history_summary.get("incorrect_streak", 0) or 0)
    recent_accuracy = float(history_summary.get("recent_accuracy", 0.0) or 0.0)
    if total == 0:
        return {
            "difficulty_adjustment": "same",
            "target_difficulty_levels": ["very_easy", "easy"],
            "reason": "no_history_baseline",
        }
    if _latest_error(history_summary) == "unclear_asr":
        return {
            "difficulty_adjustment": "same",
            "target_difficulty_levels": _nearby_levels(current_difficulty),
            "reason": "unclear_asr_retry_before_level_change",
        }
    if correct_streak >= 3 or recent_accuracy >= 0.8:
        target = _shift_level(current_difficulty, 1)
        return {
            "difficulty_adjustment": "increase",
            "target_difficulty_levels": [target, *_nearby_levels(target)],
            "reason": f"correct_streak_{correct_streak}" if correct_streak >= 3 else "high_recent_accuracy",
        }
    if incorrect_streak >= 2 or recent_accuracy < 0.5:
        target = _shift_level(current_difficulty, -1)
        return {
            "difficulty_adjustment": "decrease",
            "target_difficulty_levels": [target, *_nearby_levels(target)],
            "reason": f"incorrect_streak_{incorrect_streak}" if incorrect_streak >= 2 else "low_recent_accuracy",
        }
    return {
        "difficulty_adjustment": "same",
        "target_difficulty_levels": _nearby_levels(current_difficulty),
        "reason": "steady_recent_accuracy",
    }


def _shift_level(current: str | None, delta: int) -> str:
    level = current if current in DIFFICULTY_ORDER else "easy"
    index = DIFFICULTY_ORDER.index(level)
    return DIFFICULTY_ORDER[min(max(index + delta, 0), len(DIFFICULTY_ORDER) - 1)]


def _nearby_levels(current: str | None) -> list[str]:
    level = current if current in DIFFICULTY_ORDER else "easy"
    index = DIFFICULTY_ORDER.index(level)
    ordered = [level]
    if index > 0:
        ordered.append(DIFFICULTY_ORDER[index - 1])
    if index + 1 < len(DIFFICULTY_ORDER):
        ordered.append(DIFFICULTY_ORDER[index + 1])
    return ordered


def _latest_error(history_summary: dict[str, Any]) -> str:
    notes = history_summary.get("notes", [])
    if "latest_unclear_asr" in notes:
        return "unclear_asr"
    return ""
