from __future__ import annotations

import math
from typing import Any

from readirect_asr.adaptive.difficulty_policy import recommend_difficulty
from readirect_asr.adaptive.learner_state import summarize_history
from readirect_asr.adaptive.remediation_policy import map_error_to_focus


DEFAULT_WEIGHTS = {
    "same_error_focus": 4,
    "same_skill_signal": 3,
    "same_target_phoneme": 2,
    "appropriate_difficulty": 2,
    "not_recently_used": 2,
    "active_item": 3,
    "needs_manual_review_penalty": -5,
    "same_module": 1,
}


class AdaptiveItemSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.weights = {**DEFAULT_WEIGHTS, **(self.config.get("error_focus_weights", {}) if isinstance(self.config, dict) else {})}

    def select_next_item(
        self,
        candidates: list[dict[str, Any]],
        learner_history: list[dict[str, Any]],
        current_context: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        context = current_context or {}
        history_summary = summarize_history(learner_history, int(self.config.get("recent_window", 5) or 5))
        latest = _latest_history_item(learner_history)
        remediation = map_error_to_focus(str(latest.get("error_type") or ""), latest.get("skill_signal") or _first(history_summary.get("recommended_focus", [])))
        current_difficulty = latest.get("difficulty_level") or context.get("difficulty_level")
        difficulty = recommend_difficulty(history_summary, current_difficulty)
        recent_prompt_ids = _recent_prompt_ids(learner_history, int(self.config.get("avoid_recent_items_count", 3) or 3))
        ranked = [
            self._score_candidate(candidate, remediation, difficulty, context, recent_prompt_ids)
            for candidate in candidates or []
            if isinstance(candidate, dict)
        ]
        ranked.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
        selected = ranked[0] if ranked else None
        recommendation = {
            "recommended_action": remediation.get("recommended_action", "practice"),
            "primary_focus": remediation.get("primary_focus", "general_review"),
            "difficulty_adjustment": difficulty.get("difficulty_adjustment", "same"),
            "reason_code": remediation.get("reason_code", "general_review"),
            "confidence": "heuristic",
            "target_difficulty_levels": difficulty.get("target_difficulty_levels", []),
            "difficulty_reason": difficulty.get("reason", ""),
        }
        return {
            "selected_item": _public_candidate(selected) if selected else None,
            "ranked_candidates": [_public_candidate(item) for item in ranked[: max(top_k, 1)]],
            "learner_summary": history_summary,
            "recommendation": recommendation,
        }

    def _score_candidate(
        self,
        candidate: dict[str, Any],
        remediation: dict[str, Any],
        difficulty: dict[str, Any],
        context: dict[str, Any],
        recent_prompt_ids: set[str],
    ) -> dict[str, Any]:
        item = _normalize_candidate(candidate)
        score = 0.0
        reasons: list[str] = []
        preferred_focus = set(str(value) for value in remediation.get("preferred_error_focus", []) if value)
        if item.get("error_focus") in preferred_focus:
            score += self.weights["same_error_focus"]
            reasons.append("error_focus_match")
        if item.get("skill_signal") in preferred_focus or item.get("skill_tag") in preferred_focus:
            score += self.weights["same_skill_signal"]
            reasons.append("skill_signal_match")
        if item.get("target_phoneme") and item.get("target_phoneme") == _target_from_context(context):
            score += self.weights["same_target_phoneme"]
            reasons.append("target_phoneme_match")
        if item.get("target_position") and item.get("target_position") == remediation.get("target_position"):
            score += 1
            reasons.append("target_position_match")
        if item.get("difficulty_level") in set(difficulty.get("target_difficulty_levels", [])):
            score += self.weights["appropriate_difficulty"]
            reasons.append("difficulty_match")
        if _truthy(item.get("is_active"), default=True):
            score += self.weights["active_item"]
            reasons.append("active_item")
        else:
            score -= 10
            reasons.append("inactive_penalty")
        if _truthy(item.get("needs_manual_review"), default=False):
            score += self.weights["needs_manual_review_penalty"]
            reasons.append("manual_review_penalty")
        if str(item.get("prompt_id") or "") not in recent_prompt_ids:
            score += self.weights["not_recently_used"]
            reasons.append("not_recently_used")
        else:
            score -= 4
            reasons.append("recent_item_penalty")
        if context.get("module_key") and item.get("module_key") == context.get("module_key"):
            score += self.weights["same_module"]
            reasons.append("same_module")
        if context.get("activity_type") and item.get("activity_type") == context.get("activity_type"):
            score += 1
            reasons.append("same_activity_type")
        if not self.config.get("allow_mastery_items_by_default", False) and _truthy(item.get("mastery_candidate"), default=False) and not context.get("allow_mastery"):
            score -= 3
            reasons.append("mastery_item_penalty")
        item["_score"] = round(score, 3) if not math.isnan(score) else 0.0
        item["_score_reasons"] = reasons
        return item


def _latest_history_item(history: list[dict[str, Any]]) -> dict[str, Any]:
    return next((item for item in reversed(history or []) if isinstance(item, dict)), {})


def _recent_prompt_ids(history: list[dict[str, Any]], count: int) -> set[str]:
    return {str(item.get("prompt_id")) for item in (history or [])[-count:] if item.get("prompt_id")}


def _first(values: Any) -> str | None:
    if isinstance(values, list) and values:
        return str(values[0])
    return None


def _target_from_context(context: dict[str, Any]) -> str:
    return str(context.get("target_phoneme") or context.get("last_target_phoneme") or "")


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "active"}


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    item.setdefault("is_active", True)
    item.setdefault("needs_manual_review", False)
    item.setdefault("difficulty_level", "easy")
    if not item.get("error_focus") and item.get("skill_signal"):
        item["error_focus"] = item.get("skill_signal")
    return item


def _public_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {key: value for key, value in candidate.items() if not key.startswith("__")}
