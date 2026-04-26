from __future__ import annotations

import math
from typing import Any

from readirect_asr.adaptive.explanation import generate_recommendation_explanation
from readirect_asr.adaptive.item_selector import AdaptiveItemSelector


class AdaptiveRecommendationEngine:
    def __init__(self, content_repository: Any | None = None, config: dict[str, Any] | None = None) -> None:
        self.content_repository = content_repository
        self.config = config or {}
        self.selector = AdaptiveItemSelector(self.config)

    def recommend_next(
        self,
        history: list[dict[str, Any]],
        current_context: dict[str, Any] | None = None,
        candidate_items: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        debug: bool = False,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        context = current_context or {}
        if candidate_items:
            candidates = [_json_safe(item) for item in candidate_items if isinstance(item, dict)]
            warnings.append("using_request_provided_candidates")
        else:
            candidates = self.load_candidates_from_content_repository(context.get("module_key"), context.get("activity_type"))
            if not candidates:
                warnings.append("no_content_repository_candidates_found")
        if not candidates:
            result = {
                "selected_item": None,
                "ranked_candidates": [],
                "learner_summary": self.selector.select_next_item([], history or [], context, top_k).get("learner_summary", {}),
                "recommendation": {
                    "recommended_action": "fallback",
                    "primary_focus": "baseline",
                    "difficulty_adjustment": "same",
                    "reason_code": "no_candidates_available",
                    "confidence": "none",
                },
                "warnings": warnings,
                "debug_info": {"candidate_count": 0} if debug else None,
            }
            result["explanation"] = generate_recommendation_explanation(result["recommendation"], result["learner_summary"])
            return _json_safe(result)
        selected = self.selector.select_next_item(candidates, history or [], context, top_k)
        selected["warnings"] = warnings
        selected["explanation"] = generate_recommendation_explanation(selected.get("recommendation", {}), selected.get("learner_summary", {}))
        selected["debug_info"] = {"candidate_count": len(candidates)} if debug else None
        return _json_safe(selected)

    def load_candidates_from_content_repository(self, module_key: str | None = None, activity_type: str | None = None) -> list[dict[str, Any]]:
        repo = self.content_repository
        if repo is None or not getattr(repo, "is_loaded", lambda: False)():
            return []
        df = getattr(repo, "df", None)
        if df is None or getattr(df, "empty", True):
            return []
        filtered = df
        if module_key and "module_key" in filtered.columns:
            filtered = filtered[filtered["module_key"].fillna("").astype(str) == str(module_key)]
        if activity_type and "activity_type" in filtered.columns:
            same_activity = filtered[filtered["activity_type"].fillna("").astype(str) == str(activity_type)]
            if not same_activity.empty:
                filtered = same_activity
        return [_json_safe(row) for row in filtered.fillna("").to_dict(orient="records")]

    def build_recommendation_response(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": bool(result.get("selected_item")),
            "selected_item": result.get("selected_item"),
            "ranked_candidates": result.get("ranked_candidates", []),
            "learner_summary": result.get("learner_summary", {}),
            "recommendation": result.get("recommendation", {}),
            "explanation": result.get("explanation", {}),
            "warnings": result.get("warnings", []),
            "debug_info": result.get("debug_info"),
            "error": None if result.get("selected_item") else "no_recommendation_available",
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return str(value)
