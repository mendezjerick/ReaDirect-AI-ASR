from readirect_asr.adaptive.recommendation import AdaptiveRecommendationEngine


def test_engine_uses_request_candidate_items():
    engine = AdaptiveRecommendationEngine()
    result = engine.recommend_next(
        history=[{"is_correct": False, "error_type": "vowel_error", "skill_signal": "vowel_sound", "difficulty_level": "easy"}],
        candidate_items=[{"prompt_id": "V1", "error_focus": "vowel_sound", "difficulty_level": "easy", "is_active": True}],
    )
    assert result["selected_item"]["prompt_id"] == "V1"
    assert "using_request_provided_candidates" in result["warnings"]


def test_engine_falls_back_safely_when_no_candidates():
    engine = AdaptiveRecommendationEngine()
    result = engine.recommend_next(history=[], candidate_items=[])
    assert result["selected_item"] is None
    assert result["recommendation"]["reason_code"] == "no_candidates_available"


class FakeRepository:
    def __init__(self):
        import pandas as pd

        self.df = pd.DataFrame([{"prompt_id": "M2-014", "module_key": "module_2", "error_focus": "final_consonant", "difficulty_level": "easy", "is_active": True}])

    def is_loaded(self):
        return True


def test_engine_loads_candidates_from_repository():
    engine = AdaptiveRecommendationEngine(FakeRepository())
    result = engine.recommend_next(
        history=[{"is_correct": False, "error_type": "final_sound_error", "skill_signal": "final_consonant", "difficulty_level": "easy"}],
        current_context={"module_key": "module_2"},
    )
    assert result["selected_item"]["prompt_id"] == "M2-014"
