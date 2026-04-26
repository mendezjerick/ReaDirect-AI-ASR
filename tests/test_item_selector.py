from readirect_asr.adaptive.item_selector import AdaptiveItemSelector


def _history():
    return [
        {
            "prompt_id": "M2-001",
            "is_correct": False,
            "error_type": "final_sound_error",
            "skill_signal": "final_consonant",
            "target_phoneme": "T",
            "difficulty_level": "easy",
        }
    ]


def test_selects_item_matching_final_consonant():
    selector = AdaptiveItemSelector()
    result = selector.select_next_item(
        [
            {"prompt_id": "M2-014", "expected_text": "hat", "error_focus": "final_consonant", "target_phoneme": "T", "difficulty_level": "easy", "is_active": True},
            {"prompt_id": "M2-020", "expected_text": "map", "error_focus": "initial_consonant", "difficulty_level": "easy", "is_active": True},
        ],
        _history(),
        {"module_key": "module_2"},
    )
    assert result["selected_item"]["prompt_id"] == "M2-014"
    assert result["recommendation"]["primary_focus"] == "final_consonant"


def test_avoids_recent_and_inactive_items():
    selector = AdaptiveItemSelector()
    result = selector.select_next_item(
        [
            {"prompt_id": "M2-001", "error_focus": "final_consonant", "difficulty_level": "easy", "is_active": True},
            {"prompt_id": "M2-014", "error_focus": "final_consonant", "difficulty_level": "easy", "is_active": False},
            {"prompt_id": "M2-015", "error_focus": "final_consonant", "difficulty_level": "easy", "is_active": True},
        ],
        _history(),
    )
    assert result["selected_item"]["prompt_id"] == "M2-015"


def test_penalizes_manual_review_and_respects_module():
    selector = AdaptiveItemSelector()
    result = selector.select_next_item(
        [
            {"prompt_id": "A", "module_key": "module_1", "error_focus": "final_consonant", "difficulty_level": "easy", "needs_manual_review": True},
            {"prompt_id": "B", "module_key": "module_2", "error_focus": "final_consonant", "difficulty_level": "easy", "needs_manual_review": False},
        ],
        _history(),
        {"module_key": "module_2"},
    )
    assert result["selected_item"]["prompt_id"] == "B"
    assert result["ranked_candidates"]
