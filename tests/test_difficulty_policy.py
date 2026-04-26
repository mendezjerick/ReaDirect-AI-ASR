from readirect_asr.adaptive.difficulty_policy import recommend_difficulty


def test_correct_streak_increases_difficulty():
    result = recommend_difficulty({"total_attempts": 3, "correct_streak": 3, "incorrect_streak": 0, "recent_accuracy": 1.0}, "easy")
    assert result["difficulty_adjustment"] == "increase"
    assert "medium" in result["target_difficulty_levels"]


def test_incorrect_streak_decreases_difficulty():
    result = recommend_difficulty({"total_attempts": 2, "correct_streak": 0, "incorrect_streak": 2, "recent_accuracy": 0.0}, "medium")
    assert result["difficulty_adjustment"] == "decrease"
    assert "easy" in result["target_difficulty_levels"]


def test_low_recent_accuracy_decreases():
    result = recommend_difficulty({"total_attempts": 4, "correct_streak": 0, "incorrect_streak": 1, "recent_accuracy": 0.25}, "medium")
    assert result["difficulty_adjustment"] == "decrease"


def test_no_history_recommends_baseline_easy():
    result = recommend_difficulty({"total_attempts": 0}, None)
    assert result["difficulty_adjustment"] == "same"
    assert "easy" in result["target_difficulty_levels"]
