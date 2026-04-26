from readirect_asr.adaptive.learner_state import (
    compute_correct_streak,
    compute_incorrect_streak,
    infer_weak_skill_signals,
    summarize_history,
)


def test_summarize_history_counts_accuracy_and_recent_accuracy():
    history = [
        {"is_correct": True, "skill_signal": "final_consonant"},
        {"is_correct": False, "error_type": "final_sound_error", "skill_signal": "final_consonant"},
        {"is_correct": False, "error_type": "vowel_error", "skill_signal": "vowel_sound"},
    ]
    summary = summarize_history(history, recent_window=2)
    assert summary["total_attempts"] == 3
    assert summary["correct_count"] == 1
    assert summary["incorrect_count"] == 2
    assert summary["accuracy"] == 0.333
    assert summary["recent_accuracy"] == 0.0
    assert summary["error_type_counts"]["final_sound_error"] == 1


def test_streaks_and_weak_skills():
    history = [
        {"is_correct": True, "skill_signal": "vowel_sound"},
        {"is_correct": False, "skill_signal": "final_consonant"},
        {"is_correct": False, "skill_signal": "final_consonant"},
    ]
    assert compute_correct_streak(history) == 0
    assert compute_incorrect_streak(history) == 2
    assert infer_weak_skill_signals(history)[0] == "final_consonant"


def test_correct_streak():
    history = [{"is_correct": False}, {"is_correct": True}, {"is_correct": True}]
    assert compute_correct_streak(history) == 2
    assert compute_incorrect_streak(history) == 0
