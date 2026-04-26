from readirect_asr.scoring.skill_signals import infer_skill_signal


def test_skill_signal_mappings() -> None:
    assert infer_skill_signal("final_sound_error", "cat", ["K", "AE", "T"])["skill_signal"] == "final_consonant"
    assert infer_skill_signal("vowel_error", "cat", ["K", "AE", "T"])["skill_signal"] == "vowel_sound"
    assert infer_skill_signal("skipped_word", "red cat")["skill_signal"] == "sentence_tracking"

