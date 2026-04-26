from readirect_asr.adaptive.remediation_policy import map_error_to_focus


def test_error_focus_mappings():
    assert map_error_to_focus("final_sound_error")["primary_focus"] == "final_consonant"
    assert map_error_to_focus("initial_sound_error")["primary_focus"] == "initial_consonant"
    assert map_error_to_focus("vowel_error")["primary_focus"] == "vowel_sound"
    assert map_error_to_focus("skipped_word")["primary_focus"] == "sentence_tracking"
    assert map_error_to_focus("unclear_asr")["recommended_action"] == "retry_recording"


def test_skill_signal_fallback():
    result = map_error_to_focus("incorrect_general", "final_consonant")
    assert result["primary_focus"] == "final_consonant"
