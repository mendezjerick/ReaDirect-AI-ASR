from readirect_asr.finetuning.decision_rules import decide_finetuning_need


READY = {"ready": True, "issues": [], "recommendations": []}
NOT_READY = {"ready": False, "issues": ["too_few_rows"], "recommendations": ["Collect more rows."]}


def test_baseline_missing():
    result = decide_finetuning_need(None, READY)
    assert result["decision"] == "baseline_missing"


def test_more_data_needed():
    result = decide_finetuning_need({"wer": 0.5, "cer": 0.2}, NOT_READY)
    assert result["decision"] == "more_data_needed"


def test_recommends_for_high_wer_cer():
    result = decide_finetuning_need({"wer": 0.3, "cer": 0.15, "evaluated_rows": 100, "blank_hypothesis_count": 0}, READY)
    assert result["decision"] == "fine_tuning_recommended"


def test_not_needed_for_good_metrics():
    result = decide_finetuning_need(
        {"wer": 0.05, "cer": 0.02, "exact_match_rate": 0.9, "evaluated_rows": 100, "blank_hypothesis_count": 0},
        READY,
        {"exact_match_rate": 0.9},
    )
    assert result["decision"] == "not_needed_yet"


def test_recommends_for_poor_short_word_accuracy():
    result = decide_finetuning_need(
        {"wer": 0.12, "cer": 0.04, "exact_match_rate": 0.8, "evaluated_rows": 100, "blank_hypothesis_count": 0},
        READY,
        {"exact_match_rate": 0.5},
    )
    assert result["decision"] == "fine_tuning_recommended"
