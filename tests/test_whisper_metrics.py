from readirect_asr.finetuning.whisper_metrics import (
    compute_cer_metric,
    compute_exact_match_rate,
    compute_wer_metric,
    normalize_prediction_text,
)


def test_metric_exact_matches_are_zero_error():
    predictions = ["cat", "the dog"]
    references = ["cat", "the dog"]
    assert compute_wer_metric(predictions, references) == 0.0
    assert compute_cer_metric(predictions, references) == 0.0
    assert compute_exact_match_rate(predictions, references) == 1.0


def test_normalization_works():
    assert normalize_prediction_text(" Cat! ") == "cat"
