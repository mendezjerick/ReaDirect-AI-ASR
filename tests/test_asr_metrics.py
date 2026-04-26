import pandas as pd

from readirect_asr.evaluation.asr_metrics import (
    compute_cer,
    compute_wer,
    evaluate_asr_dataframe,
    exact_match,
    token_accuracy,
)


def test_wer_and_cer_exact_match_are_zero() -> None:
    assert compute_wer("cat sat", "cat sat") == 0
    assert compute_cer("cat", "cat") == 0


def test_exact_match_and_token_accuracy() -> None:
    assert exact_match("Cat!", "cat")
    assert token_accuracy("cat sat", "cat ran") == 0.5


def test_blank_reference_and_hypothesis_handled() -> None:
    df = pd.DataFrame({"ref": ["cat", ""], "hyp": ["", "dog"]})
    metrics = evaluate_asr_dataframe(df, "ref", "hyp")
    assert metrics["evaluated_rows"] == 1
    assert metrics["blank_reference_count"] == 1
    assert metrics["blank_hypothesis_count"] == 1

