import pandas as pd

from readirect_asr.evaluation.error_analysis import (
    categorize_asr_error,
    summarize_common_substitutions,
    summarize_short_word_accuracy,
)


def test_error_categories() -> None:
    assert categorize_asr_error("cat", "cat") == "exact"
    assert categorize_asr_error("cat", "cat sat") == "insertion"
    assert categorize_asr_error("cat sat", "cat") == "deletion"
    assert categorize_asr_error("cat", "bat") == "substitution"
    assert categorize_asr_error("cat", "") == "blank_hypothesis"


def test_short_word_summary_and_substitutions() -> None:
    df = pd.DataFrame(
        {
            "manual_transcript": ["cat", "dog", "long sentence here"],
            "normalized_transcript": ["bat", "dog", "long sentence"],
        }
    )

    summary = summarize_short_word_accuracy(df)
    substitutions = summarize_common_substitutions(df, "manual_transcript", "normalized_transcript")

    assert summary["short_word_rows"] == 2
    assert summary["exact_match_rate"] == 0.5
    assert substitutions[0]["reference"] == "cat"

