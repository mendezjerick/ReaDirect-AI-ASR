import pandas as pd

from readirect_asr.evaluation.readirect_metrics import (
    evaluate_cvc_words,
    evaluate_readirect_keywords,
    evaluate_short_words,
    is_short_word,
)


def test_short_word_detection():
    assert is_short_word("cat")
    assert is_short_word("box")
    assert not is_short_word("the cat")
    assert not is_short_word("yellow")


def test_short_word_metrics_confusions_and_blank_rate():
    df = pd.DataFrame({"ref": ["cat", "dog", "the cat"], "hyp": ["cat", "", "the cat"]})
    result = evaluate_short_words(df, "ref", "hyp")
    assert result["total_short_word_rows"] == 2
    assert result["exact_match_rate"] == 0.5
    assert result["blank_rate"] == 0.5
    assert result["common_confusions"][0]["reference"] == "dog"


def test_cvc_fallback_and_keywords():
    df = pd.DataFrame({"ref": ["cat", "sun", "tree"], "hyp": ["cap", "sun", "tree"]})
    cvc = evaluate_cvc_words(df, "ref", "hyp")
    keywords = evaluate_readirect_keywords(df, "ref", "hyp", ["cat", "sun"])
    assert cvc["total_cvc_rows"] == 2
    assert keywords["total_short_word_rows"] == 2
