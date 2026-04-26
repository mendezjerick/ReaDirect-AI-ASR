from pathlib import Path

import pandas as pd

from readirect_asr.content.enricher import ContentEnricher
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _loader(tmp_path: Path) -> CMUDictLoader:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("CAT K AE1 T\nRED R EH1 D\nSAT S AE1 T\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\nR liquid\nEH vowel\nD stop\nS fricative\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("K\nAE\nAE1\nT\nR\nEH\nEH1\nD\nS\n", encoding="utf-8")
    return CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()


def test_enriches_single_word_cat(tmp_path: Path) -> None:
    result = ContentEnricher(_loader(tmp_path)).enrich_row({"expected_text": "cat", "module_key": "module_2", "activity_type": "read_word"})
    assert result["expected_phonemes"] == "K AE T"
    assert result["phoneme_pattern"] == "CVC"
    assert result["skill_group"] == "word_reading"
    assert result["has_cmudict_match"] is True


def test_handles_missing_cmudict_word(tmp_path: Path) -> None:
    result = ContentEnricher(_loader(tmp_path)).enrich_row({"expected_text": "zzzz", "module_key": "module_2"})
    assert result["needs_manual_review"] is True
    assert result["cmudict_missing_words"] == "zzzz"


def test_enriches_sentence_and_letter(tmp_path: Path) -> None:
    enricher = ContentEnricher(_loader(tmp_path))
    sentence = enricher.enrich_row({"expected_text": "red cat sat", "module_key": "module_3", "activity_type": "read_sentence"})
    letter = enricher.enrich_row({"expected_text": "A", "module_key": "module_1", "activity_type": "letter_sound"})
    assert sentence["skill_group"] == "sentence_reading"
    assert sentence["word_count"] == 3
    assert letter["skill_group"] == "letter_sound"
    assert letter["target_grapheme"] == "A"


def test_enrich_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame([{"prompt_id": "M2-001", "expected_text": "cat", "source_group": "modules", "source_file": "fake.csv"}])
    enriched = ContentEnricher(_loader(tmp_path)).enrich_dataframe(df)
    assert enriched.loc[0, "prompt_id"] == "M2-001"
    assert enriched.loc[0, "expected_phonemes"] == "K AE T"

