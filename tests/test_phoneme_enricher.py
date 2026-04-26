from pathlib import Path

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.phonemes.phoneme_enricher import (
    classify_phoneme_pattern,
    extract_final_phoneme,
    extract_initial_phoneme,
    extract_vowel_phonemes,
    text_to_phonemes,
    word_to_phonemes,
)
from readirect_asr.phonemes.phoneme_schema import PhonemeSchema


def _fake_loader(tmp_path: Path) -> CMUDictLoader:
    dict_path = tmp_path / "cmudict.dict"
    phones_path = tmp_path / "cmudict.phones"
    symbols_path = tmp_path / "cmudict.symbols"
    dict_path.write_text("CAT K AE1 T\n", encoding="utf-8")
    phones_path.write_text("K stop\nAE vowel\nT stop\n", encoding="utf-8")
    symbols_path.write_text("K\nAE\nT\n", encoding="utf-8")
    return CMUDictLoader(dict_path, phones_path, symbols_path).load()


def test_word_and_text_to_phonemes(tmp_path: Path) -> None:
    loader = _fake_loader(tmp_path)
    assert word_to_phonemes("cat", loader) == ["K", "AE", "T"]
    assert text_to_phonemes("cat", loader) == ["K", "AE", "T"]


def test_extractors_and_pattern(tmp_path: Path) -> None:
    loader = _fake_loader(tmp_path)
    schema = PhonemeSchema(loader.phone_categories, loader.symbols)
    phonemes = ["K", "AE", "T"]
    assert extract_initial_phoneme(phonemes) == "K"
    assert extract_final_phoneme(phonemes) == "T"
    assert extract_vowel_phonemes(phonemes, schema) == ["AE"]
    assert classify_phoneme_pattern(phonemes, schema) == "CVC"

