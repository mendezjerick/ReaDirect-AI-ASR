from pathlib import Path

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader, load_cmudict_dict, normalize_word


def test_cmudict_loader_handles_alternates_and_stress(tmp_path: Path) -> None:
    dict_path = tmp_path / "cmudict.dict"
    phones_path = tmp_path / "cmudict.phones"
    symbols_path = tmp_path / "cmudict.symbols"
    dict_path.write_text("CAT K AE1 T\nCAT(1) K AE2 T\nBADLINE\n", encoding="utf-8")
    phones_path.write_text("AE vowel\nK stop\nT stop\n", encoding="utf-8")
    symbols_path.write_text("K\nAE\nAE1\nT\n", encoding="utf-8")

    loader = CMUDictLoader(dict_path, phones_path, symbols_path).load()

    assert normalize_word("CAT(1)") == "cat"
    assert loader.get_pronunciations("cat") == [["K", "AE", "T"], ["K", "AE", "T"]]
    assert loader.get_primary_pronunciation("CAT") == ["K", "AE", "T"]
    assert loader.get_pronunciations("missing") == []


def test_load_cmudict_dict_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_cmudict_dict(tmp_path / "missing.dict") == {}

