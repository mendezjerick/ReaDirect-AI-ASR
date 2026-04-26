from pathlib import Path

from readirect_asr.content.loader import build_content_index
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def test_content_index_enriches_phonemes_and_handles_missing_words(tmp_path: Path) -> None:
    content = tmp_path / "content_bank" / "modules"
    content.mkdir(parents=True)
    (content / "module2_word_reading_activities.csv").write_text(
        "id,module_key,activity_type,prompt_text,expected_answer,accepted_answers,is_active\n"
        "M2-001,module_2,word_reading,Read cat,cat,cat,1\n"
        "M2-002,module_2,word_reading,Read zzz,zzzz,zzzz,1\n",
        encoding="utf-8",
    )
    cmu_dir = tmp_path / "cmu"
    cmu_dir.mkdir()
    (cmu_dir / "cmudict.dict").write_text("CAT K AE1 T\n", encoding="utf-8")
    (cmu_dir / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\n", encoding="utf-8")
    (cmu_dir / "cmudict.symbols").write_text("K\nAE\nT\n", encoding="utf-8")
    loader = CMUDictLoader(cmu_dir / "cmudict.dict", cmu_dir / "cmudict.phones", cmu_dir / "cmudict.symbols").load()

    index = build_content_index(tmp_path / "content_bank", loader, enrich_phonemes=True)

    assert index.get_item("M2-001").expected_phonemes == "K AE T"  # type: ignore[union-attr]
    assert index.get_item("M2-002").expected_phonemes is None  # type: ignore[union-attr]

