from pathlib import Path

import pandas as pd

import scripts.enrich_content_bank as script


def _cmu(tmp_path: Path) -> Path:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("CAT K AE1 T\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("K\nAE\nAE1\nT\n", encoding="utf-8")
    return cmu


def test_script_runs_on_tiny_content_index(tmp_path: Path) -> None:
    index = tmp_path / "content_index.csv"
    pd.DataFrame([{"prompt_id": "M2-001", "source_file": "module2_word_reading_activities.csv", "source_group": "modules", "module_key": "module_2", "activity_type": "read_word", "expected_text": "cat", "prompt_text": "Read cat"}]).to_csv(index, index=False)
    output = tmp_path / "out"
    df = script.enrich_content_bank(tmp_path / "empty", _cmu(tmp_path), output, content_index=index)
    assert (output / "enriched_content_index.csv").exists()
    assert df.loc[0, "expected_phonemes"] == "K AE T"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    index = tmp_path / "content_index.csv"
    pd.DataFrame([{"prompt_id": "M2-001", "source_file": "module2.csv", "source_group": "modules", "expected_text": "cat"}]).to_csv(index, index=False)
    output = tmp_path / "out"
    script.enrich_content_bank(tmp_path / "empty", _cmu(tmp_path), output, content_index=index, dry_run=True)
    assert not output.exists()

