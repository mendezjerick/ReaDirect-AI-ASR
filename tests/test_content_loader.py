from pathlib import Path

import pandas as pd

from readirect_asr.content.loader import build_content_index, load_assessment_content
from readirect_asr.content.validation import validate_csv_columns


def test_loads_content_and_builds_index(tmp_path: Path) -> None:
    assessment = tmp_path / "content_bank" / "assessment"
    assessment.mkdir(parents=True)
    (assessment / "task1_letter_pronunciation.csv").write_text(
        "id,sequence,content_type,prompt_text,expected_answer,accepted_answers,is_active\n"
        "T1-001,1,letter,What sound?,cat,cat|kat,1\n",
        encoding="utf-8",
    )

    loaded = load_assessment_content(tmp_path / "content_bank")
    assert "task1_letter_pronunciation.csv" in loaded

    index = build_content_index(tmp_path / "content_bank", enrich_phonemes=False)
    item = index.get_item("T1-001")
    assert item is not None
    assert item.expected_text == "cat"


def test_validate_columns_and_duplicate_prompt_ids(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": ["A"]})
    report = validate_csv_columns(df, ["id", "expected_answer"], "fake.csv")
    assert report["missing_columns"] == ["expected_answer"]

    assessment = tmp_path / "content_bank" / "assessment"
    assessment.mkdir(parents=True)
    (assessment / "task1_letter_pronunciation.csv").write_text(
        "id,prompt_text,expected_answer,accepted_answers,is_active\n"
        "DUP,prompt,cat,cat,1\n"
        "DUP,prompt,dog,dog,1\n",
        encoding="utf-8",
    )
    index = build_content_index(tmp_path / "content_bank", enrich_phonemes=False)
    assert index.duplicate_prompt_ids() == ["DUP"]

