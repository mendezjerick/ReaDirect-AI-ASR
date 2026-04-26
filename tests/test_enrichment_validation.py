import pandas as pd

from readirect_asr.content.enrichment_schema import ENRICHMENT_COLUMNS
from readirect_asr.content.enrichment_validation import validate_enriched_dataframe, validate_required_enrichment_columns


def test_required_columns_checked() -> None:
    df = pd.DataFrame({"prompt_id": ["a"]})
    assert "source_file" in validate_required_enrichment_columns(df)


def test_invalid_skill_group_and_duplicate_prompt_id_detected() -> None:
    rows = [{column: "" for column in ENRICHMENT_COLUMNS} for _ in range(2)]
    for row in rows:
        row.update({"prompt_id": "dup", "skill_group": "invalid", "error_focus": "unknown", "difficulty_level": "easy", "practice_role": "practice", "recommended_for_error_type": "incorrect_general"})
    report = validate_enriched_dataframe(pd.DataFrame(rows))
    assert report["duplicate_prompt_ids"] == ["dup"]
    assert report["invalid_skill_groups"] == ["invalid"]

