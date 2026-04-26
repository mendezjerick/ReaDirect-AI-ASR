from pathlib import Path

import pandas as pd

from readirect_asr.content.content_repository import ContentRepository


def test_loads_content_index_and_lookup_by_prompt_id(tmp_path: Path) -> None:
    index = tmp_path / "content_index.csv"
    pd.DataFrame([{"prompt_id": "M2-001", "expected_text": "cat", "accepted_answers": "cat"}]).to_csv(index, index=False)
    repo = ContentRepository(content_index_path=index, enriched_content_index_path=tmp_path / "missing.csv").load()
    assert repo.is_loaded()
    assert repo.get_by_prompt_id("M2-001")["expected_text"] == "cat"


def test_prefers_enriched_index_if_available(tmp_path: Path) -> None:
    base = tmp_path / "content_index.csv"
    enriched = tmp_path / "enriched.csv"
    pd.DataFrame([{"prompt_id": "A", "expected_text": "base"}]).to_csv(base, index=False)
    pd.DataFrame([{"prompt_id": "A", "expected_text": "enriched", "skill_group": "word_reading"}]).to_csv(enriched, index=False)
    repo = ContentRepository(base, enriched, prefer_enriched_content=True).load()
    assert repo.loaded_path == enriched
    assert repo.get_metadata("A")["expected_text"] == "enriched"


def test_missing_index_handled_gracefully(tmp_path: Path) -> None:
    repo = ContentRepository(tmp_path / "missing.csv", tmp_path / "missing2.csv").load()
    assert not repo.is_loaded()
    assert repo.get_metadata("x") == {}

