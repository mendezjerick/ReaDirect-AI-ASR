import pandas as pd

from readirect_asr.finetuning.split import create_splits


def test_creates_train_validation_test_splits():
    df = pd.DataFrame({"recording_id": list(range(10))})
    result = create_splits(df, seed=1)
    assert set(result["split"]).issubset({"train", "validation", "test"})
    assert {"train", "validation", "test"}.issubset(set(result["split"]))


def test_speaker_disjoint_split_when_speaker_ids_exist():
    df = pd.DataFrame({"speaker_id_anonymized": ["A", "A", "B", "B", "C", "C", "D", "D"]})
    result = create_splits(df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=2)
    speaker_splits = result.groupby("speaker_id_anonymized")["split"].nunique()
    assert speaker_splits.max() == 1


def test_deterministic_with_seed():
    df = pd.DataFrame({"recording_id": list(range(20))})
    first = create_splits(df, seed=123)
    second = create_splits(df, seed=123)
    assert first["split"].tolist() == second["split"].tolist()
