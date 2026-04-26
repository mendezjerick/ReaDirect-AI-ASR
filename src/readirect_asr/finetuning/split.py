from __future__ import annotations

import random
from typing import Any

import pandas as pd


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    group_by_speaker: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    output = df.copy()
    if output.empty:
        output["split"] = []
        return output
    rng = random.Random(seed)
    if group_by_speaker and "speaker_id_anonymized" in output.columns and output["speaker_id_anonymized"].fillna("").astype(str).str.strip().ne("").any():
        groups = list(output["speaker_id_anonymized"].fillna("").astype(str).unique())
        rng.shuffle(groups)
        group_split = _assign_units(groups, train_ratio, val_ratio)
        output["split"] = output["speaker_id_anonymized"].fillna("").astype(str).map(group_split)
        return output
    indices = list(output.index)
    rng.shuffle(indices)
    index_split = _assign_units(indices, train_ratio, val_ratio)
    output["split"] = output.index.map(index_split)
    return output


def _assign_units(units: list[Any], train_ratio: float, val_ratio: float) -> dict[Any, str]:
    total = len(units)
    train_end = int(round(total * train_ratio))
    val_end = train_end + int(round(total * val_ratio))
    if total >= 3:
        train_end = min(max(train_end, 1), total - 2)
        val_end = min(max(val_end, train_end + 1), total - 1)
    result: dict[Any, str] = {}
    for position, unit in enumerate(units):
        if position < train_end:
            result[unit] = "train"
        elif position < val_end:
            result[unit] = "validation"
        else:
            result[unit] = "test"
    return result
