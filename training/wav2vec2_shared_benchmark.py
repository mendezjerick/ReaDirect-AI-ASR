from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.wav2vec2_alpha_data import (
    _duration_filter,
    _expand_env_value,
    load_local_manifest_dataset,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


SOURCE_ORDER = (
    "librispeech",
    "gigaspeech",
    "readirect_letters",
    "speechocean",
    "slr83",
)


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Shared benchmark config not found: {config_path}")
    return _expand_env_value(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


def _stable_select(dataset: Any, count: int, seed: int):
    if len(dataset) < count:
        raise RuntimeError(f"Requested {count} rows but source contains only {len(dataset)}.")
    ranked = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{dataset[index]['source_id']}".encode("utf-8")
        ).digest(),
    )
    return dataset.select(ranked[:count])


def _load_manifest_source(
    source_name: str,
    source_cfg: dict[str, Any],
):
    return load_local_manifest_dataset(
        source_cfg["manifest"],
        source_name,
        str(source_cfg["split"]),
        root_dir=source_cfg.get("root_dir"),
    )


def _load_gigaspeech_validation(
    source_cfg: dict[str, Any],
    *,
    count: int,
    seed: int,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Audio, Dataset

    parquet_dir = resolve_repo_path(source_cfg["parquet_dir"])
    files = sorted(parquet_dir.glob(str(source_cfg.get("file_glob", "validation-*.parquet"))))
    if not files:
        raise FileNotFoundError(
            f"GigaSpeech validation parquet is missing under {parquet_dir}. "
            "Run scripts/download_gigaspeech_validation.py manually."
        )

    ranked = []
    file_counts = []
    global_index = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        file_counts.append(parquet.metadata.num_rows)
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["segment_id"])
            for source_id in table.column("segment_id").to_pylist():
                rank = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
                ranked.append((rank, global_index, source_id))
                global_index += 1
    if len(ranked) < count:
        raise RuntimeError(f"GigaSpeech validation has only {len(ranked)} rows; need {count}.")
    selected = sorted(
        ((global_index, source_id) for _, global_index, source_id in sorted(ranked)[:count]),
        key=lambda item: item[0],
    )
    selected_offset = 0
    global_file_start = 0
    rows = []
    columns = ["audio", "text", "segment_id", "speaker", "begin_time", "end_time"]
    for path, file_count in zip(files, file_counts):
        global_file_end = global_file_start + file_count
        file_selected = []
        while selected_offset < len(selected) and selected[selected_offset][0] < global_file_end:
            file_selected.append(selected[selected_offset][0] - global_file_start)
            selected_offset += 1
        if file_selected:
            parquet = pq.ParquetFile(path)
            row_group_start = 0
            local_offset = 0
            for group in range(parquet.num_row_groups):
                row_group_rows = parquet.metadata.row_group(group).num_rows
                row_group_end = row_group_start + row_group_rows
                group_selected = []
                while (
                    local_offset < len(file_selected)
                    and file_selected[local_offset] < row_group_end
                ):
                    group_selected.append(file_selected[local_offset] - row_group_start)
                    local_offset += 1
                if group_selected:
                    table = parquet.read_row_group(group, columns=columns)
                    table = table.take(pa.array(group_selected, type=pa.int64()))
                    for row in table.to_pylist():
                        rows.append(
                            {
                                "audio": row["audio"],
                                "text": row["text"],
                                "dataset": "gigaspeech",
                                "split": str(source_cfg["split"]),
                                "source_id": row["segment_id"],
                                "speaker_id": row["speaker"],
                                "duration_seconds": max(
                                    0.0,
                                    float(row["end_time"]) - float(row["begin_time"]),
                                ),
                            }
                        )
                row_group_start = row_group_end
        global_file_start = global_file_end
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def _load_epsilon_heldout_slr83(source_cfg: dict[str, Any]):
    from datasets import Audio, Dataset

    manifest_path = resolve_repo_path(source_cfg["epsilon_split_manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Epsilon SLR83 split manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        {
            "audio": row["audio_path"],
            "text": row["transcript"],
            "dataset": "slr83",
            "split": str(source_cfg["split"]),
            "source_id": row["source_id"],
            "speaker_id": row.get("speaker_id", ""),
            "duration_seconds": None,
        }
        for row in manifest["rows"]
        if row["split"] == "evaluation"
    ]
    if not rows:
        raise RuntimeError("Epsilon SLR83 split contains no held-out evaluation rows.")
    return Dataset.from_list(rows).cast_column(
        "audio", Audio(sampling_rate=16000, num_channels=1)
    )


def build_shared_benchmark(config: dict[str, Any]):
    from datasets import concatenate_datasets

    benchmark_cfg = config["benchmark"]
    sources_cfg = config["sources"]
    count = int(benchmark_cfg.get("rows_per_source", 250))
    seed = int(config["run"].get("seed", 20260609))
    minimum = float(benchmark_cfg.get("min_duration_seconds", 0.2))
    maximum = float(benchmark_cfg.get("max_duration_seconds", 30.0))
    sources = {}
    for offset, source_name in enumerate(SOURCE_ORDER):
        source_cfg = sources_cfg[source_name]
        if source_name == "gigaspeech":
            dataset = _load_gigaspeech_validation(
                source_cfg,
                count=count,
                seed=seed + offset,
            )
        elif source_name == "slr83":
            dataset = _load_epsilon_heldout_slr83(source_cfg)
        else:
            dataset = _load_manifest_source(source_name, source_cfg)
        dataset = _duration_filter(dataset, minimum, maximum)
        sources[source_name] = (
            dataset
            if source_name == "gigaspeech"
            else _stable_select(dataset, count, seed + offset)
        )

    combined = concatenate_datasets([sources[name] for name in SOURCE_ORDER])
    summary = {
        "benchmark_name": config["run"]["name"],
        "seed": seed,
        "rows_per_source": count,
        "total_rows": len(combined),
        "source_order": list(SOURCE_ORDER),
        "sources": {
            name: {
                "rows": len(sources[name]),
                "split": str(sources_cfg[name]["split"]),
                "leakage": str(sources_cfg[name]["leakage"]),
                "source_ids": list(sources[name]["source_id"]),
            }
            for name in SOURCE_ORDER
        },
        "primary_metric": (
            "clean_macro_average_of_librispeech_gigaspeech_"
            "readirect_letters_and_speechocean"
        ),
        "diagnostic_metric": "macro_average_of_all_five_sources_including_slr83",
        "letter_metric": "readirect_letters_exact_match",
        "fairness_note": (
            "SLR83 uses Epsilon's deterministic speaker-held-out split. It is clean "
            "for Epsilon and models that did not train on SLR83, but contaminated "
            "for Delta because Delta trained on every SLR83 row."
        ),
    }
    output = resolve_repo_path(benchmark_cfg["manifest_summary"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return combined, summary
