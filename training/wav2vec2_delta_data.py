from __future__ import annotations

import csv
import json
import math
import random
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import (
    _duration_filter,
    _expand_env_value,
    load_cached_gigaspeech_split,
    load_local_manifest_dataset,
    prepare_alpha_dataset,
)
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


SOURCE_GIGASPEECH = "gigaspeech"
SOURCE_SLR83 = "slr83_southern_english"
SOURCE_LETTERS = "readirect_letters"


def load_delta_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Delta config not found: {config_path}")
    return _expand_env_value(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def parse_slr83_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slr_cfg = config["data"]["slr83"]
    root = resolve_repo_path(slr_cfg["root_dir"])
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    speakers: set[str] = set()
    missing_audio: list[str] = []
    empty_labels: list[str] = []

    for source_name, key in (
        ("southern_english_female", "female_dir"),
        ("southern_english_male", "male_dir"),
    ):
        directory = root / str(slr_cfg[key])
        index_path = directory / str(slr_cfg.get("index_file", "line_index.csv"))
        if not index_path.exists():
            raise FileNotFoundError(f"SLR83 index not found: {index_path}")
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, fields in enumerate(csv.reader(handle), start=1):
                if len(fields) < 3:
                    raise RuntimeError(f"Malformed SLR83 index row {index_path}:{line_number}")
                prompt_id = fields[0].strip()
                audio_id = fields[1].strip()
                transcript = ",".join(fields[2:]).strip()
                audio_path = directory / f"{audio_id}.wav"
                if not transcript:
                    empty_labels.append(audio_id)
                    continue
                if not audio_path.exists():
                    missing_audio.append(str(audio_path))
                    continue
                speaker_code = audio_id.split("_", 2)[1] if "_" in audio_id else audio_id
                speaker_id = f"{source_name}_{speaker_code}"
                rows.append(
                    {
                        "audio": str(audio_path),
                        "text": transcript,
                        "dataset": SOURCE_SLR83,
                        "split": "train",
                        "source_id": audio_id,
                        "speaker_id": speaker_id,
                        "duration_seconds": _wav_duration(audio_path),
                        "slr83_source": source_name,
                        "prompt_id": prompt_id,
                    }
                )
                source_counts[source_name] += 1
                speakers.add(speaker_id)

    if missing_audio or empty_labels:
        raise RuntimeError(
            f"SLR83 parsing failed: missing_audio={len(missing_audio)}, "
            f"empty_labels={len(empty_labels)}"
        )
    summary = {
        "root": str(root),
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "speaker_count": len(speakers),
        "missing_audio": 0,
        "empty_labels": 0,
    }
    return rows, summary


def load_slr83_dataset(config: dict[str, Any]):
    from datasets import Audio, Dataset

    rows, _ = parse_slr83_rows(config)
    dataset = Dataset.from_list(rows)
    dataset = dataset.select_columns(
        ["audio", "text", "dataset", "split", "source_id", "speaker_id", "duration_seconds"]
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def effective_counts(config: dict[str, Any]) -> dict[str, int]:
    sampling = config["data"]["sampling"]
    total = int(sampling.get("effective_epoch_rows", 40000))
    ratios = sampling["ratios"]
    counts = {
        SOURCE_GIGASPEECH: int(round(total * float(ratios[SOURCE_GIGASPEECH]))),
        SOURCE_SLR83: int(round(total * float(ratios[SOURCE_SLR83]))),
    }
    counts[SOURCE_LETTERS] = total - sum(counts.values())
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"Delta effective counts must be positive: {counts}")
    return counts


def _sample_to_count(dataset: Any, target: int, seed: int):
    from datasets import concatenate_datasets

    if not len(dataset):
        raise RuntimeError("Cannot sample an empty Delta source dataset.")
    rng = random.Random(seed)
    pieces = []
    remaining = target
    cycle = 0
    while remaining:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        take = min(remaining, len(indices))
        pieces.append(dataset.select(indices[:take]))
        remaining -= take
        cycle += 1
    return pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces)


def _filter_source(dataset: Any, config: dict[str, Any]):
    minimum = float(config["data"].get("min_duration_seconds", 0.2))
    maximum = float(config["data"].get("max_duration_seconds", 30.0))
    return _duration_filter(dataset, minimum, maximum)


def _gigaspeech_raw_count(config: dict[str, Any]) -> tuple[int, int]:
    import pyarrow.parquet as pq

    giga_cfg = config["data"]["gigaspeech"]
    parquet_dir = resolve_repo_path(giga_cfg["parquet_dir"])
    files = sorted(parquet_dir.glob(str(giga_cfg.get("train_glob", "train-*.parquet"))))
    if not files:
        raise FileNotFoundError(f"No GigaSpeech S parquet files found under {parquet_dir}")
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files), len(files)


def load_delta_gigaspeech_sample(
    config: dict[str, Any],
    *,
    max_samples: int,
    seed: int,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Audio, Dataset

    giga_cfg = config["data"]["gigaspeech"]
    parquet_dir = resolve_repo_path(giga_cfg["parquet_dir"])
    files = sorted(parquet_dir.glob(str(giga_cfg.get("train_glob", "train-*.parquet"))))
    if not files:
        raise FileNotFoundError(f"No GigaSpeech S parquet files found under {parquet_dir}")
    file_counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
    total_rows = sum(file_counts)
    if max_samples > total_rows:
        raise RuntimeError(
            f"Requested {max_samples} GigaSpeech rows but only {total_rows} are available."
        )
    selected = sorted(random.Random(seed).sample(range(total_rows), max_samples))

    def generate():
        selected_offset = 0
        global_file_start = 0
        columns = ["audio", "text", "segment_id", "speaker", "begin_time", "end_time"]
        for path, file_count in zip(files, file_counts):
            global_file_end = global_file_start + file_count
            file_selected = []
            while selected_offset < len(selected) and selected[selected_offset] < global_file_end:
                file_selected.append(selected[selected_offset] - global_file_start)
                selected_offset += 1
            if file_selected:
                parquet = pq.ParquetFile(path)
                row_group_start = 0
                local_offset = 0
                for row_group in range(parquet.num_row_groups):
                    row_group_rows = parquet.metadata.row_group(row_group).num_rows
                    row_group_end = row_group_start + row_group_rows
                    row_group_selected = []
                    while (
                        local_offset < len(file_selected)
                        and file_selected[local_offset] < row_group_end
                    ):
                        row_group_selected.append(
                            file_selected[local_offset] - row_group_start
                        )
                        local_offset += 1
                    if row_group_selected:
                        table = parquet.read_row_group(row_group, columns=columns)
                        table = table.take(pa.array(row_group_selected, type=pa.int64()))
                        for row in table.to_pylist():
                            yield {
                                "audio": row["audio"],
                                "text": row["text"],
                                "dataset": SOURCE_GIGASPEECH,
                                "split": "train",
                                "source_id": row["segment_id"],
                                "speaker_id": row["speaker"],
                                "duration_seconds": max(
                                    0.0,
                                    float(row["end_time"]) - float(row["begin_time"]),
                                ),
                            }
                    row_group_start = row_group_end
            global_file_start = global_file_end

    cache_dir = resolve_repo_path(config["model"]["report_dir"]) / "dataset_cache" / "gigaspeech"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        generate,
        cache_dir=str(cache_dir),
        keep_in_memory=False,
        fingerprint=f"delta-gigaspeech-{seed}-{max_samples}",
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def _manifest_valid_rows(
    manifest: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> int:
    count = 0
    root = resolve_repo_path(root_dir) if root_dir else None
    for row in read_jsonl(manifest):
        raw_path = Path(str(row.get("audio_path", "")).strip())
        if raw_path.is_absolute():
            path = raw_path
        elif root and (root / raw_path).exists():
            path = root / raw_path
        else:
            path = resolve_repo_path(raw_path)
        if path.exists() and str(row.get("text") or row.get("letter") or "").strip():
            count += 1
    return count


def plan_delta_dataset_mix(
    config: dict[str, Any],
    *,
    write_summary: bool = True,
) -> dict[str, Any]:
    counts = effective_counts(config)
    giga_rows, giga_files = _gigaspeech_raw_count(config)
    slr_rows, slr_summary = parse_slr83_rows(config)
    letters_cfg = config["data"]["readirect_letters"]
    letter_rows = _manifest_valid_rows(
        letters_cfg["train_manifest"], root_dir=letters_cfg["root_dir"]
    )
    speech_cfg = config["data"]["evaluation_speechocean"]
    validation_rows = _manifest_valid_rows(speech_cfg["validation_manifest"])
    validation_rows += _manifest_valid_rows(
        letters_cfg["validation_manifest"], root_dir=letters_cfg["root_dir"]
    )
    if counts[SOURCE_SLR83] < len(slr_rows):
        raise RuntimeError(
            f"Effective SLR83 count {counts[SOURCE_SLR83]} would exclude rows; "
            f"increase DELTA_EFFECTIVE_EPOCH_ROWS to at least "
            f"{math.ceil(len(slr_rows) / 0.40)}."
        )
    summary = {
        "run_name": "Delta",
        "sampling_strategy": "deterministic_virtual_epoch",
        "raw_counts": {
            SOURCE_GIGASPEECH: giga_rows,
            SOURCE_SLR83: len(slr_rows),
            SOURCE_LETTERS: letter_rows,
        },
        "effective_counts": counts,
        "effective_ratios": {
            source: count / sum(counts.values()) for source, count in counts.items()
        },
        "source_names": list(counts),
        "train_rows": sum(counts.values()),
        "validation_rows": validation_rows,
        "speechocean_training_rows": 0,
        "gigaspeech_parquet_files": giga_files,
        "slr83": slr_summary,
        "slr83_all_rows_included_before_repetition": counts[SOURCE_SLR83] >= len(slr_rows),
    }
    if write_summary:
        report_dir = resolve_repo_path(config["model"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "dataset_mix_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def build_delta_train_dataset(
    config: dict[str, Any],
    *,
    write_summary: bool = True,
):
    from datasets import concatenate_datasets

    seed = int(config["run"].get("seed", 45))
    counts = effective_counts(config)
    slr83 = _filter_source(load_slr83_dataset(config), config)
    letters_cfg = config["data"]["readirect_letters"]
    letters = _filter_source(
        load_local_manifest_dataset(
            letters_cfg["train_manifest"],
            SOURCE_LETTERS,
            "train",
            root_dir=letters_cfg["root_dir"],
        ),
        config,
    )
    if counts[SOURCE_SLR83] < len(slr83):
        raise RuntimeError(
            f"Effective SLR83 count {counts[SOURCE_SLR83]} would exclude rows; "
            f"increase DELTA_EFFECTIVE_EPOCH_ROWS to at least "
            f"{math.ceil(len(slr83) / 0.40)}."
        )
    gigaspeech = _filter_source(
        load_delta_gigaspeech_sample(
            config,
            max_samples=counts[SOURCE_GIGASPEECH],
            seed=seed,
        ),
        config,
    )
    effective = {
        SOURCE_GIGASPEECH: _sample_to_count(gigaspeech, counts[SOURCE_GIGASPEECH], seed),
        SOURCE_SLR83: _sample_to_count(slr83, counts[SOURCE_SLR83], seed + 1),
        SOURCE_LETTERS: _sample_to_count(letters, counts[SOURCE_LETTERS], seed + 2),
    }
    combined = concatenate_datasets(list(effective.values()))
    if bool(config["data"]["sampling"].get("shuffle", True)):
        combined = combined.shuffle(seed=seed)
    summary = plan_delta_dataset_mix(config, write_summary=False)
    summary["filtered_raw_counts"] = {
        SOURCE_GIGASPEECH: len(gigaspeech),
        SOURCE_SLR83: len(slr83),
        SOURCE_LETTERS: len(letters),
    }
    summary["train_rows"] = len(combined)
    if write_summary:
        report_dir = resolve_repo_path(config["model"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "dataset_mix_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return combined, summary


def build_delta_shared_dataset(
    config: dict[str, Any],
    split: str = "validation",
    *,
    include_gigaspeech: bool = False,
):
    from datasets import concatenate_datasets

    split_key = "validation" if split == "validation" else split
    data_cfg = config["data"]
    speech_cfg = data_cfg["evaluation_speechocean"]
    letters_cfg = data_cfg["readirect_letters"]
    sources = [
        load_local_manifest_dataset(
            speech_cfg[f"{split_key}_manifest"], "speechocean", split
        ),
        load_local_manifest_dataset(
            letters_cfg[f"{split_key}_manifest"],
            SOURCE_LETTERS,
            split,
            root_dir=letters_cfg["root_dir"],
        ),
    ]
    if include_gigaspeech:
        giga_split = str(data_cfg["gigaspeech"].get("validation_split", "validation"))
        try:
            sources.insert(
                0,
                load_cached_gigaspeech_split(data_cfg["gigaspeech"]["cache_dir"], giga_split),
            )
        except FileNotFoundError as exc:
            print(
                "GigaSpeech validation cache is unavailable; continuing with the "
                f"historical shared set only. Details: {exc}"
            )
    return concatenate_datasets(
        [_filter_source(source, config) for source in sources if len(source)]
    )


def validate_normalized_slr83(config: dict[str, Any], vocab: set[str]) -> dict[str, Any]:
    rows, summary = parse_slr83_rows(config)
    empty_after_normalization = []
    changed_examples = []
    for row in rows:
        normalized = normalize_asr_text(row["text"], vocab)
        if not normalized:
            empty_after_normalization.append(row["source_id"])
        elif normalized != row["text"]:
            if len(changed_examples) < 5:
                changed_examples.append(
                    {
                        "source_id": row["source_id"],
                        "original": row["text"],
                        "normalized": normalized,
                    }
                )
    summary["empty_after_normalization"] = len(empty_after_normalization)
    summary["normalization_examples"] = changed_examples
    if empty_after_normalization:
        raise RuntimeError(
            f"{len(empty_after_normalization)} SLR83 transcripts became empty after normalization."
        )
    return summary


def prepare_delta_dataset(dataset: Any, processor: Any, config: dict[str, Any]):
    return prepare_alpha_dataset(dataset, processor, config)
