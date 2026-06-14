from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import (
    _duration_filter,
    _expand_env_value,
    load_local_manifest_dataset,
    prepare_alpha_dataset,
)
from training.wav2vec2_delta_data import (
    SOURCE_GIGASPEECH,
    SOURCE_LETTERS,
    SOURCE_SLR83,
    _manifest_valid_rows,
    _sample_to_count,
    load_delta_gigaspeech_sample,
    parse_slr83_rows,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


SOURCE_SPEECHOCEAN = "speechocean"
PUNCTUATION_TAGS = {
    "APOSTROPHE", "COLON", "COMMA", "DASH", "EXCLAMATIONPOINT",
    "HYPHEN", "PERIOD", "QUESTIONMARK", "SEMICOLON",
}
TAG_PATTERN = re.compile(r"<([^>]+)>")


def load_epsilon_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Epsilon config not found: {config_path}")
    return _expand_env_value(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


def _delta_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "data": {
            **config["data"],
            "slr83": config["data"]["slr83"],
        },
    }


def split_slr83_rows(
    config: dict[str, Any],
    *,
    write_manifest: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, parse_summary = parse_slr83_rows(_delta_compatible_config(config))
    slr_cfg = config["data"]["slr83"]
    seed = int(slr_cfg.get("split_seed", config["run"].get("seed", 47)))
    train_ratio = float(slr_cfg.get("train_ratio", 0.8))
    if not 0.5 <= train_ratio < 1:
        raise RuntimeError("SLR83 train_ratio must be at least 0.5 and below 1.0.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    if len(groups) < 2:
        raise RuntimeError("SLR83 speaker-aware split requires at least two speaker groups.")

    by_gender: dict[str, list[str]] = defaultdict(list)
    for speaker_id, group_rows in groups.items():
        by_gender[group_rows[0]["slr83_source"]].append(speaker_id)
    eval_speakers: set[str] = set()
    rng = random.Random(seed)
    for gender, speakers in sorted(by_gender.items()):
        speakers = sorted(speakers)
        rng.shuffle(speakers)
        eval_count = max(1, round(len(speakers) * (1.0 - train_ratio)))
        if eval_count >= len(speakers):
            eval_count = len(speakers) - 1
        eval_speakers.update(speakers[:eval_count])

    train_rows = [row for row in rows if row["speaker_id"] not in eval_speakers]
    eval_rows = [row for row in rows if row["speaker_id"] in eval_speakers]
    train_ids = {row["source_id"] for row in train_rows}
    eval_ids = {row["source_id"] for row in eval_rows}
    if train_ids & eval_ids:
        raise RuntimeError("SLR83 train/evaluation source IDs overlap.")
    if {row["speaker_id"] for row in train_rows} & {row["speaker_id"] for row in eval_rows}:
        raise RuntimeError("SLR83 train/evaluation speakers overlap.")

    manifest_rows = []
    for row in sorted(rows, key=lambda item: item["source_id"]):
        manifest_rows.append(
            {
                "source_id": row["source_id"],
                "audio_path": row["audio"],
                "transcript": row["text"],
                "gender_folder": row["slr83_source"],
                "speaker_id": row["speaker_id"],
                "group_id": row["speaker_id"],
                "split": "evaluation" if row["speaker_id"] in eval_speakers else "train",
            }
        )
    summary = {
        "strategy": "deterministic_speaker_aware",
        "seed": seed,
        "requested_train_ratio": train_ratio,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "evaluation_rows": len(eval_rows),
        "train_ratio": len(train_rows) / len(rows),
        "evaluation_ratio": len(eval_rows) / len(rows),
        "train_speakers": len({row["speaker_id"] for row in train_rows}),
        "evaluation_speakers": len(eval_speakers),
        "speaker_overlap": 0,
        "source_id_overlap": 0,
        "parse_summary": parse_summary,
        "rows": manifest_rows,
    }
    if write_manifest:
        report_dir = resolve_repo_path(config["model"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "slr83_split_manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return train_rows, eval_rows, summary


def _rows_to_dataset(rows: list[dict[str, Any]], split: str):
    from datasets import Audio, Dataset

    standardized = [
        {
            "audio": row["audio"],
            "text": row["text"],
            "dataset": SOURCE_SLR83,
            "split": split,
            "source_id": row["source_id"],
            "speaker_id": row["speaker_id"],
            "duration_seconds": row["duration_seconds"],
        }
        for row in rows
    ]
    return Dataset.from_list(standardized).cast_column(
        "audio", Audio(sampling_rate=16000, num_channels=1)
    )


def effective_counts(config: dict[str, Any]) -> dict[str, int]:
    sampling = config["data"]["sampling"]
    total = int(sampling.get("effective_epoch_rows", 40000))
    ratios = sampling["ratios"]
    sources = (SOURCE_GIGASPEECH, SOURCE_SLR83, SOURCE_SPEECHOCEAN)
    counts = {source: round(total * float(ratios[source])) for source in sources}
    counts[SOURCE_LETTERS] = total - sum(counts.values())
    expected = {
        SOURCE_GIGASPEECH: 0.45,
        SOURCE_SLR83: 0.30,
        SOURCE_SPEECHOCEAN: 0.15,
        SOURCE_LETTERS: 0.10,
    }
    if any(abs(float(ratios[key]) - value) > 1e-9 for key, value in expected.items()):
        raise RuntimeError(f"Epsilon ratios must remain 45/30/15/10, received {ratios}.")
    return counts


def scan_gigaspeech_clean_candidates(
    config: dict[str, Any],
    vocab: set[str],
) -> tuple[list[int], dict[str, Any]]:
    import pyarrow.parquet as pq

    giga_cfg = config["data"]["gigaspeech"]
    parquet_dir = resolve_repo_path(giga_cfg["parquet_dir"])
    files = sorted(parquet_dir.glob(str(giga_cfg.get("train_glob", "train-*.parquet"))))
    if not files:
        raise FileNotFoundError(f"No GigaSpeech parquet files found under {parquet_dir}")
    minimum = float(config["data"].get("min_duration_seconds", 0.3))
    maximum = float(config["data"].get("max_duration_seconds", 25.0))
    accepted: list[int] = []
    rejected: Counter[str] = Counter()
    global_index = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["audio", "text", "begin_time", "end_time"],
            batch_size=4096,
        ):
            for row in batch.to_pylist():
                text = str(row.get("text") or "").strip()
                duration = float(row["end_time"]) - float(row["begin_time"])
                reason = None
                if row.get("audio") is None:
                    reason = "missing_audio_payload"
                elif not text:
                    reason = "empty_transcript"
                elif not minimum <= duration <= maximum:
                    reason = "duration_out_of_range"
                else:
                    tags = {tag.upper() for tag in TAG_PATTERN.findall(text)}
                    unsupported = tags - PUNCTUATION_TAGS
                    if unsupported:
                        reason = "unsupported_or_noise_tag"
                    elif not normalize_asr_text(text, vocab):
                        reason = "empty_after_normalization"
                if reason:
                    rejected[reason] += 1
                else:
                    accepted.append(global_index)
                global_index += 1
    return accepted, {
        "raw_rows": global_index,
        "eligible_rows": len(accepted),
        "excluded_rows": sum(rejected.values()),
        "excluded_by_reason": dict(sorted(rejected.items())),
        "parquet_files": len(files),
        "duration_range_seconds": [minimum, maximum],
    }


def _load_selected_gigaspeech(config: dict[str, Any], selected: list[int]):
    # Reuse Delta's deterministic parquet loader by reproducing its selected set
    # through a temporary seed-independent Dataset generator.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Audio, Dataset

    giga_cfg = config["data"]["gigaspeech"]
    parquet_dir = resolve_repo_path(giga_cfg["parquet_dir"])
    files = sorted(parquet_dir.glob(str(giga_cfg.get("train_glob", "train-*.parquet"))))
    file_counts = [pq.ParquetFile(path).metadata.num_rows for path in files]

    def generate():
        selected_offset = 0
        global_start = 0
        columns = ["audio", "text", "segment_id", "speaker", "begin_time", "end_time"]
        for path, file_count in zip(files, file_counts):
            global_end = global_start + file_count
            local = []
            while selected_offset < len(selected) and selected[selected_offset] < global_end:
                local.append(selected[selected_offset] - global_start)
                selected_offset += 1
            if local:
                parquet = pq.ParquetFile(path)
                row_group_start = 0
                local_offset = 0
                for row_group in range(parquet.num_row_groups):
                    row_group_rows = parquet.metadata.row_group(row_group).num_rows
                    row_group_end = row_group_start + row_group_rows
                    row_group_selected = []
                    while (
                        local_offset < len(local)
                        and local[local_offset] < row_group_end
                    ):
                        row_group_selected.append(
                            local[local_offset] - row_group_start
                        )
                        local_offset += 1
                    if row_group_selected:
                        table = parquet.read_row_group(row_group, columns=columns)
                        table = table.take(
                            pa.array(row_group_selected, type=pa.int64())
                        )
                        for row in table.to_pylist():
                            yield {
                                "audio": row["audio"],
                                "text": row["text"],
                                "dataset": SOURCE_GIGASPEECH,
                                "split": "train",
                                "source_id": row["segment_id"],
                                "speaker_id": row["speaker"],
                                "duration_seconds": (
                                    float(row["end_time"]) - float(row["begin_time"])
                                ),
                            }
                    row_group_start = row_group_end
            global_start = global_end

    cache = resolve_repo_path(config["model"]["report_dir"]) / "dataset_cache" / "gigaspeech"
    cache.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(generate, cache_dir=str(cache), keep_in_memory=False)
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def plan_epsilon_dataset_mix(
    config: dict[str, Any],
    vocab: set[str],
    *,
    write_summary: bool = True,
) -> dict[str, Any]:
    counts = effective_counts(config)
    slr_train, slr_eval, split_summary = split_slr83_rows(config, write_manifest=True)
    candidates, giga_summary = scan_gigaspeech_clean_candidates(config, vocab)
    speech_cfg = config["data"]["speechocean"]
    letters_cfg = config["data"]["readirect_letters"]
    raw_counts = {
        SOURCE_GIGASPEECH: giga_summary["raw_rows"],
        SOURCE_SLR83: split_summary["total_rows"],
        SOURCE_SPEECHOCEAN: _manifest_valid_rows(speech_cfg["train_manifest"]),
        SOURCE_LETTERS: _manifest_valid_rows(
            letters_cfg["train_manifest"], root_dir=letters_cfg["root_dir"]
        ),
    }
    if len(candidates) < counts[SOURCE_GIGASPEECH]:
        raise RuntimeError("Not enough clean GigaSpeech rows for Epsilon's effective epoch.")
    summary = {
        "run_name": "Epsilon",
        "sampling_strategy": "deterministic_virtual_epoch",
        "raw_counts": raw_counts,
        "filtered_counts": {
            SOURCE_GIGASPEECH: len(candidates),
            SOURCE_SLR83: len(slr_train),
            SOURCE_SPEECHOCEAN: raw_counts[SOURCE_SPEECHOCEAN],
            SOURCE_LETTERS: raw_counts[SOURCE_LETTERS],
        },
        "effective_counts": counts,
        "effective_ratios": {
            source: count / sum(counts.values()) for source, count in counts.items()
        },
        "train_rows": sum(counts.values()),
        "validation_counts": {
            SOURCE_SPEECHOCEAN: _manifest_valid_rows(speech_cfg["validation_manifest"]),
            SOURCE_LETTERS: _manifest_valid_rows(
                letters_cfg["validation_manifest"], root_dir=letters_cfg["root_dir"]
            ),
            "slr83_heldout_evaluation": len(slr_eval),
        },
        "slr83_split": {
            key: value for key, value in split_summary.items() if key != "rows"
        },
        "gigaspeech_filtering": giga_summary,
        "librispeech_training_rows": 0,
        "slr83_heldout_used_for_training": False,
        "slr83_heldout_used_for_early_stopping": False,
    }
    if write_summary:
        report_dir = resolve_repo_path(config["model"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "dataset_mix_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def build_epsilon_train_dataset(config: dict[str, Any], vocab: set[str]):
    from datasets import concatenate_datasets

    seed = int(config["run"].get("seed", 47))
    counts = effective_counts(config)
    slr_train, slr_eval, _ = split_slr83_rows(config, write_manifest=True)
    heldout_ids = {row["source_id"] for row in slr_eval}
    candidates, _ = scan_gigaspeech_clean_candidates(config, vocab)
    selected = sorted(random.Random(seed).sample(candidates, counts[SOURCE_GIGASPEECH]))
    giga = _load_selected_gigaspeech(config, selected)
    slr = _rows_to_dataset(slr_train, "train")
    speech_cfg = config["data"]["speechocean"]
    speech = load_local_manifest_dataset(
        speech_cfg["train_manifest"], SOURCE_SPEECHOCEAN, "train"
    )
    letters_cfg = config["data"]["readirect_letters"]
    letters = load_local_manifest_dataset(
        letters_cfg["train_manifest"],
        SOURCE_LETTERS,
        "train",
        root_dir=letters_cfg["root_dir"],
    )
    minimum = float(config["data"].get("min_duration_seconds", 0.3))
    maximum = float(config["data"].get("max_duration_seconds", 25.0))
    sources = {
        SOURCE_GIGASPEECH: _duration_filter(giga, minimum, maximum),
        SOURCE_SLR83: _duration_filter(slr, minimum, maximum),
        SOURCE_SPEECHOCEAN: _duration_filter(speech, minimum, maximum),
        SOURCE_LETTERS: _duration_filter(letters, minimum, maximum),
    }
    effective = {
        source: _sample_to_count(dataset, counts[source], seed + offset)
        for offset, (source, dataset) in enumerate(sources.items())
    }
    if heldout_ids & set(effective[SOURCE_SLR83]["source_id"]):
        raise RuntimeError("Held-out SLR83 rows entered Epsilon training. Training refused.")
    combined = concatenate_datasets(list(effective.values()))
    if config["data"]["sampling"].get("shuffle", True):
        combined = combined.shuffle(seed=seed)
    summary = plan_epsilon_dataset_mix(config, vocab, write_summary=False)
    summary["selected_source_counts_before_virtual_sampling"] = {
        key: len(value) for key, value in sources.items()
    }
    report_dir = resolve_repo_path(config["model"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_mix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return combined, summary


def build_epsilon_shared_dataset(config: dict[str, Any], split: str = "validation"):
    from datasets import concatenate_datasets

    key = "validation" if split == "validation" else split
    speech_cfg = config["data"]["speechocean"]
    letters_cfg = config["data"]["readirect_letters"]
    sources = [
        load_local_manifest_dataset(
            speech_cfg[f"{key}_manifest"], SOURCE_SPEECHOCEAN, split
        ),
        load_local_manifest_dataset(
            letters_cfg[f"{key}_manifest"],
            SOURCE_LETTERS,
            split,
            root_dir=letters_cfg["root_dir"],
        ),
    ]
    minimum = float(config["data"].get("min_duration_seconds", 0.3))
    maximum = float(config["data"].get("max_duration_seconds", 25.0))
    return concatenate_datasets(
        [_duration_filter(source, minimum, maximum) for source in sources]
    )


def build_epsilon_slr83_heldout(config: dict[str, Any]):
    _, eval_rows, _ = split_slr83_rows(config, write_manifest=True)
    minimum = float(config["data"].get("min_duration_seconds", 0.3))
    maximum = float(config["data"].get("max_duration_seconds", 25.0))
    return _duration_filter(_rows_to_dataset(eval_rows, "evaluation"), minimum, maximum)


def prepare_epsilon_dataset(dataset: Any, processor: Any, config: dict[str, Any]):
    return prepare_alpha_dataset(dataset, processor, config)
