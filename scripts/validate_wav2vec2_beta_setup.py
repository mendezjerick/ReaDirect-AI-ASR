from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import configure_windows_ffmpeg, dataset_distribution
from training.wav2vec2_beta_data import (
    build_beta_shared_dataset,
    build_beta_train_dataset,
    gigaspeech_s_parquet_files,
    load_beta_config,
    load_gigaspeech_s_train,
)
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Beta without training or full evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_beta.yaml"))
    parser.add_argument("--decode-sample", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    config = load_beta_config(args.config)
    ffmpeg_dir = configure_windows_ffmpeg()
    if ffmpeg_dir is None:
        errors.append("FFmpeg shared libraries were not found.")
    else:
        print(f"FFmpeg bin: {ffmpeg_dir}")

    alpha_path = resolve_repo_path(config["model"]["alpha_checkpoint_path"])
    required_alpha = ["config.json", "model.safetensors", "vocab.json", "processor_config.json"]
    missing_alpha = [name for name in required_alpha if not (alpha_path / name).exists()]
    if missing_alpha:
        errors.append(
            f"Alpha checkpoint is incomplete at {alpha_path}; missing {missing_alpha}. "
            "Set ALPHA_CHECKPOINT_PATH explicitly."
        )
    else:
        print(f"Alpha checkpoint: {alpha_path}")

    files = gigaspeech_s_parquet_files(config)
    if not files:
        errors.append("No local GigaSpeech S train parquet files were found.")
    else:
        print(f"GigaSpeech S parquet files: {len(files)}")
        print(f"GigaSpeech S local bytes: {sum(path.stat().st_size for path in files)}")

    for dataset_name in ("speechocean", "readirect_letters"):
        source = config["data"][dataset_name]
        for split in ("train", "validation", "test"):
            rows = read_jsonl(source[f"{split}_manifest"])
            if not rows:
                errors.append(f"{dataset_name} {split} manifest is missing or empty.")
            else:
                print(f"{dataset_name} {split}: {len(rows)} rows")

    try:
        print(f"TorchCodec package version: {importlib.metadata.version('torchcodec')}")
        import torchcodec

        print(f"TorchCodec import OK: {torchcodec.__version__}")
    except Exception as exc:
        errors.append(f"TorchCodec cannot load: {exc}")

    if not errors:
        from scripts.train_wav2vec2_beta import verify_processor_and_head

        verify_processor_and_head(
            alpha_path,
            resolve_repo_path(config["model"]["reference_base_model_path"]),
        )
        print("Tokenizer mapping and CTC head shape match base-960h.")

        raw_train = build_beta_train_dataset(config)
        distribution = dataset_distribution(raw_train)
        print(f"Beta train distribution: {json.dumps(distribution, sort_keys=True)}")
        letter_ids = [
            source_id
            for dataset_name, source_id in zip(raw_train["dataset"], raw_train["source_id"])
            if dataset_name == "readirect_letters"
        ]
        counts = Counter(letter_ids)
        if len(counts) != len(letter_ids) or max(counts.values(), default=0) > 1:
            errors.append("ReaDirect letters are duplicated or oversampled in the Beta train mix.")
        else:
            print(f"ReaDirect letter rows are unique: {len(letter_ids)}")
        shared = build_beta_shared_dataset(config, "validation")
        print(f"Shared validation rows: {len(shared)}")

        if args.decode_sample:
            sample = load_gigaspeech_s_train(config, max_samples=1).select([0])[0]["audio"]
            print(
                f"Decoded GigaSpeech S sample: samples={len(sample['array'])}, "
                f"sample_rate={sample['sampling_rate']}"
            )
        else:
            print("WARNING: audio decoding not tested; rerun with --decode-sample.")

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("Beta setup validation failed. Training and evaluation were not started.")
        return 1
    print("Beta setup validation passed. Training and evaluation were not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
