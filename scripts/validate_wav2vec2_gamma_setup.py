from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_beta_data import gigaspeech_s_parquet_files
from training.wav2vec2_gamma_data import (
    build_gamma_shared_dataset,
    configured_row_limit,
    load_gamma_config,
    load_gamma_gigaspeech_train,
)
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gamma without training or full evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_gamma.yaml"))
    parser.add_argument("--decode-sample", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    config = load_gamma_config(args.config)
    ffmpeg_dir = configure_windows_ffmpeg()
    if ffmpeg_dir is None:
        errors.append("FFmpeg shared libraries were not found.")
    else:
        print(f"FFmpeg bin: {ffmpeg_dir}")

    beta_path = resolve_repo_path(config["model"]["beta_checkpoint_path"])
    required = ["config.json", "model.safetensors", "vocab.json", "processor_config.json"]
    missing = [name for name in required if not (beta_path / name).exists()]
    if missing:
        errors.append(
            f"Beta checkpoint is incomplete at {beta_path}; missing {missing}. "
            "Set BETA_CHECKPOINT_PATH explicitly."
        )
    else:
        print(f"Beta checkpoint: {beta_path}")

    files = gigaspeech_s_parquet_files(config)
    if not files:
        errors.append("No local GigaSpeech S parquet files were found.")
        full_count = 0
    else:
        print(f"GigaSpeech S parquet files: {len(files)}")
        print(f"GigaSpeech S local bytes: {sum(path.stat().st_size for path in files)}")
        from datasets import load_dataset

        metadata_dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in files]},
            split="train",
            cache_dir=str(resolve_repo_path(config["data"]["gigaspeech"]["cache_dir"])),
        )
        full_count = len(metadata_dataset)
        print(f"GigaSpeech S full train rows: {full_count}")

    limit = configured_row_limit(config["data"]["gigaspeech"].get("max_train_samples"))
    if limit is not None:
        errors.append(
            f"Gamma is configured with a GigaSpeech S row cap of {limit}. "
            "Set GIGASPEECH_S_MAX_ROWS=0 to use the full split."
        )
    else:
        print("GigaSpeech S row limit: ALL (0 means full split)")

    letters_cfg = config["data"]["readirect_letters"]
    train_letters = read_jsonl(letters_cfg["train_manifest"])
    valid_letters = read_jsonl(letters_cfg["validation_manifest"])
    test_letters = read_jsonl(letters_cfg["test_manifest"])
    if not train_letters:
        errors.append("ReaDirect letter train manifest is missing or empty.")
    print(f"ReaDirect letters: train={len(train_letters)}, validation={len(valid_letters)}, test={len(test_letters)}")
    source_ids = [
        str(row.get("source_id") or row.get("original_audio_path") or row.get("audio_path"))
        for row in train_letters
    ]
    counts = Counter(source_ids)
    repeat_factor = int(letters_cfg.get("repeat_factor", 1))
    if repeat_factor != 1:
        errors.append(
            f"Gamma letter repeat factor is {repeat_factor}; default must remain 1 unless explicitly reviewed."
        )
    if len(counts) != len(source_ids):
        errors.append("ReaDirect letter source IDs are not unique.")
    else:
        print(f"ReaDirect letter rows are unique: {len(source_ids)}")

    if "speechocean" in config["data"]:
        errors.append("SpeechOcean training configuration is present in Gamma data.")
    else:
        print("SpeechOcean training rows: 0")

    try:
        print(f"TorchCodec package version: {importlib.metadata.version('torchcodec')}")
        import torchcodec

        print(f"TorchCodec import OK: {torchcodec.__version__}")
    except Exception as exc:
        errors.append(f"TorchCodec cannot load: {exc}")

    if not errors:
        from scripts.train_wav2vec2_gamma import verify_beta_processor_and_head

        verify_beta_processor_and_head(
            beta_path,
            resolve_repo_path(config["model"]["reference_base_model_path"]),
        )
        shared = build_gamma_shared_dataset(config, "validation")
        print(f"Shared validation rows: {len(shared)}")
        print(
            f"Expected Gamma training rows before duration filtering: "
            f"{full_count + len(train_letters) * repeat_factor}"
        )
        if args.decode_sample:
            sample = load_gamma_gigaspeech_train(config, max_samples=1).select([0])[0]["audio"]
            print(
                f"Decoded GigaSpeech S sample: samples={len(sample['array'])}, "
                f"sample_rate={sample['sampling_rate']}"
            )
        else:
            print("WARNING: audio decoding not tested; rerun with --decode-sample.")

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("Gamma setup validation failed. Training and evaluation were not started.")
        return 1
    print("Gamma setup validation passed. Training and evaluation were not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
