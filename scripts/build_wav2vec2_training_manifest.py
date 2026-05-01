from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_manifest_utils import deterministic_sample, read_jsonl, write_jsonl


DEFAULT_INPUTS = {
    "librispeech_train": "external_datasets/manifests/librispeech_train_clean_100.jsonl",
    "librispeech_valid": "external_datasets/manifests/librispeech_dev_clean.jsonl",
    "librispeech_test": "external_datasets/manifests/librispeech_test_clean.jsonl",
    "speechocean_train": "external_datasets/manifests/speechocean_train.jsonl",
    "speechocean_valid": "external_datasets/manifests/speechocean_valid.jsonl",
    "speechocean_test": "external_datasets/manifests/speechocean_test.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed ReaDirect Wav2Vec2 training manifests.")
    parser.add_argument("--librispeech-weight", type=float, default=0.30)
    parser.add_argument("--speechocean-weight", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-librispeech-train", type=int, default=None)
    parser.add_argument("--max-speechocean-train", type=int, default=None)
    return parser.parse_args()


def weighted_train_rows(
    librispeech_rows: list[dict],
    speechocean_rows: list[dict],
    librispeech_weight: float,
    speechocean_weight: float,
    seed: int,
    max_librispeech: int | None,
    max_speechocean: int | None,
) -> list[dict]:
    if max_librispeech is not None:
        librispeech_rows = deterministic_sample(librispeech_rows, max_librispeech, seed)
    if max_speechocean is not None:
        speechocean_rows = deterministic_sample(speechocean_rows, max_speechocean, seed)
    if not librispeech_rows:
        return list(speechocean_rows)
    if not speechocean_rows:
        return list(librispeech_rows)

    total_weight = max(0.000001, librispeech_weight + speechocean_weight)
    target_lib_fraction = librispeech_weight / total_weight
    target_speech_fraction = speechocean_weight / total_weight

    lib_limit_from_speech = int(round(len(speechocean_rows) * (target_lib_fraction / target_speech_fraction)))
    speech_limit_from_lib = int(round(len(librispeech_rows) * (target_speech_fraction / target_lib_fraction)))

    lib_count = min(len(librispeech_rows), max(1, lib_limit_from_speech))
    speech_count = min(len(speechocean_rows), max(1, speech_limit_from_lib))
    sampled = deterministic_sample(librispeech_rows, lib_count, seed) + deterministic_sample(speechocean_rows, speech_count, seed + 1)
    return sorted(sampled, key=lambda row: (str(row.get("dataset", "")), str(row.get("source_id", ""))))


def main() -> int:
    args = parse_args()
    libri_train = read_jsonl(DEFAULT_INPUTS["librispeech_train"])
    speech_train = read_jsonl(DEFAULT_INPUTS["speechocean_train"])
    train_rows = weighted_train_rows(
        libri_train,
        speech_train,
        args.librispeech_weight,
        args.speechocean_weight,
        args.seed,
        args.max_librispeech_train,
        args.max_speechocean_train,
    )
    train_count = write_jsonl("external_datasets/manifests/readirect_train_mixed.jsonl", train_rows)

    valid_rows = read_jsonl(DEFAULT_INPUTS["librispeech_valid"]) + read_jsonl(DEFAULT_INPUTS["speechocean_valid"])
    test_rows = read_jsonl(DEFAULT_INPUTS["librispeech_test"]) + read_jsonl(DEFAULT_INPUTS["speechocean_test"])
    valid_count = write_jsonl("external_datasets/manifests/readirect_valid_mixed.jsonl", valid_rows)
    test_count = write_jsonl("external_datasets/manifests/readirect_test_mixed.jsonl", test_rows)

    print(f"Wrote mixed train rows: {train_count}")
    print(f"Wrote mixed valid rows: {valid_count}")
    print(f"Wrote mixed test rows: {test_count}")
    print(f"Train source counts: librispeech={sum(1 for row in train_rows if row.get('dataset') == 'librispeech')}, speechocean={sum(1 for row in train_rows if row.get('dataset') == 'speechocean')}")
    if train_count == 0:
        print("No train rows were written. Prepare at least one train manifest first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

