from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_epsilon_data import (
    build_epsilon_slr83_heldout,
    load_epsilon_config,
    split_slr83_rows,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Epsilon paths and dataset parsing.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_epsilon.yaml"))
    parser.add_argument("--decode-sample", action="store_true")
    args = parser.parse_args()
    configure_windows_ffmpeg()
    config = load_epsilon_config(args.config)

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    delta_path = resolve_repo_path(config["model"]["delta_checkpoint_path"])
    required = ("config.json", "model.safetensors", "vocab.json", "processor_config.json")
    missing = [name for name in required if not (delta_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Delta model is incomplete at {delta_path}: {missing}")
    processor = Wav2Vec2Processor.from_pretrained(str(delta_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(delta_path), local_files_only=True)
    vocab = set(processor.tokenizer.get_vocab())
    train_rows, eval_rows, split = split_slr83_rows(config, write_manifest=True)
    empty = [
        row["source_id"]
        for row in train_rows + eval_rows
        if not normalize_asr_text(row["text"], vocab)
    ]
    if empty:
        raise RuntimeError(f"{len(empty)} SLR83 transcripts normalize to empty.")
    summary = {
        "delta_checkpoint": str(delta_path),
        "delta_vocab_size": len(vocab),
        "delta_ctc_head_shape": list(model.lm_head.weight.shape),
        "sample_rate": processor.feature_extractor.sampling_rate,
        "slr83_split": {key: value for key, value in split.items() if key != "rows"},
        "empty_normalized_slr83_transcripts": 0,
        "librispeech_training_rows": 0,
    }
    if args.decode_sample:
        heldout = build_epsilon_slr83_heldout(config)
        audio = heldout[0]["audio"]
        summary["decoded_heldout_sample"] = {
            "samples": len(audio["array"]),
            "sample_rate": audio["sampling_rate"],
            "source_id": heldout[0]["source_id"],
        }
    print(json.dumps(summary, indent=2))
    print("Epsilon parsing validation passed. Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
