from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import (
    build_alpha_raw_dataset,
    configure_windows_ffmpeg,
    load_alpha_config,
)
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw greedy-decoded Wav2Vec2 Alpha ASR.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_alpha.yaml"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test", "hard-cases"), default=None)
    parser.add_argument(
        "--include-dataset",
        action="append",
        choices=("gigaspeech", "speechocean", "readirect_letters"),
        default=None,
        help="Restrict evaluation to one or more datasets. Repeat the option to include multiple datasets.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_hard_cases(config: dict[str, Any]):
    from datasets import Audio, Dataset

    manifest = config.get("evaluation", {}).get("hard_case_manifest")
    if not manifest:
        raise RuntimeError("No real hard-case manifest is configured for Alpha.")
    rows = []
    for row in read_jsonl(manifest):
        path = resolve_repo_path(row.get("audio_path", ""))
        if path.exists() and row.get("text"):
            rows.append(
                {
                    "audio": str(path),
                    "text": row["text"],
                    "dataset": str(row.get("dataset", "hard_cases")),
                    "split": "hard-cases",
                    "source_id": str(row.get("source_id", path.name)),
                    "speaker_id": str(row.get("speaker_id", "")),
                    "duration_seconds": row.get("duration_seconds"),
                }
            )
    return Dataset.from_list(rows).cast_column(
        "audio", Audio(sampling_rate=int(config["data"]["sample_rate"]), num_channels=1)
    )


def main() -> int:
    args = parse_args()
    configure_windows_ffmpeg()
    config = load_alpha_config(args.config)
    split = args.split or str(config["evaluation"].get("default_split", "validation"))
    model_path = resolve_repo_path(args.model or config["model"]["output_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Alpha model not found: {model_path}")

    import jiwer
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dataset = load_hard_cases(config) if split == "hard-cases" else build_alpha_raw_dataset(config, split)
    if args.include_dataset:
        included = set(args.include_dataset)
        dataset = dataset.filter(
            lambda name: name in included,
            input_columns=["dataset"],
            desc=f"Selecting datasets: {', '.join(sorted(included))}",
        )
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    vocab = set(processor.tokenizer.get_vocab().keys())

    references: list[str] = []
    predictions: list[str] = []
    records: list[dict[str, Any]] = []
    letter_correct = 0
    letter_total = 0
    for row in dataset:
        reference = normalize_asr_text(row["text"], vocab)
        audio = row["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = normalize_asr_text(processor.batch_decode(predicted_ids)[0], vocab)
        references.append(reference)
        predictions.append(prediction)
        is_letter = row["dataset"] == "readirect_letters" and len(reference) == 1
        if is_letter:
            letter_total += 1
            letter_correct += int(prediction == reference)
        records.append(
            {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "reference": reference,
                "prediction": prediction,
                "exact_match": prediction == reference,
            }
        )

    metrics = {
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
        "readirect_letter_accuracy": (letter_correct / letter_total) if letter_total else None,
        "readirect_letter_correct": letter_correct,
        "readirect_letter_total": letter_total,
    }
    output = resolve_repo_path(
        args.output or Path(config["model"]["report_dir"]) / f"alpha_{split}_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": "Alpha",
        "model": str(model_path),
        "split": split,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "greedy_ctc_decode": True,
        "beam_search": False,
        "included_datasets": args.include_dataset,
        "evaluated_rows": len(references),
        "metrics": metrics,
        "predictions": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model", "split", "device", "evaluated_rows", "metrics")}, indent=2))
    print(f"Saved Alpha evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
