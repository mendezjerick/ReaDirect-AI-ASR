from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_beta_data import build_beta_shared_dataset, load_beta_config
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Beta with raw greedy CTC decoding.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_beta.yaml"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test", "hard-cases"), default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_hard_cases(config: dict[str, Any]):
    from datasets import Audio, Dataset

    manifest = config.get("evaluation", {}).get("hard_case_manifest")
    if not manifest:
        raise RuntimeError("No real hard-case manifest is configured for Beta.")
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
    return Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def metric_block(references: list[str], predictions: list[str]) -> dict[str, float]:
    import jiwer

    return {
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
        "exact_match": sum(a == b for a, b in zip(references, predictions)) / len(references),
    }


def shared_metrics_from_artifact(path: str | Path) -> dict[str, Any] | None:
    artifact = resolve_repo_path(path)
    if not artifact.exists():
        return None
    data = json.loads(artifact.read_text(encoding="utf-8"))
    grouped: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    shared_references: list[str] = []
    shared_predictions: list[str] = []
    for row in data.get("predictions", []):
        dataset = row.get("dataset")
        if dataset not in {"speechocean", "readirect_letters"}:
            continue
        if "reference" not in row or "prediction" not in row:
            continue
        reference = str(row["reference"])
        prediction = str(row["prediction"])
        refs, preds = grouped[dataset]
        refs.append(reference)
        preds.append(prediction)
        shared_references.append(reference)
        shared_predictions.append(prediction)
    if not shared_references:
        return None
    speech = grouped.get("speechocean", ([], []))
    letters = grouped.get("readirect_letters", ([], []))
    shared = metric_block(shared_references, shared_predictions)
    speech_metrics = metric_block(*speech) if speech[0] else {}
    letter_metrics = metric_block(*letters) if letters[0] else {}
    return {
        "shared_wer": shared["wer"],
        "shared_cer": shared["cer"],
        "speechocean_wer": speech_metrics.get("wer"),
        "speechocean_cer": speech_metrics.get("cer"),
        "readirect_letter_accuracy": letter_metrics.get("exact_match"),
    }


def main() -> int:
    args = parse_args()
    configure_windows_ffmpeg()
    config = load_beta_config(args.config)
    model_path = resolve_repo_path(args.model or config["model"]["output_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Beta model not found: {model_path}")

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dataset = load_hard_cases(config) if args.split == "hard-cases" else build_beta_shared_dataset(config, args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    vocab = set(processor.tokenizer.get_vocab())

    references: list[str] = []
    predictions: list[str] = []
    grouped: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    records = []
    for row in dataset:
        reference = normalize_asr_text(row["text"], vocab)
        audio = row["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        prediction = normalize_asr_text(
            processor.batch_decode(torch.argmax(logits, dim=-1))[0], vocab
        )
        references.append(reference)
        predictions.append(prediction)
        group_refs, group_predictions = grouped[row["dataset"]]
        group_refs.append(reference)
        group_predictions.append(prediction)
        records.append(
            {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference == prediction,
            }
        )

    per_dataset = {
        name: {"rows": len(refs), **metric_block(refs, preds)}
        for name, (refs, preds) in sorted(grouped.items())
    }
    letter_metrics = per_dataset.get("readirect_letters", {})
    metrics = {
        "shared_wer": metric_block(references, predictions)["wer"],
        "shared_cer": metric_block(references, predictions)["cer"],
        "speechocean_wer": per_dataset.get("speechocean", {}).get("wer"),
        "speechocean_cer": per_dataset.get("speechocean", {}).get("cer"),
        "readirect_letter_accuracy": letter_metrics.get("exact_match"),
    }
    comparison = {
        name: result
        for name, artifact in config.get("evaluation", {}).get("comparison_artifacts", {}).items()
        if (result := shared_metrics_from_artifact(artifact)) is not None
    }
    comparison["beta"] = metrics
    output = resolve_repo_path(
        args.output or Path(config["model"]["report_dir"]) / f"beta_shared_{args.split}_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": "Beta",
        "model": str(model_path),
        "split": args.split,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "greedy_ctc_decode": True,
        "beam_search": False,
        "evaluated_rows": len(references),
        "metrics": metrics,
        "per_dataset": per_dataset,
        "comparison": comparison,
        "comparison_artifacts": config.get("evaluation", {}).get("comparison_artifacts", {}),
        "predictions": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("model", "split", "evaluated_rows", "metrics", "per_dataset", "comparison")
            },
            indent=2,
        )
    )
    print(f"Saved Beta evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
