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
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually evaluate a Wav2Vec2 ASR model on a JSONL manifest.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/wav2vec2_readirect_full_eval.json"))
    parser.add_argument("--allow-base-fallback", action="store_true")
    parser.add_argument("--base-model", type=Path, default=Path("models/wav2vec2-base-960h"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    requested = resolve_repo_path(args.model)
    if requested.exists():
        return requested
    if args.allow_base_fallback:
        fallback = resolve_repo_path(args.base_model)
        if fallback.exists():
            print(f"Warning: requested model missing; falling back to {fallback.relative_to(PROJECT_ROOT)}")
            return fallback
    raise FileNotFoundError(f"Model path not found: {requested}")


def load_audio(path: str | Path, sample_rate: int):
    import librosa

    audio, sr = librosa.load(str(resolve_repo_path(path)), sr=sample_rate, mono=True)
    return audio, int(sr)


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"wer": None, "cer": None, "metrics_available": False}
    try:
        import jiwer

        metrics["wer"] = float(jiwer.wer(references, predictions))
        metrics["cer"] = float(jiwer.cer(references, predictions))
        metrics["metrics_available"] = True
    except Exception as exc:
        metrics["metrics_error"] = f"Install jiwer/evaluate for WER and CER: {exc}"
    return metrics


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)
    rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {args.manifest}")

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    vocab = set(processor.tokenizer.get_vocab().keys())
    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        audio_path = str(row.get("audio_path", "")).strip()
        reference = normalize_asr_text(row.get("text", ""), vocab)
        if not audio_path or not reference or not resolve_repo_path(audio_path).exists():
            continue
        try:
            audio, sr = load_audio(audio_path, args.sample_rate)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            prediction = normalize_asr_text(processor.batch_decode(predicted_ids)[0], vocab)
            predictions.append(prediction)
            references.append(reference)
            records.append(
                {
                    "audio_path": audio_path,
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "reference": reference,
                    "prediction": prediction,
                    "source_id": row.get("source_id"),
                }
            )
        except Exception as exc:
            records.append({"audio_path": audio_path, "error": str(exc), "source_id": row.get("source_id")})

    metrics = compute_metrics(predictions, references)
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": str(model_path),
        "manifest": str(resolve_repo_path(args.manifest)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "total_manifest_rows": len(rows),
        "evaluated_rows": len(predictions),
        "metrics": metrics,
        "predictions": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model", "manifest", "device", "total_manifest_rows", "evaluated_rows", "metrics")}, indent=2))
    print(f"Saved evaluation output to {output.relative_to(PROJECT_ROOT)}")
    return 0 if predictions else 1


if __name__ == "__main__":
    raise SystemExit(main())

