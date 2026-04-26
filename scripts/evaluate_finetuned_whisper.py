from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.finetuning.whisper_metrics import compute_cer_metric, compute_exact_match_rate, compute_wer_metric
from readirect_asr.finetuning.whisper_audio import load_audio_array
from readirect_asr.finetuning.whisper_generation_config import prepare_whisper_generation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Hugging Face Whisper model.")
    parser.add_argument("--model-dir", default="model_artifacts/readirect-whisper-base-en-v1-hf", type=Path)
    parser.add_argument("--test-jsonl", default="data/processed/whisper_finetune/test.jsonl", type=Path)
    parser.add_argument("--output", default="reports/finetuned_whisper_eval.md", type=Path)
    parser.add_argument("--metrics-json", default="reports/finetuned_whisper_metrics.json", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}")
        return 2
    if not args.test_jsonl.exists():
        print(f"Test JSONL not found: {args.test_jsonl}")
        return 2
    try:
        import torch
        from datasets import load_dataset
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception as exc:
        print(f"Missing evaluation dependency: {exc}")
        return 2

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    prepare_whisper_generation_config(model, processor, language="en", task="transcribe")
    dataset = load_dataset("json", data_files=str(args.test_jsonl), split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    predictions: list[str] = []
    references: list[str] = []
    examples = []
    for row in dataset:
        audio_array, sampling_rate = load_audio_array(row["audio"], sampling_rate=16000, backend="librosa")
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        try:
            with torch.no_grad():
                pred_ids = model.generate(inputs)
        except Exception as exc:
            print("Whisper generation failed during evaluation.")
            print("This is usually a generation_config compatibility issue. Training output may still be valid.")
            print(f"Error: {exc}")
            return 3
        prediction = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        reference = str(row.get("sentence", ""))
        predictions.append(prediction)
        references.append(reference)
        if len(examples) < 20:
            examples.append({"reference": reference, "prediction": prediction})
    metrics = {
        "wer": compute_wer_metric(predictions, references),
        "cer": compute_cer_metric(predictions, references),
        "exact_match_rate": compute_exact_match_rate(predictions, references),
        "rows": len(references),
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = ["# Fine-Tuned Whisper Evaluation", "", *(f"- {key}: {value}" for key, value in metrics.items()), "", "## Sample Predictions", ""]
    for item in examples:
        report.append(f"- Reference: `{item['reference']}` | Prediction: `{item['prediction']}`")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Metrics: {metrics}")
    print(f"Report path: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
