from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_wav2vec2_beta import metric_block, shared_metrics_from_artifact
from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_delta_data import build_delta_shared_dataset, load_delta_config
from training.wav2vec2_manifest_utils import resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Delta with raw greedy CTC decoding.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_delta.yaml"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    configure_windows_ffmpeg()
    config = load_delta_config(args.config)
    model_path = resolve_repo_path(args.model or config["model"]["output_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Delta model not found: {model_path}")

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dataset = build_delta_shared_dataset(
        config,
        "validation",
        include_gigaspeech=bool(config["evaluation"].get("include_gigaspeech_validation", True)),
    )
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    vocab = set(processor.tokenizer.get_vocab())

    grouped = defaultdict(lambda: ([], []))
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
        refs, predictions = grouped[row["dataset"]]
        refs.append(reference)
        predictions.append(prediction)
        records.append(
            {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference == prediction,
            }
        )

    per_source = {
        name: {"rows": len(refs), **metric_block(refs, predictions)}
        for name, (refs, predictions) in sorted(grouped.items())
    }
    shared_refs = []
    shared_predictions = []
    all_refs = []
    all_predictions = []
    for name, (refs, predictions) in grouped.items():
        all_refs.extend(refs)
        all_predictions.extend(predictions)
        if name in {"speechocean", "readirect_letters"}:
            shared_refs.extend(refs)
            shared_predictions.extend(predictions)
    shared = metric_block(shared_refs, shared_predictions)
    overall = metric_block(all_refs, all_predictions)
    metrics = {
        "shared_wer": shared["wer"],
        "shared_cer": shared["cer"],
        "overall_wer": overall["wer"],
        "overall_cer": overall["cer"],
        "speechocean_wer": per_source.get("speechocean", {}).get("wer"),
        "speechocean_cer": per_source.get("speechocean", {}).get("cer"),
        "readirect_letter_accuracy": per_source.get("readirect_letters", {}).get("exact_match"),
        "gigaspeech_wer": per_source.get("gigaspeech", {}).get("wer"),
        "gigaspeech_cer": per_source.get("gigaspeech", {}).get("cer"),
    }
    comparison = {
        name: result
        for name, artifact in config["evaluation"].get("comparison_artifacts", {}).items()
        if (result := shared_metrics_from_artifact(artifact)) is not None
    }
    comparison["delta"] = {
        key: metrics[key]
        for key in (
            "shared_wer",
            "shared_cer",
            "speechocean_wer",
            "speechocean_cer",
            "readirect_letter_accuracy",
        )
    }
    comparison_deltas = {}
    for baseline in ("beta", "gamma"):
        if baseline not in comparison:
            continue
        baseline_metrics = comparison[baseline]
        comparison_deltas[f"delta_vs_{baseline}"] = {
            "shared_wer_percentage_points": 100
            * (metrics["shared_wer"] - baseline_metrics["shared_wer"]),
            "shared_cer_percentage_points": 100
            * (metrics["shared_cer"] - baseline_metrics["shared_cer"]),
            "letter_accuracy_percentage_points": 100
            * (
                metrics["readirect_letter_accuracy"]
                - baseline_metrics["readirect_letter_accuracy"]
            ),
        }
    output = resolve_repo_path(
        Path(config["model"]["report_dir"]) / "delta_shared_validation_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": "Delta",
        "model": str(model_path),
        "split": "validation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "greedy_ctc_decode": True,
        "beam_search": False,
        "evaluated_rows": len(records),
        "shared_comparison_rows": len(shared_refs),
        "metrics": metrics,
        "per_source": per_source,
        "comparison": comparison,
        "comparison_deltas": comparison_deltas,
        "gigaspeech_validation_available": "gigaspeech" in per_source,
        "predictions": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("metrics", "per_source", "comparison", "comparison_deltas")
            },
            indent=2,
        )
    )
    print(f"Saved Delta evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
