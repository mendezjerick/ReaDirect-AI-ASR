from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_wav2vec2_beta import metric_block
from training.ctc_decoding import CTCTextDecoder, DecodeSettings
from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_manifest_utils import resolve_repo_path
from training.wav2vec2_shared_benchmark import (
    SOURCE_ORDER,
    build_shared_benchmark,
    load_benchmark_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one model on the fixed five-source benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wav2vec2_shared_benchmark.yaml"),
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=("base", "v1", "v2", "alpha", "beta", "gamma", "delta", "epsilon"),
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--decode_mode",
        choices=("greedy", "beam", "beam_lm"),
        default="beam_lm",
    )
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument(
        "--lm_path",
        type=Path,
        default=Path("external_datasets/language_models/3-gram.pruned.1e-7.arpa"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_windows_ffmpeg()
    config = load_benchmark_config(args.config)
    model_path = resolve_repo_path(args.model or config["models"][args.model_name])
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Complete model not found: {model_path}")

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dataset, benchmark_summary = build_shared_benchmark(config)
    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    vocab = set(processor.tokenizer.get_vocab())
    beam_width = args.beam_width or (100 if args.decode_mode == "beam_lm" else 50)
    lm_path = (
        str(resolve_repo_path(args.lm_path))
        if args.decode_mode == "beam_lm"
        else None
    )
    decoder = CTCTextDecoder(
        processor,
        DecodeSettings(
            decode_mode=args.decode_mode,
            beam_width=beam_width,
            alpha=args.alpha,
            beta=args.beta,
            lm_path=lm_path,
        ),
    )
    if args.decode_mode == "beam_lm" and not decoder.language_model_used:
        raise RuntimeError("Shared benchmark requested LM beam, but KenLM was not used.")
    print(
        f"Shared benchmark decoder: {decoder.backend}; "
        f"beam={decoder.beam_search_used}; LM={decoder.language_model_used}"
    )

    grouped = defaultdict(lambda: ([], []))
    records = []
    decode_seconds = 0.0
    for index, row in enumerate(dataset, start=1):
        reference = normalize_asr_text(row["text"], vocab)
        audio = row["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
        decode_started = time.perf_counter()
        prediction = normalize_asr_text(decoder.decode(logits), vocab)
        row_decode_seconds = time.perf_counter() - decode_started
        decode_seconds += row_decode_seconds
        references, predictions = grouped[row["dataset"]]
        references.append(reference)
        predictions.append(prediction)
        records.append(
            {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference == prediction,
                "decode_runtime_seconds": row_decode_seconds,
            }
        )
        if index % 100 == 0:
            print(f"Evaluated {index}/{len(dataset)} rows")

    per_source = {
        name: {"rows": len(grouped[name][0]), **metric_block(*grouped[name])}
        for name in SOURCE_ORDER
    }
    macro_wer = sum(per_source[name]["wer"] for name in SOURCE_ORDER) / len(SOURCE_ORDER)
    macro_cer = sum(per_source[name]["cer"] for name in SOURCE_ORDER) / len(SOURCE_ORDER)
    clean_sources = [
        name for name in SOURCE_ORDER if config["sources"][name]["leakage"] == "clean"
    ]
    clean_macro_wer = sum(per_source[name]["wer"] for name in clean_sources) / len(clean_sources)
    clean_macro_cer = sum(per_source[name]["cer"] for name in clean_sources) / len(clean_sources)
    metrics = {
        "macro_wer": macro_wer,
        "macro_cer": macro_cer,
        "clean_macro_wer": clean_macro_wer,
        "clean_macro_cer": clean_macro_cer,
        "readirect_letter_accuracy": per_source["readirect_letters"]["exact_match"],
    }
    contamination = {
        name: (name == "slr83" and args.model_name == "delta")
        for name in SOURCE_ORDER
    }
    output_dir_key = (
        "beam_lm_output_dir" if args.decode_mode == "beam_lm" else "output_dir"
    )
    suffix = "shared_benchmark_beam_lm" if args.decode_mode == "beam_lm" else "shared_benchmark"
    output = resolve_repo_path(
        args.output
        or Path(config["benchmark"][output_dir_key]) / f"{args.model_name}_{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark_name": config["run"]["name"],
        "benchmark_manifest": str(resolve_repo_path(config["benchmark"]["manifest_summary"])),
        "model_name": args.model_name,
        "model": str(model_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "evaluated_rows": len(records),
        "rows_per_source": int(config["benchmark"]["rows_per_source"]),
        **decoder.metadata(),
        "runtime": {
            "decode_seconds": decode_seconds,
            "decode_seconds_per_sample": decode_seconds / len(records),
        },
        "metrics": metrics,
        "per_source": per_source,
        "contamination": contamination,
        "clean_macro_sources": clean_sources,
        "fairness_note": (
            "SLR83 uses Epsilon's speaker-held-out split and is clean for Epsilon. "
            "It remains contaminated only for Delta, which trained on all SLR83 rows."
        ),
        "predictions": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "evaluated_rows": len(records),
                "metrics": metrics,
                "per_source": per_source,
                "contamination": contamination,
            },
            indent=2,
        )
    )
    print(f"Saved shared benchmark evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
