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
from training.wav2vec2_beta_data import build_beta_shared_dataset, load_beta_config
from training.wav2vec2_manifest_utils import resolve_repo_path


MODEL_PATHS = {
    "beta": "models/asr/beta",
    "delta": "models/asr/delta",
    "epsilon": "models/asr/epsilon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate greedy, no-LM beam, or KenLM CTC decoding on shared validation."
    )
    parser.add_argument("--model-name", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_beta.yaml"))
    parser.add_argument(
        "--decode_mode",
        choices=("greedy", "beam", "beam_lm"),
        default="greedy",
    )
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lm_path", type=Path, default=None)
    parser.add_argument("--hotwords", nargs="*", default=[])
    parser.add_argument("--hotword_weight", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    beam_width = args.beam_width
    if beam_width is None:
        beam_width = 100 if args.decode_mode == "beam_lm" else 50
    configure_windows_ffmpeg()
    config = load_beta_config(args.config)
    model_path = resolve_repo_path(args.model or MODEL_PATHS[args.model_name])
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Complete model not found: {model_path}")

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dataset = build_beta_shared_dataset(config, "validation")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    vocab = set(processor.tokenizer.get_vocab())
    hotwords = tuple(
        normalized
        for word in args.hotwords
        if (normalized := normalize_asr_text(word, vocab))
    )
    settings = DecodeSettings(
        decode_mode=args.decode_mode,
        beam_width=beam_width,
        alpha=args.alpha,
        beta=args.beta,
        lm_path=str(args.lm_path.resolve()) if args.lm_path else None,
        hotwords=hotwords,
        hotword_weight=args.hotword_weight,
    )
    decoder = CTCTextDecoder(processor, settings)
    if args.decode_mode in {"beam", "beam_lm"} and not decoder.beam_search_used:
        raise RuntimeError("Beam mode requested but a beam decoder was not initialized.")
    if args.decode_mode == "beam_lm" and not decoder.language_model_used:
        raise RuntimeError("LM beam mode requested but KenLM was not actually loaded.")
    print(
        f"Decoder mode: {args.decode_mode}; backend: {decoder.backend}; "
        f"beam search actually used: {decoder.beam_search_used}"
    )
    if args.decode_mode == "beam_lm":
        print("LANGUAGE MODEL BEAM SEARCH ACTIVE: True")
        print(f"Loaded KenLM language model: {settings.lm_path}")

    grouped = defaultdict(lambda: ([], []))
    records = []
    inference_seconds = 0.0
    decode_seconds = 0.0
    evaluation_started = time.perf_counter()
    for index, row in enumerate(dataset, start=1):
        reference = normalize_asr_text(row["text"], vocab)
        audio = row["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )
        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
        inference_seconds += time.perf_counter() - inference_started
        decode_started = time.perf_counter()
        decoded_text = decoder.decode(logits)
        row_decode_seconds = time.perf_counter() - decode_started
        decode_seconds += row_decode_seconds
        prediction = normalize_asr_text(decoded_text, vocab)
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
                "decode_runtime_seconds": row_decode_seconds,
            }
        )
        if index % 100 == 0:
            print(f"Decoded {index}/{len(dataset)} rows with {decoder.backend}")

    per_source = {
        name: {"rows": len(refs), **metric_block(refs, predictions)}
        for name, (refs, predictions) in sorted(grouped.items())
    }
    references = []
    predictions = []
    for refs, source_predictions in grouped.values():
        references.extend(refs)
        predictions.extend(source_predictions)
    shared = metric_block(references, predictions)
    metrics = {
        "shared_wer": shared["wer"],
        "shared_cer": shared["cer"],
        "shared_exact_match": shared["exact_match"],
        "speechocean_wer": per_source["speechocean"]["wer"],
        "speechocean_cer": per_source["speechocean"]["cer"],
        "speechocean_exact_match": per_source["speechocean"]["exact_match"],
        "readirect_letter_accuracy": per_source["readirect_letters"]["exact_match"],
    }
    output_suffix = {
        "greedy": "greedy",
        "beam": "beam_no_lm",
        "beam_lm": "beam_lm",
    }[args.decode_mode]
    default_output = (
        Path("reports/asr/epsilon") / f"epsilon_{output_suffix}_evaluation.json"
        if args.model_name == "epsilon"
        else Path("reports/asr/decoder_lm_comparison")
        / f"{args.model_name}_{output_suffix}_evaluation.json"
    )
    output = resolve_repo_path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "decoder_lm_comparison",
        "model_name": args.model_name,
        "model": str(model_path),
        "split": "shared_validation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "evaluated_rows": len(records),
        **decoder.metadata(),
        "greedy_ctc_decode": args.decode_mode == "greedy",
        "runtime": {
            "total_seconds": time.perf_counter() - evaluation_started,
            "model_inference_seconds": inference_seconds,
            "model_inference_seconds_per_sample": inference_seconds / len(records),
            "decode_seconds": decode_seconds,
            "decode_seconds_per_sample": decode_seconds / len(records),
        },
        "metrics": metrics,
        "per_source": per_source,
        "prediction_examples": records[:25],
        "predictions": records,
    }
    if args.decode_mode in {"beam", "beam_lm"} and result["beam_search"] is not True:
        raise RuntimeError("Refusing to save beam output because beam search was not used.")
    if args.decode_mode == "beam_lm" and result["language_model_used"] is not True:
        raise RuntimeError("Refusing to save LM output because KenLM was not used.")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decode": decoder.metadata(), "metrics": metrics, "per_source": per_source}, indent=2))
    print(f"Saved decoder evaluation to {output}")
    print(f"BEAM SEARCH ACTUALLY USED: {decoder.beam_search_used}")
    print(f"LANGUAGE MODEL USED: {decoder.language_model_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
