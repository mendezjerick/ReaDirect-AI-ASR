from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_wav2vec2_beta import metric_block
from training.ctc_decoding import CTCTextDecoder, DecodeSettings
from training.text_normalization import normalize_asr_text
from training.wav2vec2_alpha_data import configure_windows_ffmpeg
from training.wav2vec2_epsilon_data import (
    build_epsilon_slr83_heldout,
    load_epsilon_config,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Epsilon on held-out SLR83.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_epsilon.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/asr/epsilon"))
    parser.add_argument(
        "--decode_mode", choices=("greedy", "beam", "beam_lm"), default="beam_lm"
    )
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--lm_path", type=Path, default=None)
    parser.add_argument("--hotwords", nargs="*", default=[])
    parser.add_argument("--hotword_weight", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    configure_windows_ffmpeg()
    config = load_epsilon_config(args.config)
    model_path = resolve_repo_path(args.model)
    lm_path = args.lm_path or (
        Path(config["evaluation"]["lm_path"]) if args.decode_mode == "beam_lm" else None
    )
    beam_width = args.beam_width or (100 if args.decode_mode == "beam_lm" else 50)

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dataset = build_epsilon_slr83_heldout(config)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    vocab = set(processor.tokenizer.get_vocab())
    hotwords = tuple(
        value for word in args.hotwords
        if (value := normalize_asr_text(word, vocab))
    )
    decoder = CTCTextDecoder(
        processor,
        DecodeSettings(
            decode_mode=args.decode_mode,
            beam_width=beam_width,
            alpha=args.alpha,
            beta=args.beta,
            lm_path=str(resolve_repo_path(lm_path)) if lm_path else None,
            hotwords=hotwords,
            hotword_weight=args.hotword_weight,
        ),
    )
    if args.decode_mode == "beam_lm" and not decoder.language_model_used:
        raise RuntimeError("Held-out SLR83 LM evaluation requested but KenLM was not used.")
    print(
        f"Held-out SLR83 decoder: {decoder.backend}; "
        f"beam={decoder.beam_search_used}; LM={decoder.language_model_used}"
    )

    references = []
    predictions = []
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
        started = time.perf_counter()
        prediction = normalize_asr_text(decoder.decode(logits), vocab)
        row_seconds = time.perf_counter() - started
        decode_seconds += row_seconds
        references.append(reference)
        predictions.append(prediction)
        records.append(
            {
                "dataset": "slr83_southern_english_heldout",
                "source_id": row["source_id"],
                "speaker_id": row["speaker_id"],
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference == prediction,
                "decode_runtime_seconds": row_seconds,
            }
        )
        if index % 250 == 0:
            print(f"Decoded {index}/{len(dataset)} held-out SLR83 rows")

    metrics = metric_block(references, predictions)
    output = resolve_repo_path(
        Path(config["model"]["report_dir"]) / "epsilon_slr83_heldout_evaluation.json"
    )
    result = {
        "experiment": "epsilon_slr83_heldout",
        "model_name": "epsilon",
        "model": str(model_path),
        "split": "slr83_heldout_evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "evaluated_rows": len(records),
        **decoder.metadata(),
        "metrics": {
            "heldout_slr83_wer": metrics["wer"],
            "heldout_slr83_cer": metrics["cer"],
            "heldout_slr83_exact_match": metrics["exact_match"],
        },
        "per_source": {
            "slr83_southern_english_heldout": {
                "rows": len(records),
                **metrics,
            }
        },
        "runtime": {
            "decode_seconds": decode_seconds,
            "decode_seconds_per_sample": decode_seconds / len(records),
        },
        "prediction_examples": records[:25],
        "predictions": records,
        "heldout_used_for_training_or_early_stopping": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved held-out SLR83 evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
