from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_manifest_utils import resolve_repo_path
from training.wav2vec2_shared_benchmark import SOURCE_ORDER, load_benchmark_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completed five-source benchmark results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wav2vec2_shared_benchmark.yaml"),
    )
    parser.add_argument(
        "--decode-mode",
        choices=("greedy", "beam_lm"),
        default="beam_lm",
    )
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    output_dir = resolve_repo_path(
        config["benchmark"][
            "beam_lm_output_dir" if args.decode_mode == "beam_lm" else "output_dir"
        ]
    )
    results = {}
    for model_name in config["models"]:
        suffix = (
            "shared_benchmark_beam_lm"
            if args.decode_mode == "beam_lm"
            else "shared_benchmark"
        )
        path = output_dir / f"{model_name}_{suffix}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if args.decode_mode == "beam_lm":
                if result.get("language_model_used") is not True:
                    raise RuntimeError(f"KenLM was not used in {path}")
                if result.get("decoder_backend") != "pyctcdecode_with_lm":
                    raise RuntimeError(f"Unexpected LM decoder backend in {path}")
            results[model_name] = result
    if not results:
        raise FileNotFoundError(f"No completed shared benchmark results found under {output_dir}")

    comparison = {
        name: {
            "metrics": result["metrics"],
            "per_source": result["per_source"],
            "contamination": result["contamination"],
        }
        for name, result in results.items()
    }
    clean_ranking = sorted(
        (
            {
                "model": name,
                "clean_macro_wer": result["metrics"]["clean_macro_wer"],
                "clean_macro_cer": result["metrics"]["clean_macro_cer"],
                "letter_accuracy": result["metrics"]["readirect_letter_accuracy"],
            }
            for name, result in results.items()
        ),
        key=lambda row: row["clean_macro_wer"],
    )
    epsilon_heldout_path = (
        PROJECT_ROOT / "reports" / "asr" / "epsilon"
        / "epsilon_slr83_heldout_evaluation.json"
    )
    epsilon_heldout = None
    if epsilon_heldout_path.exists():
        heldout_result = json.loads(epsilon_heldout_path.read_text(encoding="utf-8"))
        if heldout_result.get("language_model_used") is not True:
            raise RuntimeError("Epsilon held-out SLR83 result did not use KenLM beam.")
        epsilon_heldout = {
            "artifact": str(epsilon_heldout_path),
            "evaluated_rows": heldout_result["evaluated_rows"],
            "decode_mode": heldout_result["decode_mode"],
            "decoder_backend": heldout_result["decoder_backend"],
            "metrics": heldout_result["metrics"],
            "clean_speaker_heldout": True,
        }
    report = {
        "benchmark_name": config["run"]["name"],
        "decode_mode": args.decode_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_models": list(results),
        "source_order": list(SOURCE_ORDER),
        "comparison": comparison,
        "clean_macro_wer_ranking": clean_ranking,
        "epsilon_clean_slr83_heldout": epsilon_heldout,
        "actual_rows_per_source": {
            name: results[next(iter(results))]["per_source"][name]["rows"]
            for name in SOURCE_ORDER
        },
        "fairness_warning": (
            "SLR83 uses Epsilon's speaker-held-out split and is uncontaminated for "
            "Epsilon. Delta remains contaminated because it trained on all SLR83 rows."
        ),
    }
    output = output_dir / (
        "shared_benchmark_beam_lm_comparison.json"
        if args.decode_mode == "beam_lm"
        else "shared_benchmark_comparison.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed_models": list(results), "ranking": clean_ranking}, indent=2))
    print(f"Saved benchmark comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
