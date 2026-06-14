from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "asr" / "decoder_lm_comparison"
OLD_REPORT_DIR = PROJECT_ROOT / "reports" / "asr" / "decoder_comparison"


def import_existing_baseline(
    report_dir: Path,
    model_name: str,
    mode: str,
) -> Path:
    suffix = "greedy" if mode == "greedy" else "beam_no_lm"
    target = report_dir / f"{model_name}_{suffix}_evaluation.json"
    if target.exists():
        return target

    old_suffix = "greedy" if mode == "greedy" else "beam"
    source = OLD_REPORT_DIR / f"{model_name}_{old_suffix}_evaluation.json"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {mode} baseline. Run it or restore the existing result: {source}"
        )
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("decode_mode") != mode:
        raise RuntimeError(f"Unexpected decode mode in baseline: {source}")
    if mode == "beam" and result.get("decoder_backend") == "transformers_greedy":
        raise RuntimeError(f"No-LM baseline did not use beam search: {source}")
    result.update(
        {
            "experiment": "decoder_lm_comparison",
            "language_model_used": False,
            "lm_path": None,
            "hotwords": result.get("hotwords", []),
            "hotword_weight": result.get("hotword_weight", 5.0),
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Imported existing {model_name} {mode} baseline to {target}")
    return target


def load_result(path: Path, expected_mode: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing decoder evaluation: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("decode_mode") != expected_mode:
        raise RuntimeError(f"Unexpected decode mode in {path}")
    if expected_mode in {"beam", "beam_lm"} and result.get("beam_search") is not True:
        raise RuntimeError(f"Beam search was not actually used in {path}")
    if expected_mode == "beam_lm" and result.get("language_model_used") is not True:
        raise RuntimeError(f"KenLM was not actually used in {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare greedy, no-LM beam, and KenLM beam for Beta and Delta."
    )
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in ("beta", "delta"):
        greedy_path = import_existing_baseline(report_dir, model_name, "greedy")
        beam_path = import_existing_baseline(report_dir, model_name, "beam")
        lm_path = report_dir / f"{model_name}_beam_lm_evaluation.json"
        results[f"{model_name}_greedy"] = load_result(greedy_path, "greedy")
        results[f"{model_name}_beam_no_lm"] = load_result(beam_path, "beam")
        results[f"{model_name}_beam_lm"] = load_result(lm_path, "beam_lm")

    dataset_signatures = {
        (result.get("split"), result.get("evaluated_rows"))
        for result in results.values()
    }
    if len(dataset_signatures) != 1:
        raise RuntimeError(
            "Decoder results do not use the same split and row count: "
            f"{sorted(dataset_signatures)}"
        )

    comparisons = {}
    for model_name in ("beta", "delta"):
        greedy = results[f"{model_name}_greedy"]
        no_lm = results[f"{model_name}_beam_no_lm"]
        lm = results[f"{model_name}_beam_lm"]
        greedy_metrics = greedy["metrics"]
        no_lm_metrics = no_lm["metrics"]
        lm_metrics = lm["metrics"]
        comparisons[model_name] = {
            "greedy": greedy_metrics,
            "beam_no_lm": no_lm_metrics,
            "beam_lm": lm_metrics,
            "lm_settings": {
                key: lm.get(key)
                for key in (
                    "lm_path",
                    "beam_width",
                    "alpha",
                    "beta",
                    "hotwords",
                    "hotword_weight",
                    "decoder_backend",
                )
            },
            "lm_vs_greedy_wer_percentage_point_change": 100
            * (lm_metrics["shared_wer"] - greedy_metrics["shared_wer"]),
            "lm_vs_no_lm_wer_percentage_point_change": 100
            * (lm_metrics["shared_wer"] - no_lm_metrics["shared_wer"]),
            "lm_vs_greedy_cer_percentage_point_change": 100
            * (lm_metrics["shared_cer"] - greedy_metrics["shared_cer"]),
            "lm_vs_no_lm_cer_percentage_point_change": 100
            * (lm_metrics["shared_cer"] - no_lm_metrics["shared_cer"]),
            "lm_improved_wer_vs_greedy": lm_metrics["shared_wer"]
            < greedy_metrics["shared_wer"],
            "lm_improved_wer_vs_no_lm": lm_metrics["shared_wer"]
            < no_lm_metrics["shared_wer"],
            "lm_improved_cer_vs_greedy": lm_metrics["shared_cer"]
            < greedy_metrics["shared_cer"],
            "lm_improved_cer_vs_no_lm": lm_metrics["shared_cer"]
            < no_lm_metrics["shared_cer"],
        }

    candidates = []
    for result in results.values():
        candidates.append(
            {
                "model_name": result["model_name"],
                "decode_mode": result["decode_mode"],
                "decoder_backend": result["decoder_backend"],
                "shared_wer": result["metrics"]["shared_wer"],
                "shared_cer": result["metrics"]["shared_cer"],
            }
        )
    summary = {
        "experiment": "decoder_lm_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
        "best_shared_wer": min(
            candidates,
            key=lambda row: (row["shared_wer"], row["shared_cer"]),
        ),
        "letter_task_note": (
            "KenLM is optimized for open ASR text and is not expected to improve "
            "isolated letters. A later letter-task path should use closed-set A-Z decoding."
        ),
    }
    output = report_dir / "decoder_lm_comparison_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved LM decoder comparison summary to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
