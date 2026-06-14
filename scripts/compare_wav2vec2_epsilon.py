from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "asr" / "epsilon"


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing Epsilon evaluation: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def prior_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result.get("metrics", result)
    return {
        key: metrics.get(key)
        for key in (
            "shared_wer",
            "shared_cer",
            "speechocean_wer",
            "speechocean_cer",
            "speechocean_exact_match",
            "readirect_letter_accuracy",
        )
    }


def main() -> int:
    modes = {
        "greedy": load(REPORT_DIR / "epsilon_greedy_evaluation.json"),
        "beam_no_lm": load(REPORT_DIR / "epsilon_beam_no_lm_evaluation.json"),
        "beam_lm": load(REPORT_DIR / "epsilon_beam_lm_evaluation.json"),
    }
    expected = {"greedy": "greedy", "beam_no_lm": "beam", "beam_lm": "beam_lm"}
    for name, result in modes.items():
        if result.get("decode_mode") != expected[name]:
            raise RuntimeError(f"Unexpected decode mode in Epsilon {name} result.")
        if name == "beam_lm" and result.get("language_model_used") is not True:
            raise RuntimeError("Epsilon LM result did not actually use KenLM.")
    heldout = load(REPORT_DIR / "epsilon_slr83_heldout_evaluation.json")
    prior_paths = {
        "beta": PROJECT_ROOT / "reports/asr/decoder_lm_comparison/beta_beam_lm_evaluation.json",
        "delta": PROJECT_ROOT / "reports/asr/decoder_lm_comparison/delta_beam_lm_evaluation.json",
        "gamma": PROJECT_ROOT / "reports/asr/gamma/gamma_shared_validation_evaluation.json",
    }
    comparisons = {
        name: {
            "decode_mode": result["decode_mode"],
            "decoder_backend": result["decoder_backend"],
            "metrics": result["metrics"],
        }
        for name, result in modes.items()
    }
    candidates = [
        {
            "decode_mode": name,
            "shared_wer": result["metrics"]["shared_wer"],
            "shared_cer": result["metrics"]["shared_cer"],
        }
        for name, result in modes.items()
    ]
    best = min(candidates, key=lambda row: (row["shared_wer"], row["shared_cer"]))
    best_metrics = modes[best["decode_mode"]]["metrics"]
    prior = {
        name: metrics
        for name, path in prior_paths.items()
        if (metrics := prior_metrics(path)) is not None
    }
    deltas = {}
    for name, metrics in prior.items():
        if metrics.get("shared_wer") is None or metrics.get("shared_cer") is None:
            continue
        deltas[f"epsilon_best_vs_{name}"] = {
            "shared_wer_percentage_points": 100
            * (best_metrics["shared_wer"] - metrics["shared_wer"]),
            "shared_cer_percentage_points": 100
            * (best_metrics["shared_cer"] - metrics["shared_cer"]),
            "letter_accuracy_percentage_points": (
                100
                * (
                    best_metrics["readirect_letter_accuracy"]
                    - metrics["readirect_letter_accuracy"]
                )
                if metrics.get("readirect_letter_accuracy") is not None
                else None
            ),
        }
    summary = {
        "experiment": "epsilon_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "epsilon_decoders": comparisons,
        "epsilon_best_open_asr": best,
        "epsilon_slr83_heldout": heldout["metrics"],
        "prior_models": prior,
        "comparison_deltas": deltas,
        "deployment_policy": {
            "open_sentence_asr": "Use the best validated open-ASR decoder.",
            "isolated_letters": (
                "Do not use LM beam as the final letter decision. Route letters "
                "to no-LM beam or a closed-set A-Z decoder."
            ),
        },
    }
    output = REPORT_DIR / "epsilon_comparison_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved Epsilon comparison summary to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
