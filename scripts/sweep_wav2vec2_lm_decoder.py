from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "asr" / "decoder_lm_comparison"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Delta KenLM alpha/beta sweep and select by WER then CER."
    )
    parser.add_argument("--lm_path", type=Path, required=True)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--hotwords", nargs="*", default=[])
    parser.add_argument("--hotword_weight", type=float, default=5.0)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    lm_path = args.lm_path.resolve()
    if not lm_path.is_file():
        raise FileNotFoundError(f"KenLM language model not found: {lm_path}")
    report_dir = args.report_dir.resolve()
    run_dir = report_dir / "sweep_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for alpha in (0.3, 0.5, 0.7):
        for beta in (0.5, 1.0, 1.5):
            output = run_dir / f"delta_alpha_{alpha:.1f}_beta_{beta:.1f}.json"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_wav2vec2_decoder.py"),
                "--model-name",
                "delta",
                "--decode_mode",
                "beam_lm",
                "--lm_path",
                str(lm_path),
                "--beam_width",
                str(args.beam_width),
                "--alpha",
                str(alpha),
                "--beta",
                str(beta),
                "--hotword_weight",
                str(args.hotword_weight),
                "--output",
                str(output),
            ]
            if args.hotwords:
                command.extend(["--hotwords", *args.hotwords])
            print(f"Running Delta LM sweep: alpha={alpha}, beta={beta}")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            evaluation = json.loads(output.read_text(encoding="utf-8"))
            if evaluation.get("language_model_used") is not True:
                raise RuntimeError(f"KenLM was not used in sweep output: {output}")
            metrics = evaluation["metrics"]
            results.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "beam_width": args.beam_width,
                    "shared_wer": metrics["shared_wer"],
                    "shared_cer": metrics["shared_cer"],
                    "readirect_letter_accuracy": metrics[
                        "readirect_letter_accuracy"
                    ],
                    "decode_seconds_per_sample": evaluation.get("runtime", {}).get(
                        "decode_seconds_per_sample"
                    ),
                    "evaluation_file": str(output),
                }
            )

    best = min(results, key=lambda row: (row["shared_wer"], row["shared_cer"]))
    summary = {
        "experiment": "delta_kenlm_parameter_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "lowest shared WER, then lowest shared CER",
        "lm_path": str(lm_path),
        "beam_width": args.beam_width,
        "hotwords": args.hotwords,
        "hotword_weight": args.hotword_weight,
        "alpha_values": [0.3, 0.5, 0.7],
        "beta_values": [0.5, 1.0, 1.5],
        "results": results,
        "best": best,
    }
    output = report_dir / "delta_lm_sweep.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best}, indent=2))
    print(f"Saved Delta LM sweep to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
