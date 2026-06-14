from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "asr" / "decoder_comparison"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Beta/Delta greedy-versus-beam comparison summary."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
    )
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    results = {}
    required = [
        ("beta", "greedy"),
        ("beta", "beam"),
        ("delta", "greedy"),
        ("delta", "beam"),
    ]
    for model_name, mode in required:
        path = report_dir / f"{model_name}_{mode}_evaluation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing decoder evaluation: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("decode_mode") != mode:
            raise RuntimeError(f"Unexpected decode mode in {path}")
        if mode == "beam" and result.get("beam_search") is not True:
            raise RuntimeError(f"Beam search was not actually used in {path}")
        results[f"{model_name}_{mode}"] = result

    comparisons = {}
    for model_name in ("beta", "delta"):
        greedy = results[f"{model_name}_greedy"]["metrics"]
        beam = results[f"{model_name}_beam"]["metrics"]
        comparisons[model_name] = {
            "greedy": greedy,
            "beam": beam,
            "beam_backend": results[f"{model_name}_beam"]["decoder_backend"],
            "beam_search_verified": results[f"{model_name}_beam"]["beam_search"],
            "wer_percentage_point_change": 100 * (beam["shared_wer"] - greedy["shared_wer"]),
            "cer_percentage_point_change": 100 * (beam["shared_cer"] - greedy["shared_cer"]),
            "letter_accuracy_percentage_point_change": 100
            * (beam["readirect_letter_accuracy"] - greedy["readirect_letter_accuracy"]),
            "wer_improved": beam["shared_wer"] < greedy["shared_wer"],
            "cer_improved": beam["shared_cer"] < greedy["shared_cer"],
        }
    summary = {
        "experiment": "decoder_only_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
        "best_shared_wer": min(
            (
                {
                    "model_name": result["model_name"],
                    "decode_mode": result["decode_mode"],
                    "shared_wer": result["metrics"]["shared_wer"],
                    "shared_cer": result["metrics"]["shared_cer"],
                }
                for result in results.values()
            ),
            key=lambda row: row["shared_wer"],
        ),
    }
    output = report_dir / "decoder_comparison_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved decoder comparison summary to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
