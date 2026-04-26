from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.adaptive.recommendation import AdaptiveRecommendationEngine
from readirect_asr.content.content_repository import ContentRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate adaptive tutoring recommendations.")
    parser.add_argument("--history-csv", default=None)
    parser.add_argument("--content-index", default="data/manifests/content_index.csv")
    parser.add_argument("--enriched-index", default="content_bank_enriched/enriched_content_index.csv")
    parser.add_argument("--module-key", default=None)
    parser.add_argument("--output", default="reports/adaptive_simulation_report.md")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    histories = load_histories(args.history_csv)
    repo = ContentRepository(
        content_index_path=args.content_index,
        enriched_content_index_path=args.enriched_index,
        prefer_enriched_content=True,
    ).load()
    engine = AdaptiveRecommendationEngine(repo)
    lines = ["# Adaptive Simulation Report", ""]
    for name, history in histories.items():
        result = engine.recommend_next(
            history=history,
            current_context={"module_key": args.module_key} if args.module_key else {},
            top_k=args.top_k,
            debug=True,
        )
        selected = result.get("selected_item") or {}
        recommendation = result.get("recommendation") or {}
        explanation = result.get("explanation") or {}
        print(f"{name}: {selected.get('prompt_id', 'none')} -> {recommendation.get('reason_code')}")
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Selected item: `{selected.get('prompt_id', 'none')}`",
                f"- Expected text: `{selected.get('expected_text', '')}`",
                f"- Focus: `{recommendation.get('primary_focus', '')}`",
                f"- Difficulty adjustment: `{recommendation.get('difficulty_adjustment', '')}`",
                f"- Reason: `{recommendation.get('reason_code', '')}`",
                f"- Learner summary: {explanation.get('learner_safe_summary', '')}",
                "",
            ]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


def load_histories(path: str | None) -> dict[str, list[dict[str, Any]]]:
    if path and Path(path).exists():
        df = pd.read_csv(path).fillna("")
        return {"history_csv": df.to_dict(orient="records")}
    return {
        "final_sound_error": [
            {
                "prompt_id": "M2-001",
                "expected_text": "cat",
                "actual_text": "cap",
                "is_correct": False,
                "error_type": "final_sound_error",
                "skill_signal": "final_consonant",
                "target_phoneme": "T",
                "difficulty_level": "easy",
            }
        ],
        "vowel_error": [
            {
                "expected_text": "cat",
                "actual_text": "cut",
                "is_correct": False,
                "error_type": "vowel_error",
                "skill_signal": "vowel_sound",
                "difficulty_level": "easy",
            }
        ],
        "skipped_word": [
            {
                "expected_text": "the cat ran",
                "actual_text": "cat ran",
                "is_correct": False,
                "error_type": "skipped_word",
                "skill_signal": "sentence_tracking",
                "difficulty_level": "medium",
            }
        ],
        "correct_streak": [
            {"is_correct": True, "skill_signal": "word_reading", "difficulty_level": "easy"},
            {"is_correct": True, "skill_signal": "word_reading", "difficulty_level": "easy"},
            {"is_correct": True, "skill_signal": "word_reading", "difficulty_level": "easy"},
        ],
        "unclear_asr": [
            {
                "is_correct": False,
                "error_type": "unclear_asr",
                "skill_signal": "retry_recording",
                "difficulty_level": "easy",
            }
        ],
    }


if __name__ == "__main__":
    main()
