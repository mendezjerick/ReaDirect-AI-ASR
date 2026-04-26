from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Laravel-facing AI API response contracts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = {"X-ReaDirect-AI-Token": args.token} if args.token else {}
    try:
        health = httpx.get(f"{args.base_url}/health", timeout=10, headers=headers)
        health.raise_for_status()
        text = httpx.post(
            f"{args.base_url}/analyze-text",
            json={"expected_text": "cat", "actual_text": "cap", "accepted_answers": ["cat"]},
            timeout=30,
            headers=headers,
        )
        text.raise_for_status()
        recommend = httpx.post(
            f"{args.base_url}/recommend-next",
            json={
                "learner_history": [{"is_correct": False, "error_type": "final_sound_error", "skill_signal": "final_consonant"}],
                "candidate_items": [{"prompt_id": "M2-014", "expected_text": "hat", "error_focus": "final_consonant", "difficulty_level": "easy", "is_active": True}],
            },
            timeout=30,
            headers=headers,
        )
        recommend.raise_for_status()
    except Exception as exc:
        print(f"FAIL: API contract test failed: {exc}")
        return 1
    missing_text = _missing(text.json(), {"ok", "request_id", "transcript", "normalized_transcript", "provider", "expected_text", "is_correct", "similarity_label", "error_type", "warnings", "error"})
    missing_recommend = _missing(recommend.json(), {"ok", "selected_item", "ranked_candidates", "learner_summary", "recommendation", "explanation", "warnings"})
    if missing_text or missing_recommend:
        print(f"FAIL: missing analyze fields={missing_text}, missing recommend fields={missing_recommend}")
        return 1
    print("PASS: Laravel API contract fields are present.")
    return 0


def _missing(payload: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - set(payload.keys()))


if __name__ == "__main__":
    raise SystemExit(main())
