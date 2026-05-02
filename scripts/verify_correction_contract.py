from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


CASES = [
    ("L", "Elle", True, "L", "L"),
    ("tree", "three", True, "tree", "tree"),
    ("ten", "then", True, "ten", "ten"),
    ("Z", "zy", True, "Z", "Z"),
    ("Z", "They", True, "Z", "Z"),
    ("O", "Oh", True, "O", "O"),
    ("C", "See", True, "C", "C"),
    ("Q", "Cue", True, "Q", "Q"),
    ("W", "Zavil you", True, "W", "W"),
    ("W", "Zebel you", True, "W", "W"),
    ("Z", "Banana", False, "Banana", "Banana"),
    ("C", "They", False, "They", "They"),
    ("tree", "banana", False, "banana", "banana"),
]


def main() -> int:
    failed = 0

    for expected, raw, accepted, corrected, displayed in CASES:
        result = normalize_asr_transcript(raw_transcript=raw, expected_text=expected)
        passed = (
            result.accepted is accepted
            and result.corrected_transcript == corrected
            and result.displayed_transcript == displayed
        )
        failed += 0 if passed else 1

        print(f"Expected: {expected}")
        print(f"Raw: {raw}")
        print(f"Corrected: {result.corrected_transcript}")
        print(f"Displayed: {result.displayed_transcript}")
        print(f"Accepted: {str(result.accepted).lower()}")
        print(f"Accepted by reinforcement: {str(result.accepted_by_reinforcement_match).lower()}")
        print(f"Strategy: {result.correction_strategy_used}")
        print(f"Reason: {result.normalization_reason}")
        print(f"{'PASS' if passed else 'FAIL'}")
        print()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
