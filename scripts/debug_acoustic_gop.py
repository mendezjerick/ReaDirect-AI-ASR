from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from api.dependencies import get_asr_provider, get_config, get_cmudict_loader
from readirect_asr.pronunciation.gop import compute_gop


def main() -> int:
    parser = argparse.ArgumentParser(description="Print acoustic GOP debug output for a local audio file.")
    parser.add_argument("audio_path", help="Path to a learner audio file.")
    parser.add_argument("--expected", required=True, help="Expected text or item.")
    parser.add_argument("--prompt-type", default="word", help="letter, word, sentence, passage, etc.")
    parser.add_argument("--task-type", default="", help="letter_sound, letter_name, word, sentence, etc.")
    parser.add_argument("--transcript", default="", help="Optional existing transcript.")
    args = parser.parse_args()

    provider = get_asr_provider()
    evidence_method = getattr(provider, "phoneme_frame_evidence", None)
    if not callable(evidence_method):
        print(json.dumps({"ok": False, "error": "active ASR provider does not expose phoneme_frame_evidence"}, indent=2))
        return 1

    audio_path = str(Path(args.audio_path).resolve())
    evidence = evidence_method(audio_path)
    result = compute_gop(
        audio_path_or_waveform=audio_path,
        expected_text=args.expected,
        prompt_type=args.prompt_type,
        task_type=args.task_type,
        raw_transcript=args.transcript,
        acoustic_evidence=evidence,
        cmudict_loader=get_cmudict_loader(),
        config=get_config().get("gop", {}),
        audio_quality={"passed": True, "quality_flags": {}},
    )

    printable = {
        "expected": args.expected,
        "transcript": args.transcript,
        "canonical_expected_phonemes": result.get("canonical_expected_phonemes"),
        "decoded_acoustic_phonemes": result.get("decoded_acoustic_phonemes"),
        "alignment_quality": result.get("alignment_quality"),
        "overall_gop_score": result.get("overall_gop_score"),
        "weak_phoneme": result.get("weak_phoneme"),
        "weak_phoneme_score": result.get("weak_phoneme_score"),
        "nearest_confusion": result.get("nearest_confusion"),
        "phoneme_scores": result.get("phoneme_scores"),
        "gop_supported": result.get("gop_supported"),
        "gop_error": result.get("gop_error"),
        "final_gop_decision": result.get("gop_decision"),
    }
    print(json.dumps(printable, indent=2))
    return 0 if result.get("gop_supported") else 2


if __name__ == "__main__":
    raise SystemExit(main())
