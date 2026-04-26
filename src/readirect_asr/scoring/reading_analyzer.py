from __future__ import annotations

from typing import Any

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.answer_matching import match_answer
from readirect_asr.scoring.error_detection import detect_error_type
from readirect_asr.scoring.feedback_hints import generate_feedback_hint
from readirect_asr.scoring.phoneme_comparison import compare_phonemes, get_text_phonemes
from readirect_asr.scoring.skill_signals import infer_skill_signal


def analyze_reading_response(
    expected_text: str,
    actual_text: str,
    accepted_answers: Any = None,
    cmudict_loader: CMUDictLoader | None = None,
    content_metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    matcher = match_answer(expected_text, actual_text, accepted_answers)
    expected_phonemes = get_text_phonemes(expected_text, cmudict_loader)
    actual_phonemes = get_text_phonemes(actual_text, cmudict_loader)
    phoneme_report = compare_phonemes(expected_phonemes, actual_phonemes)
    error_report = detect_error_type(
        expected_text=expected_text,
        actual_text=actual_text,
        accepted_answers=accepted_answers,
        expected_phonemes=expected_phonemes,
        actual_phonemes=actual_phonemes,
        similarity_label=str(matcher["similarity_label"]),
    )
    skill = infer_skill_signal(
        str(error_report["error_type"]),
        expected_text,
        expected_phonemes,
        content_metadata,
    )
    feedback = generate_feedback_hint(str(error_report["error_type"]), str(skill["skill_signal"]))
    return {
        **matcher,
        "expected_phonemes": expected_phonemes,
        "actual_phonemes": actual_phonemes,
        "phoneme_similarity": phoneme_report["phoneme_similarity"],
        "phoneme_edit_distance": phoneme_report["phoneme_edit_distance"],
        "initial_phoneme_match": phoneme_report["initial_phoneme_match"],
        "final_phoneme_match": phoneme_report["final_phoneme_match"],
        "vowel_phoneme_match": phoneme_report["vowel_phoneme_match"],
        "error_type": error_report["error_type"],
        "error_position": error_report["error_position"],
        "feedback_hint": feedback["feedback_hint"],
        "coach_hint_key": feedback["coach_hint_key"],
        "learner_safe_summary": feedback["learner_safe_summary"],
        "skill_signal": skill["skill_signal"],
        "target_phoneme": skill["target_phoneme"],
        "target_position": skill["target_position"],
        "recommended_practice_focus": skill["recommended_practice_focus"],
        "recommended_action": feedback["recommended_action"],
        "difficulty_adjustment": skill["difficulty_adjustment"],
        "analysis_source": "heuristic_transcript_phoneme",
        "explanation": error_report["explanation"],
    }

