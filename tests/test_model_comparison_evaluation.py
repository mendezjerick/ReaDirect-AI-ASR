from __future__ import annotations

from pathlib import Path

from readirect_asr.evaluation.model_comparison import (
    aggregate_rows,
    correct_expected_centric,
    exact_match,
    infer_prompt_type,
    normalize_eval_text,
    score_pair,
    write_csv,
    write_jsonl,
    write_markdown_report,
    write_summary_json,
)


def test_prompt_type_inference() -> None:
    assert infer_prompt_type("Z") == "letter"
    assert infer_prompt_type("ten") == "word"
    assert infer_prompt_type("I see a tree") == "sentence"


def test_evaluation_normalization() -> None:
    assert normalize_eval_text("Elle.") == "elle"
    assert normalize_eval_text("  THE RED HEN  ") == "the red hen"


def test_scoring_exact_wer_and_cer() -> None:
    assert exact_match("red hen", "RED HEN")
    exact_scores = score_pair("red hen", "red hen")
    assert exact_scores["wer"] == 0.0
    assert exact_scores["cer"] == 0.0
    miss_scores = score_pair("red hen", "red")
    assert miss_scores["wer"] > 0.0
    assert miss_scores["cer"] > 0.0


def test_corrected_scoring_letter_and_word_aliases() -> None:
    assert correct_expected_centric("L", "Elle", "letter").accepted
    assert correct_expected_centric("Z", "ZY", "letter").accepted
    assert correct_expected_centric("ten", "then", "word").accepted
    assert not correct_expected_centric("ten", "banana", "word").accepted


def test_output_writing(tmp_path: Path) -> None:
    rows = [
        {
            "row_id": 1,
            "dataset": "unit",
            "prompt_type": "word",
            "normalized_expected": "ten",
            "wav2vec2_normalized_transcript": "ten",
            "wav2vec2_corrected_transcript": "ten",
            "wav2vec2_raw_wer": 0.0,
            "wav2vec2_corrected_wer": 0.0,
            "wav2vec2_raw_cer": 0.0,
            "wav2vec2_corrected_cer": 0.0,
            "wav2vec2_exact_match": True,
            "wav2vec2_corrected_exact_match": True,
            "wav2vec2_accepted": True,
            "wav2vec2_error": "",
            "whisper_normalized_transcript": "then",
            "whisper_corrected_transcript": "ten",
            "whisper_raw_wer": 1.0,
            "whisper_corrected_wer": 0.0,
            "whisper_raw_cer": 0.25,
            "whisper_corrected_cer": 0.0,
            "whisper_exact_match": False,
            "whisper_corrected_exact_match": True,
            "whisper_accepted": True,
            "whisper_error": "",
            "winner_raw": "wav2vec2",
            "winner_corrected": "tie",
        }
    ]
    summary = {
        "manifest": "unit.jsonl",
        "rows_requested": 1,
        "rows_evaluated": 1,
        "fair_comparison": True,
        "use_correction": True,
        "phoneme_evidence": "not_used_transcript_only",
        "overall": aggregate_rows(rows),
        "by_prompt_type": aggregate_rows(rows, "prompt_type"),
        "by_dataset": aggregate_rows(rows, "dataset"),
        "recommendation": "unit recommendation",
        "notes": ["unit note"],
    }
    write_csv(tmp_path / "wav2vec2_vs_whisper_rows.csv", rows)
    write_jsonl(tmp_path / "wav2vec2_vs_whisper_rows.jsonl", rows)
    write_summary_json(tmp_path / "wav2vec2_vs_whisper_summary.json", summary)
    write_markdown_report(tmp_path / "wav2vec2_vs_whisper_report.md", summary)

    assert (tmp_path / "wav2vec2_vs_whisper_rows.csv").exists()
    assert (tmp_path / "wav2vec2_vs_whisper_rows.jsonl").exists()
    assert (tmp_path / "wav2vec2_vs_whisper_summary.json").exists()
    assert (tmp_path / "wav2vec2_vs_whisper_report.md").exists()
