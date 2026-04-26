import pandas as pd

from readirect_asr.dataset.validation import summarize_manifest, validate_manifest_frame


def test_manifest_validation_detects_missing_columns_and_prompt_ids() -> None:
    df = pd.DataFrame({"recording_id": ["r1"], "prompt_id": ["UNKNOWN"], "audio_path": ["missing.wav"]})
    content_index = pd.DataFrame({"prompt_id": ["KNOWN"]})

    report = validate_manifest_frame(df, ".", content_index)

    assert "expected_text" in report["missing_columns"]
    assert report["prompt_ids_not_found"] == ["UNKNOWN"]
    assert report["missing_audio_files"] == ["missing.wav"]


def test_summarize_manifest_counts_rows() -> None:
    df = pd.DataFrame(
        {
            "dataset_source": ["a", "a", "b"],
            "prompt_type": ["word", "word", "letter"],
            "module_key": ["m1", "m1", "m2"],
            "activity_type": ["x", "x", "y"],
            "error_type": ["correct", "blank", "correct"],
            "expected_text": ["cat", "", "dog"],
            "manual_transcript": ["cat", "", ""],
            "duration_seconds": [1.0, 2.0, ""],
        }
    )

    summary = summarize_manifest(df)

    assert summary["total_recordings"] == 3
    assert summary["total_duration_seconds"] == 3.0
    assert summary["rows_missing_expected_text"] == 1
    assert summary["counts_by_dataset_source"]["a"] == 2

