from pathlib import Path

import pandas as pd

import scripts.build_manifest as build_manifest_script


def test_build_manifest_from_metadata_and_content_index(tmp_path: Path) -> None:
    content_index = tmp_path / "content_index.csv"
    pd.DataFrame(
        [
            {
                "prompt_id": "M2-001",
                "source_file": "module2.csv",
                "source_group": "modules",
                "module_key": "module_2",
                "task_type": "word_reading",
                "activity_type": "word",
                "prompt_text": "Read cat",
                "expected_text": "cat",
                "accepted_answers": "cat",
                "expected_phonemes": "K AE T",
                "initial_phoneme": "K",
                "vowel_phonemes": "AE",
                "final_phoneme": "T",
                "phoneme_pattern": "CVC",
                "metadata": "{}",
            }
        ]
    ).to_csv(content_index, index=False)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "recording_id,audio_path,prompt_id,learner_id_anonymized,grade_level,manual_transcript,human_correct,error_type,recording_condition,noise_flag,notes\n"
        "sample_001,missing.wav,M2-001,L001,1,cat,1,correct,quiet,0,fake\n",
        encoding="utf-8",
    )

    output = tmp_path / "manifest.csv"
    df = build_manifest_script.build_manifest(
        audio_dir=tmp_path,
        output=output,
        metadata_csv=metadata,
        content_index_path=content_index,
    )

    assert df.loc[0, "expected_text"] == "cat"
    assert df.loc[0, "expected_phonemes"] == "K AE T"
    assert df.loc[0, "row_status"] == "missing_audio"


def test_build_manifest_audio_folder_only(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    output = tmp_path / "manifest.csv"

    df = build_manifest_script.build_manifest(audio_dir=tmp_path, output=output)

    assert len(df) == 1
    assert df.loc[0, "row_status"] == "missing_metadata"

