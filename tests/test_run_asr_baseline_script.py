from pathlib import Path

import pandas as pd

import scripts.run_asr_baseline as run_script


def test_run_asr_baseline_with_mock_provider(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "recording_id": "r1",
                "audio_path": str(audio),
                "manual_transcript": "cat",
                "expected_text": "cat",
            },
            {
                "recording_id": "r2",
                "audio_path": str(tmp_path / "missing.wav"),
                "manual_transcript": "dog",
                "expected_text": "dog",
            },
        ]
    ).to_csv(manifest, index=False)
    output = tmp_path / "out.csv"

    df = run_script.run_baseline(
        manifest=manifest,
        output=output,
        provider_name="mock",
        model_size="mock",
        limit=2,
        save_every=1,
    )

    assert output.exists()
    assert df.loc[0, "asr_transcript"] == "cat"
    assert "audio file not found" in df.loc[1, "asr_error"]
    for column in run_script.OUTPUT_COLUMNS:
        assert column in df.columns

