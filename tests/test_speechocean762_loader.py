from pathlib import Path

from readirect_asr.datasets.speechocean762 import Speechocean762Loader
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _fake_cmudict(tmp_path: Path) -> CMUDictLoader:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("WE W IY1\nCALL K AO1 L\nIT IH1 T\nBEAR B EH1 R\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("W semivowel\nIY vowel\nK stop\nAO vowel\nL liquid\nIH vowel\nT stop\nB stop\nEH vowel\nR liquid\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("W\nIY\nIY1\nK\nAO\nAO1\nL\nIH\nIH1\nT\nB\nEH\nEH1\nR\n", encoding="utf-8")
    return CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()


def _fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "speechocean762"
    wave = root / "WAVE" / "SPEAKER0001"
    wave.mkdir(parents=True)
    (wave / "000010011.WAV").write_bytes(b"fake")
    (root / "train.json").write_text(
        '{'
        '"000010011": {'
        '"accuracy": 8, "completeness": 10, "fluency": 9, "prosodic": 9, "text": "WE CALL IT BEAR", "total": 8,'
        '"words": [{"accuracy": 10, "phones": ["W", "IY0"], "phones-accuracy": [2, 2], "stress": 10, "text": "WE", "total": 10, "mispronunciations": []}],'
        '"speaker": "0001", "gender": "m", "age": 6'
        '},'
        '"000010012": {"text": "", "speaker": "0001", "gender": "m", "age": 6, "words": []}'
        '}',
        encoding="utf-8",
    )
    return root


def test_loader_handles_missing_dataset_dir_gracefully(tmp_path: Path) -> None:
    loader = Speechocean762Loader(tmp_path / "missing")
    df = loader.to_manifest()
    assert df.empty


def test_loader_scans_audio_and_returns_manifest_columns(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path)
    loader = Speechocean762Loader(root, _fake_cmudict(tmp_path))

    df = loader.to_manifest()

    assert len(loader.discover_audio_files()) == 1
    assert "recording_id" in df.columns
    assert df.loc[0, "dataset_source"] == "speechocean762"
    assert df.loc[0, "sentence_score"] == 8
    assert df.loc[0, "word_score"] == 10
    assert df.loc[0, "phoneme_score"] == 2
    assert df.loc[0, "expected_phonemes"] == "W IY K AO L IH T B EH R"
    assert df.loc[1, "row_status"] == "missing_audio|missing_transcript|missing_scores"

