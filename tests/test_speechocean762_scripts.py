from __future__ import annotations

import io
import tarfile
from pathlib import Path

import scripts.build_speechocean762_manifest as build_script
import scripts.extract_speechocean762 as extract_script
import scripts.inspect_speechocean762 as inspect_script


def _fake_cmudict_dir(tmp_path: Path) -> Path:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("HELLO HH AH0 L OW1\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("HH aspirate\nAH vowel\nL liquid\nOW vowel\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("HH\nAH\nAH0\nL\nOW\nOW1\n", encoding="utf-8")
    return cmu


def _fake_dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "WAVE" / "SPEAKER0001").mkdir(parents=True)
    (root / "WAVE" / "SPEAKER0001" / "000010001.WAV").write_bytes(b"fake")
    (root / "train.json").write_text(
        '{"000010001": {"text": "HELLO", "total": 9, "words": [{"text": "HELLO", "total": 9, "phones": ["HH", "AH0"], "phones-accuracy": [2, 1.5]}], "speaker": "0001", "gender": "f", "age": 10}}',
        encoding="utf-8",
    )
    return root


def test_inspect_script_runs_on_fake_folder(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "speechocean762"
    extracted = dataset_dir / "extracted"
    fake = _fake_dataset_dir(tmp_path)
    extracted.mkdir(parents=True)
    (extracted / "train.json").write_text((fake / "train.json").read_text(encoding="utf-8"), encoding="utf-8")

    report = inspect_script.inspect(dataset_dir)

    assert report["archive_exists"] is False
    assert report["extracted_file_count"] == 1


def test_extraction_rejects_unsafe_tar_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        data = b"bad"
        member = tarfile.TarInfo("../evil.txt")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    try:
        extract_script.extract_archive(archive_path, tmp_path / "dest", dry_run=True)
    except ValueError as exc:
        assert "Unsafe tar member" in str(exc)
    else:
        raise AssertionError("unsafe tar path was not rejected")


def test_manifest_builder_works_on_fake_dataset(tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"
    df = build_script.build_manifest(_fake_dataset_dir(tmp_path), _fake_cmudict_dir(tmp_path), output)

    assert output.exists()
    assert len(df) == 1
    assert df.loc[0, "dataset_source"] == "speechocean762"
    assert df.loc[0, "expected_text"] == "HELLO"

