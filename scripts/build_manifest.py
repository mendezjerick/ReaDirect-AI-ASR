from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.audio.preprocessing import get_audio_duration_seconds, list_audio_files
from readirect_asr.content.index import content_index_from_dataframe
from readirect_asr.content.loader import build_content_index
from readirect_asr.dataset.manifest import REQUIRED_MANIFEST_COLUMNS, save_manifest


def _blank_row() -> dict[str, object]:
    return {column: "" for column in REQUIRED_MANIFEST_COLUMNS}


def _resolve_audio_candidate(audio_path: str, audio_dir: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    if str(audio_path).replace("\\", "/").startswith("data/raw/"):
        return path
    return audio_dir / path


def _row_status(audio_path: str, audio_dir: Path, has_metadata: bool, prompt_found: bool) -> str:
    if not has_metadata:
        return "missing_metadata"
    candidate = _resolve_audio_candidate(audio_path, audio_dir)
    if not candidate.exists():
        return "missing_audio"
    if not prompt_found:
        return "prompt_not_found"
    return "ready"


def load_or_build_content_index(
    content_index_path: Path | None,
    content_bank_path: Path | None,
) -> pd.DataFrame:
    if content_index_path and content_index_path.exists():
        return pd.read_csv(content_index_path)
    if content_bank_path:
        return build_content_index(content_bank_path).to_dataframe()
    return pd.DataFrame()


def build_manifest(
    audio_dir: Path,
    output: Path,
    metadata_csv: Path | None = None,
    content_index_path: Path | None = None,
    content_bank_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    content_df = load_or_build_content_index(content_index_path, content_bank_path)
    content_index = content_index_from_dataframe(content_df) if not content_df.empty else None

    if metadata_csv and metadata_csv.exists():
        metadata = pd.read_csv(metadata_csv)
        for _, metadata_row in metadata.fillna("").iterrows():
            row = _blank_row()
            prompt_id = str(metadata_row.get("prompt_id", "")).strip()
            item = content_index.get_item(prompt_id) if content_index and prompt_id else None
            audio_path = str(metadata_row.get("audio_path", "")).strip()
            audio_candidate = _resolve_audio_candidate(audio_path, audio_dir)

            row.update(
                {
                    "recording_id": metadata_row.get("recording_id", ""),
                    "dataset_source": metadata_row.get("dataset_source", "readirect_local"),
                    "learner_id_anonymized": metadata_row.get("learner_id_anonymized", ""),
                    "speaker_id_anonymized": metadata_row.get("speaker_id_anonymized", metadata_row.get("learner_id_anonymized", "")),
                    "speaker_type": metadata_row.get("speaker_type", "learner"),
                    "age_group": metadata_row.get("age_group", ""),
                    "grade_level": metadata_row.get("grade_level", ""),
                    "prompt_id": prompt_id,
                    "audio_path": audio_path,
                    "duration_seconds": get_audio_duration_seconds(audio_candidate) or metadata_row.get("duration_seconds", ""),
                    "manual_transcript": metadata_row.get("manual_transcript", ""),
                    "human_correct": metadata_row.get("human_correct", ""),
                    "error_type": metadata_row.get("error_type", ""),
                    "recording_condition": metadata_row.get("recording_condition", ""),
                    "noise_flag": metadata_row.get("noise_flag", ""),
                    "license_notes": metadata_row.get("license_notes", ""),
                    "notes": metadata_row.get("notes", ""),
                    "row_status": _row_status(audio_path, audio_dir, True, item is not None),
                }
            )

            if item:
                row.update(
                    {
                        "prompt_type": item.task_type or "",
                        "module_key": item.module_key or "",
                        "activity_type": item.activity_type or "",
                        "expected_text": item.expected_text,
                        "accepted_answers": item.accepted_answers,
                        "expected_phonemes": item.expected_phonemes or "",
                        "initial_phoneme": item.initial_phoneme or "",
                        "vowel_phonemes": item.vowel_phonemes or "",
                        "final_phoneme": item.final_phoneme or "",
                        "phoneme_pattern": item.phoneme_pattern or "",
                    }
                )
            rows.append(row)
    else:
        for index, audio_file in enumerate(list_audio_files(audio_dir), start=1):
            row = _blank_row()
            row.update(
                {
                    "recording_id": audio_file.stem,
                    "dataset_source": "audio_folder_scan",
                    "speaker_id_anonymized": f"speaker_{index:04d}",
                    "speaker_type": "unknown",
                    "audio_path": audio_file.relative_to(audio_dir).as_posix(),
                    "duration_seconds": get_audio_duration_seconds(audio_file) or "",
                    "notes": "Generated from audio scan; add metadata CSV before analysis.",
                    "row_status": "missing_metadata",
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    save_manifest(df, output)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ReaDirect ASR dataset manifest.")
    parser.add_argument("--audio-dir", default="data/raw", type=Path)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--content-index", type=Path, default=None)
    parser.add_argument("--content-bank", type=Path, default=None)
    parser.add_argument("--output", default="data/manifests/dataset_manifest.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(
        audio_dir=args.audio_dir,
        output=args.output,
        metadata_csv=args.metadata_csv,
        content_index_path=args.content_index,
        content_bank_path=args.content_bank,
    )
    print(f"Wrote {len(df)} manifest rows to {args.output}")
    print(df["row_status"].value_counts().to_string() if not df.empty else "No rows created.")


if __name__ == "__main__":
    main()

