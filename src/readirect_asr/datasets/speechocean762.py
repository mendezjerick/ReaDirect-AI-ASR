from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from readirect_asr.audio.preprocessing import get_audio_duration_seconds, list_audio_files
from readirect_asr.datasets.common import (
    age_to_group,
    blank_manifest_row,
    find_dataset_root,
    json_dumps,
    speaker_type_from_age,
)
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.phonemes.phoneme_enricher import enrich_text_phonemes
from readirect_asr.phonemes.phoneme_schema import PhonemeSchema


class Speechocean762Loader:
    marker_files = ("train.json", "test.json")

    def __init__(
        self,
        dataset_dir: str | Path = "external_datasets/speechocean762/extracted",
        cmudict_loader: CMUDictLoader | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.root = find_dataset_root(self.dataset_dir, self.marker_files)
        self.cmudict_loader = cmudict_loader
        self._audio_index: dict[str, Path] | None = None

    def exists(self) -> bool:
        return self.root.exists()

    def discover_audio_files(self) -> list[Path]:
        return list_audio_files(self.root)

    def audio_index(self) -> dict[str, Path]:
        if self._audio_index is None:
            self._audio_index = {path.stem: path for path in self.discover_audio_files()}
        return self._audio_index

    def metadata_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [path for path in (self.root / "train.json", self.root / "test.json") if path.exists()]

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def _score_detail_path(self) -> Path | None:
        candidates = [
            self.root / "resource" / "scores-detail.json",
            self.root / "scores-detail.json",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _load_score_details(self) -> dict[str, Any]:
        path = self._score_detail_path()
        if not path:
            return {}
        return self._load_json(path)

    def _mean_phone_score(self, words: list[dict[str, Any]]) -> float | None:
        values: list[float] = []
        for word in words:
            for value in word.get("phones-accuracy", []):
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
        return round(mean(values), 3) if values else None

    def _mean_word_score(self, words: list[dict[str, Any]]) -> float | None:
        values: list[float] = []
        for word in words:
            try:
                values.append(float(word.get("total")))
            except (TypeError, ValueError):
                continue
        return round(mean(values), 3) if values else None

    def _row_status(self, audio_path: Path | None, text: str, words: list[dict[str, Any]]) -> str:
        warnings: list[str] = []
        if audio_path is None or not audio_path.exists():
            warnings.append("missing_audio")
        if not text:
            warnings.append("missing_transcript")
        if not words:
            warnings.append("missing_scores")
        return "|".join(warnings) if warnings else "ok"

    def _enrich_phonemes(self, text: str) -> dict[str, object]:
        if not text or not self.cmudict_loader:
            return {}
        schema = PhonemeSchema(self.cmudict_loader.phone_categories, self.cmudict_loader.symbols)
        return enrich_text_phonemes(text, self.cmudict_loader, schema)

    def _make_row(
        self,
        recording_id: str,
        item: dict[str, Any],
        split: str,
        score_detail: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        row = blank_manifest_row()
        words = item.get("words") if isinstance(item.get("words"), list) else []
        text = str(item.get("text", "")).strip()
        audio_path = self.audio_index().get(recording_id)
        enriched = self._enrich_phonemes(text)
        speaker = str(item.get("speaker", "")).strip()
        age = item.get("age", "")

        row.update(
            {
                "recording_id": recording_id,
                "dataset_source": "speechocean762",
                "speaker_id_anonymized": f"speechocean762_{speaker}" if speaker else "",
                "speaker_type": speaker_type_from_age(age),
                "age_group": age_to_group(age),
                "gender": str(item.get("gender", "")).strip(),
                "l1_language": "Mandarin",
                "prompt_id": recording_id,
                "prompt_type": "sentence_reading",
                "activity_type": "pronunciation_assessment",
                "prompt_text": text,
                "expected_text": text,
                "manual_transcript": text,
                "audio_path": str(audio_path) if audio_path else "",
                "duration_seconds": get_audio_duration_seconds(audio_path) if audio_path else "",
                "sentence_score": item.get("total", ""),
                "word_score": self._mean_word_score(words) or "",
                "phoneme_score": self._mean_phone_score(words) or "",
                "word_labels": json_dumps(words),
                "phoneme_labels": json_dumps(
                    [
                        {
                            "word": word.get("text"),
                            "phones": word.get("phones"),
                            "phones_accuracy": word.get("phones-accuracy"),
                            "mispronunciations": word.get("mispronunciations", []),
                        }
                        for word in words
                    ]
                ),
                "human_correct": "",
                "error_type": "",
                "recording_condition": "public_dataset",
                "noise_flag": "",
                "license_notes": "Speechocean762 README: free for commercial and non-commercial use; verify license before deployable training.",
                "split": split,
                "notes": "score_detail_available" if score_detail else "",
                "row_status": self._row_status(audio_path, text, words),
            }
        )
        row.update({key: value or "" for key, value in enriched.items()})
        return row

    def to_manifest(self, limit: int | None = None) -> pd.DataFrame:
        if not self.root.exists():
            return pd.DataFrame(columns=blank_manifest_row().keys())

        score_details = self._load_score_details()
        rows: list[dict[str, object]] = []
        for metadata_path in self.metadata_files():
            split = metadata_path.stem
            metadata = self._load_json(metadata_path)
            for recording_id, item in metadata.items():
                if not isinstance(item, dict):
                    continue
                rows.append(self._make_row(recording_id, item, split, score_details.get(recording_id)))
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows, columns=blank_manifest_row().keys())
        return pd.DataFrame(rows, columns=blank_manifest_row().keys())

