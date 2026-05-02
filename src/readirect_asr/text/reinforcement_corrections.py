from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from readirect_asr.text.normalization import normalize_for_wer

logger = logging.getLogger("readirect_ai_asr.reinforcement_corrections")

DEFAULT_REINFORCEMENT_DIR = "reinforcement-learning"
DEFAULT_LETTER_REINFORCEMENT_FILE = "letter-reinforcement.csv"


@dataclass(frozen=True)
class ReinforcementRule:
    expected_label_original: str
    transcript_error_original: str
    expected_label_normalized: str
    transcript_error_normalized: str
    source_file: str
    rule_type: str


@dataclass(frozen=True)
class ReinforcementMatch:
    expected_label: str
    matched_transcript: str
    expected_label_normalized: str
    matched_transcript_normalized: str
    source_file: str
    rule_type: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reinforcement_source_file": self.source_file,
            "reinforcement_expected_label": self.expected_label,
            "reinforcement_matched_transcript": self.matched_transcript,
            "reinforcement_match_normalized": {
                "expected_text": self.expected_label_normalized,
                "transcript_error": self.matched_transcript_normalized,
            },
            "reinforcement_match_original": {
                "expected_text": self.expected_label,
                "transcript_error": self.matched_transcript,
            },
        }


@dataclass
class ReinforcementCorrectionTable:
    enabled: bool = True
    corrections_dir: str = DEFAULT_REINFORCEMENT_DIR
    letter_file: str = DEFAULT_LETTER_REINFORCEMENT_FILE
    rules_by_expected: dict[str, dict[str, ReinforcementRule]] = field(default_factory=dict)
    files_loaded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    letter_rules_count: int = 0
    word_rules_count: int = 0

    def match(self, expected_text: str, raw_transcript: str, prompt_type: str) -> ReinforcementMatch | None:
        if not self.enabled or prompt_type not in {"letter", "word"}:
            return None

        normalized_expected = _normalize_cell(expected_text)
        normalized_raw = _normalize_cell(raw_transcript)
        if not normalized_expected or not normalized_raw:
            return None

        rule = self.rules_by_expected.get(normalized_expected, {}).get(normalized_raw)
        if not rule:
            return None
        if prompt_type == "letter" and rule.rule_type != "letter":
            return None

        return ReinforcementMatch(
            expected_label=rule.expected_label_original,
            matched_transcript=rule.transcript_error_original,
            expected_label_normalized=rule.expected_label_normalized,
            matched_transcript_normalized=rule.transcript_error_normalized,
            source_file=rule.source_file,
            rule_type=rule.rule_type,
        )

    def status(self) -> dict[str, Any]:
        return {
            "reinforcement_corrections_enabled": self.enabled,
            "reinforcement_corrections_dir": self.corrections_dir,
            "reinforcement_files_loaded": list(self.files_loaded),
            "reinforcement_letter_rules_count": self.letter_rules_count,
            "reinforcement_word_rules_count": self.word_rules_count,
            "reinforcement_load_warnings": list(self.warnings),
        }


def load_reinforcement_corrections(
    corrections_dir: str | Path = DEFAULT_REINFORCEMENT_DIR,
    letter_file: str = DEFAULT_LETTER_REINFORCEMENT_FILE,
    enabled: bool = True,
) -> ReinforcementCorrectionTable:
    table = ReinforcementCorrectionTable(
        enabled=bool(enabled),
        corrections_dir=str(corrections_dir),
        letter_file=letter_file,
    )
    if not table.enabled:
        return table

    root = _project_root()
    directory = Path(corrections_dir)
    if not directory.is_absolute():
        directory = root / directory

    if not directory.exists():
        warning = f"Reinforcement corrections directory not found: {_display_path(directory)}"
        table.warnings.append(warning)
        logger.warning(warning)
        return table
    if not directory.is_dir():
        warning = f"Reinforcement corrections path is not a directory: {_display_path(directory)}"
        table.warnings.append(warning)
        logger.warning(warning)
        return table

    letter_path = directory / letter_file
    csv_paths: list[Path] = []
    if letter_path.exists():
        csv_paths.append(letter_path)
    else:
        warning = f"Letter reinforcement file not found: {_display_path(letter_path)}"
        table.warnings.append(warning)
        logger.warning(warning)

    for path in sorted(directory.glob("*.csv")):
        if path not in csv_paths:
            csv_paths.append(path)

    for path in csv_paths:
        _load_csv_file(path, table, is_letter_file=path.name == letter_file)

    return table


def reinforcement_status_from_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    table = reinforcement_table_from_config(config or {})
    return table.status()


def reinforcement_table_from_config(config: dict[str, Any] | None = None) -> ReinforcementCorrectionTable:
    active = config or {}
    enabled = _as_bool(active.get("reinforcement_corrections_enabled", True))
    corrections_dir = str(active.get("reinforcement_corrections_dir", DEFAULT_REINFORCEMENT_DIR) or DEFAULT_REINFORCEMENT_DIR)
    letter_file = str(active.get("letter_reinforcement_file", DEFAULT_LETTER_REINFORCEMENT_FILE) or DEFAULT_LETTER_REINFORCEMENT_FILE)
    return _cached_reinforcement_table(corrections_dir, letter_file, enabled)


@lru_cache(maxsize=16)
def _cached_reinforcement_table(corrections_dir: str, letter_file: str, enabled: bool) -> ReinforcementCorrectionTable:
    return load_reinforcement_corrections(corrections_dir=corrections_dir, letter_file=letter_file, enabled=enabled)


def _load_csv_file(path: Path, table: ReinforcementCorrectionTable, is_letter_file: bool) -> None:
    loaded_for_file = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                _warn(table, f"Reinforcement CSV has no header: {_display_path(path)}")
                return
            field_map = {_normalize_header(name): name for name in reader.fieldnames if name}
            for row_number, row in enumerate(reader, start=2):
                parsed = _parse_row(row, field_map, path, row_number, is_letter_file)
                if parsed is None:
                    _warn(table, f"Skipped invalid reinforcement row {row_number} in {_display_path(path)}")
                    continue
                existing = table.rules_by_expected.setdefault(parsed.expected_label_normalized, {})
                existing[parsed.transcript_error_normalized] = parsed
                loaded_for_file += 1
                if parsed.rule_type == "letter":
                    table.letter_rules_count += 1
                else:
                    table.word_rules_count += 1
    except Exception as exc:
        _warn(table, f"Failed to load reinforcement CSV {_display_path(path)}: {exc}")
        return

    if loaded_for_file:
        table.files_loaded.append(path.name)


def _parse_row(
    row: dict[str, str],
    field_map: dict[str, str],
    path: Path,
    row_number: int,
    is_letter_file: bool,
) -> ReinforcementRule | None:
    expected_column = _first_present(field_map, ["letter", "expected", "word", "expected_text"])
    transcript_column = _first_present(field_map, ["transcript_error", "raw_transcript"])
    if not expected_column or not transcript_column:
        return None

    expected_original = str(row.get(expected_column, "") or "").strip()
    transcript_original = str(row.get(transcript_column, "") or "").strip()
    expected_normalized = _normalize_cell(expected_original)
    transcript_normalized = _normalize_cell(transcript_original)
    if not expected_normalized or not transcript_normalized:
        return None

    rule_type = "letter" if is_letter_file or _normalize_header(expected_column) == "letter" else "word"
    if rule_type == "letter" and not (len(expected_normalized) == 1 and expected_normalized.isalpha()):
        logger.warning("Skipped unsafe letter reinforcement row %s in %s", row_number, _display_path(path))
        return None
    if rule_type == "word" and len(expected_normalized.split()) != 1:
        logger.warning("Skipped unsafe word reinforcement row %s in %s", row_number, _display_path(path))
        return None

    return ReinforcementRule(
        expected_label_original=expected_original,
        transcript_error_original=transcript_original,
        expected_label_normalized=expected_normalized,
        transcript_error_normalized=transcript_normalized,
        source_file=_display_path(path),
        rule_type=rule_type,
    )


def _normalize_cell(value: str) -> str:
    return " ".join(normalize_for_wer(str(value or "")).split())


def _normalize_header(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _first_present(field_map: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in field_map:
            return field_map[candidate]
    return None


def _warn(table: ReinforcementCorrectionTable, warning: str) -> None:
    table.warnings.append(warning)
    logger.warning(warning)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
