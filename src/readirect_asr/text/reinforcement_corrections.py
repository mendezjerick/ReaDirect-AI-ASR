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
DEFAULT_WORD_REINFORCEMENT_FILE = "word-reinforcement.csv"
DEFAULT_AUDIT_FILE = "reinforcement-audit.log"
REINFORCEMENT_COLUMNS = [
    "expected_text",
    "raw_transcript",
    "normalized_expected",
    "normalized_raw",
    "prompt_type",
    "source",
    "created_at",
    "created_by",
    "notes",
]
LETTER_PROMPT_TYPES = {"letter"}
WORD_PROMPT_TYPES = {
    "word",
    "rhyme",
    "rhyming_word",
    "sentence",
    "paragraph",
    "passage",
    "final_sentence",
    "reading_passage",
}
NO_CORRECTION_STRATEGIES = {"none", "wav2vec2_sentence_wer_cer_scoring", "audio_quality_gate"}


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
    word_file: str = DEFAULT_WORD_REINFORCEMENT_FILE
    rules_by_expected: dict[str, dict[str, ReinforcementRule]] = field(default_factory=dict)
    files_loaded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    letter_rules_count: int = 0
    word_rules_count: int = 0
    file_mtimes: dict[str, float | None] = field(default_factory=dict)

    def match(self, expected_text: str, raw_transcript: str, prompt_type: str) -> ReinforcementMatch | None:
        route = route_for_prompt_type(prompt_type)
        if not self.enabled or route is None:
            return None

        normalized_expected = _normalize_cell(expected_text)
        normalized_raw = _normalize_cell(raw_transcript)
        if not normalized_expected or not normalized_raw:
            return None

        rule = self.rules_by_expected.get(normalized_expected, {}).get(normalized_raw)
        if not rule:
            return None
        if rule.rule_type != route:
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

    def stale(self) -> bool:
        if not self.enabled:
            return False
        root = _project_root()
        for display_path, loaded_mtime in self.file_mtimes.items():
            path = root / display_path
            current_mtime = path.stat().st_mtime if path.exists() else None
            if current_mtime != loaded_mtime:
                return True
        return False


def load_reinforcement_corrections(
    corrections_dir: str | Path = DEFAULT_REINFORCEMENT_DIR,
    letter_file: str = DEFAULT_LETTER_REINFORCEMENT_FILE,
    word_file: str = DEFAULT_WORD_REINFORCEMENT_FILE,
    enabled: bool = True,
) -> ReinforcementCorrectionTable:
    table = ReinforcementCorrectionTable(
        enabled=bool(enabled),
        corrections_dir=str(corrections_dir),
        letter_file=letter_file,
        word_file=word_file,
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
    word_path = directory / word_file
    csv_paths: list[tuple[Path, str]] = []
    if letter_path.exists():
        csv_paths.append((letter_path, "letter"))
        table.file_mtimes[_display_path(letter_path)] = letter_path.stat().st_mtime
    else:
        warning = f"Letter reinforcement file not found: {_display_path(letter_path)}"
        table.warnings.append(warning)
        logger.warning(warning)

    if word_path.exists():
        csv_paths.append((word_path, "word"))
        table.file_mtimes[_display_path(word_path)] = word_path.stat().st_mtime
    else:
        warning = f"Word reinforcement file not found: {_display_path(word_path)}"
        table.warnings.append(warning)
        logger.warning(warning)

    for path, rule_type in csv_paths:
        _load_csv_file(path, table, rule_type=rule_type)

    return table


def reinforcement_status_from_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    table = reinforcement_table_from_config(config or {})
    return table.status()


def reinforcement_table_from_config(config: dict[str, Any] | None = None) -> ReinforcementCorrectionTable:
    active = config or {}
    enabled = _as_bool(active.get("reinforcement_corrections_enabled", True))
    corrections_dir = str(active.get("reinforcement_corrections_dir", DEFAULT_REINFORCEMENT_DIR) or DEFAULT_REINFORCEMENT_DIR)
    letter_file = str(active.get("letter_reinforcement_file", DEFAULT_LETTER_REINFORCEMENT_FILE) or DEFAULT_LETTER_REINFORCEMENT_FILE)
    word_file = str(active.get("word_reinforcement_file", DEFAULT_WORD_REINFORCEMENT_FILE) or DEFAULT_WORD_REINFORCEMENT_FILE)
    table = _cached_reinforcement_table(corrections_dir, letter_file, word_file, enabled)
    if table.stale():
        _cached_reinforcement_table.cache_clear()
        table = _cached_reinforcement_table(corrections_dir, letter_file, word_file, enabled)
    return table


@lru_cache(maxsize=16)
def _cached_reinforcement_table(corrections_dir: str, letter_file: str, word_file: str, enabled: bool) -> ReinforcementCorrectionTable:
    return load_reinforcement_corrections(corrections_dir=corrections_dir, letter_file=letter_file, word_file=word_file, enabled=enabled)


def append_developer_correction(
    *,
    expected_text: str,
    raw_transcript: str,
    prompt_type: str,
    accepted: bool,
    retry_required: bool = False,
    uncertain: bool = False,
    correction_strategy_used: str = "",
    developer_reinforcement_enabled: bool = False,
    developer_user_role: str = "",
    created_by: str = "",
    source: str = "developer_auto",
    notes: str = "auto-added from developer reinforcement mode",
    corrections_dir: str | Path = DEFAULT_REINFORCEMENT_DIR,
    letter_file: str = DEFAULT_LETTER_REINFORCEMENT_FILE,
    word_file: str = DEFAULT_WORD_REINFORCEMENT_FILE,
) -> dict[str, Any]:
    route = route_for_prompt_type(prompt_type)
    directory = _resolve_corrections_dir(corrections_dir)
    target_file = letter_file if route == "letter" else word_file if route == "word" else ""
    target_path = directory / target_file if target_file else None

    result = {
        "saved": False,
        "target_file": target_file,
        "reason": "",
        "duplicate": False,
    }

    reason = _append_skip_reason(
        expected_text=expected_text,
        raw_transcript=raw_transcript,
        prompt_type=prompt_type,
        route=route,
        accepted=accepted,
        retry_required=retry_required,
        uncertain=uncertain,
        correction_strategy_used=correction_strategy_used,
        developer_reinforcement_enabled=developer_reinforcement_enabled,
        developer_user_role=developer_user_role,
        created_by=created_by,
    )
    if reason:
        result["reason"] = reason
        _audit(directory, expected_text, raw_transcript, prompt_type, target_file, "skipped", reason, created_by)
        return result

    assert target_path is not None
    directory.mkdir(parents=True, exist_ok=True)
    try:
        _migrate_csv_schema(target_path, route or "word")
    except OSError as exc:
        result["reason"] = f"failed to prepare correction CSV: {exc}"
        _audit(directory, expected_text, raw_transcript, prompt_type, target_file, "skipped", result["reason"], created_by)
        return result

    normalized_expected = _normalize_cell(expected_text)
    normalized_raw = _normalize_cell(raw_transcript)
    if _pair_exists(target_path, normalized_expected, normalized_raw):
        result["duplicate"] = True
        result["reason"] = "correction already exists"
        _audit(directory, expected_text, raw_transcript, prompt_type, target_file, "duplicate", result["reason"], created_by)
        return result

    from datetime import datetime, timezone

    row = {
        "expected_text": str(expected_text or "").strip(),
        "raw_transcript": str(raw_transcript or "").strip(),
        "normalized_expected": normalized_expected,
        "normalized_raw": normalized_raw,
        "prompt_type": normalize_prompt_type(prompt_type),
        "source": source or "developer_auto",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by": created_by or developer_user_role or "admin",
        "notes": notes,
    }
    try:
        with target_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REINFORCEMENT_COLUMNS)
            writer.writerow(row)
    except OSError as exc:
        result["reason"] = f"failed to append correction CSV: {exc}"
        _audit(directory, expected_text, raw_transcript, prompt_type, target_file, "skipped", result["reason"], created_by)
        return result

    _cached_reinforcement_table.cache_clear()
    result["saved"] = True
    result["reason"] = "new correction added"
    _audit(directory, expected_text, raw_transcript, prompt_type, target_file, "saved", result["reason"], created_by)
    return result


def route_for_prompt_type(prompt_type: str | None) -> str | None:
    normalized = normalize_prompt_type(prompt_type)
    if normalized in LETTER_PROMPT_TYPES:
        return "letter"
    if normalized in WORD_PROMPT_TYPES:
        return "word"
    return None


def normalize_prompt_type(prompt_type: str | None) -> str:
    return str(prompt_type or "").strip().lower().replace("-", "_").replace(" ", "_")


def _load_csv_file(path: Path, table: ReinforcementCorrectionTable, rule_type: str) -> None:
    loaded_for_file = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                _warn(table, f"Reinforcement CSV has no header: {_display_path(path)}")
                return
            field_map = {_normalize_header(name): name for name in reader.fieldnames if name}
            for row_number, row in enumerate(reader, start=2):
                parsed = _parse_row(row, field_map, path, row_number, rule_type)
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
    rule_type: str,
) -> ReinforcementRule | None:
    expected_column = _first_present(field_map, ["letter", "expected", "word", "expected_text"])
    transcript_column = _first_present(field_map, ["transcript_error", "raw_transcript", "transcript-error"])
    if not expected_column or not transcript_column:
        return None

    expected_original = str(row.get(expected_column, "") or "").strip()
    transcript_original = str(row.get(transcript_column, "") or "").strip()
    expected_normalized = _normalize_cell(expected_original)
    transcript_normalized = _normalize_cell(transcript_original)
    if not expected_normalized or not transcript_normalized:
        return None

    rule_type = "letter" if rule_type == "letter" else "word"
    if rule_type == "letter" and not (len(expected_normalized) == 1 and expected_normalized.isalpha()):
        logger.warning("Skipped unsafe letter reinforcement row %s in %s", row_number, _display_path(path))
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


def _append_skip_reason(
    *,
    expected_text: str,
    raw_transcript: str,
    prompt_type: str,
    route: str | None,
    accepted: bool,
    retry_required: bool,
    uncertain: bool,
    correction_strategy_used: str,
    developer_reinforcement_enabled: bool,
    developer_user_role: str,
    created_by: str,
) -> str:
    normalized_expected = _normalize_cell(expected_text)
    normalized_raw = _normalize_cell(raw_transcript)
    if not developer_reinforcement_enabled:
        return "developer reinforcement mode is off"
    if not _is_developer_role(developer_user_role or created_by):
        return "user is not admin/developer"
    if route is None:
        return "unsupported prompt type"
    if not normalized_expected:
        return "no expected text"
    if not normalized_raw:
        return "raw transcript is empty"
    if normalized_expected == normalized_raw:
        return "raw transcript already equals expected text"
    if retry_required:
        return "bad audio"
    if uncertain:
        return "uncertain audio"
    strategy = normalize_prompt_type(correction_strategy_used)
    if accepted and strategy not in NO_CORRECTION_STRATEGIES:
        return "existing correction accepted transcript"
    return ""


def _is_developer_role(value: str) -> bool:
    normalized = normalize_prompt_type(value)
    return normalized in {"admin", "developer", "system_admin", "school_admin"}


def _pair_exists(path: Path, normalized_expected: str, normalized_raw: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return False
        field_map = {_normalize_header(name): name for name in reader.fieldnames if name}
        expected_column = _first_present(field_map, ["expected_text", "letter", "expected", "word"])
        raw_column = _first_present(field_map, ["raw_transcript", "transcript_error", "transcript-error"])
        for row in reader:
            expected = _normalize_cell(row.get(field_map.get("normalized_expected", "") or "", "") or row.get(expected_column or "", ""))
            raw = _normalize_cell(row.get(field_map.get("normalized_raw", "") or "", "") or row.get(raw_column or "", ""))
            if expected == normalized_expected and raw == normalized_raw:
                return True
    return False


def _migrate_csv_schema(path: Path, rule_type: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=REINFORCEMENT_COLUMNS).writeheader()
        return

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if fieldnames == REINFORCEMENT_COLUMNS:
        return

    field_map = {_normalize_header(name): name for name in fieldnames if name}
    expected_column = _first_present(field_map, ["expected_text", "letter", "expected", "word"])
    raw_column = _first_present(field_map, ["raw_transcript", "transcript_error", "transcript-error"])
    migrated: list[dict[str, str]] = []
    for row in rows:
        expected = str(row.get(expected_column or "", "") or "").strip()
        raw = str(row.get(raw_column or "", "") or "").strip()
        if not expected or not raw:
            continue
        migrated.append(
            {
                "expected_text": expected,
                "raw_transcript": raw,
                "normalized_expected": str(row.get(field_map.get("normalized_expected", "") or "", "") or _normalize_cell(expected)),
                "normalized_raw": str(row.get(field_map.get("normalized_raw", "") or "", "") or _normalize_cell(raw)),
                "prompt_type": str(row.get(field_map.get("prompt_type", "") or "", "") or rule_type),
                "source": str(row.get(field_map.get("source", "") or "", "") or "curated"),
                "created_at": str(row.get(field_map.get("created_at", "") or "", "") or ""),
                "created_by": str(row.get(field_map.get("created_by", "") or "", "") or ""),
                "notes": str(row.get(field_map.get("notes", "") or "", "") or "migrated from older reinforcement CSV schema"),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REINFORCEMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def _audit(directory: Path, expected_text: str, raw_transcript: str, prompt_type: str, target_file: str, status: str, reason: str, created_by: str) -> None:
    from datetime import datetime, timezone

    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    line = "\t".join(
        [
            timestamp,
            f"expected_text={expected_text}",
            f"raw_transcript={raw_transcript}",
            f"prompt_type={prompt_type}",
            f"target_csv={target_file}",
            f"status={status}",
            f"reason={reason}",
            f"created_by={created_by}",
        ]
    )
    with (directory / DEFAULT_AUDIT_FILE).open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def _resolve_corrections_dir(corrections_dir: str | Path) -> Path:
    directory = Path(corrections_dir)
    if not directory.is_absolute():
        directory = _project_root() / directory
    return directory


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
