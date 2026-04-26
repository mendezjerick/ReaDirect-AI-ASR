from __future__ import annotations

import re
from pathlib import Path


STRESS_RE = re.compile(r"\d$")
ALT_PRON_RE = re.compile(r"\(\d+\)$")


def strip_stress(phoneme: str) -> str:
    return STRESS_RE.sub("", phoneme)


def normalize_word(word: str) -> str:
    cleaned = ALT_PRON_RE.sub("", str(word or "").strip())
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9']", "", cleaned)
    return cleaned


def load_cmudict_dict(path: str | Path) -> dict[str, list[list[str]]]:
    pronunciations: dict[str, list[list[str]]] = {}
    dict_path = Path(path)
    if not dict_path.exists():
        return pronunciations

    with dict_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith(";;;") or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            word = normalize_word(parts[0])
            phones = [strip_stress(part.upper()) for part in parts[1:]]
            if not word or not phones:
                continue
            pronunciations.setdefault(word, []).append(phones)
    return pronunciations


def load_cmudict_phones(path: str | Path) -> dict[str, str]:
    phones: dict[str, str] = {}
    phones_path = Path(path)
    if not phones_path.exists():
        return phones

    with phones_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            phones[parts[0].upper()] = parts[1].lower()
    return phones


def load_cmudict_symbols(path: str | Path) -> set[str]:
    symbols_path = Path(path)
    if not symbols_path.exists():
        return set()

    symbols: set[str] = set()
    with symbols_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                symbols.add(stripped.upper())
                symbols.add(strip_stress(stripped.upper()))
    return symbols


class CMUDictLoader:
    def __init__(
        self,
        dict_path: str | Path = "external_datasets/cmudict/cmudict.dict",
        phones_path: str | Path = "external_datasets/cmudict/cmudict.phones",
        symbols_path: str | Path = "external_datasets/cmudict/cmudict.symbols",
    ) -> None:
        self.dict_path = Path(dict_path)
        self.phones_path = Path(phones_path)
        self.symbols_path = Path(symbols_path)
        self.pronunciations: dict[str, list[list[str]]] = {}
        self.phone_categories: dict[str, str] = {}
        self.symbols: set[str] = set()

    def load(self) -> "CMUDictLoader":
        self.pronunciations = load_cmudict_dict(self.dict_path)
        self.phone_categories = load_cmudict_phones(self.phones_path)
        self.symbols = load_cmudict_symbols(self.symbols_path)
        return self

    def missing_files(self) -> list[str]:
        return [
            str(path)
            for path in (self.dict_path, self.phones_path, self.symbols_path)
            if not path.exists()
        ]

    def normalize_word(self, word: str) -> str:
        return normalize_word(word)

    def get_pronunciations(self, word: str) -> list[list[str]]:
        if not self.pronunciations:
            self.load()
        return self.pronunciations.get(normalize_word(word), [])

    def get_primary_pronunciation(self, word: str) -> list[str] | None:
        pronunciations = self.get_pronunciations(word)
        return pronunciations[0] if pronunciations else None

