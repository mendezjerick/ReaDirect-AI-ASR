from __future__ import annotations

from typing import Any


def prepare_whisper_generation_config(
    model: Any,
    processor: Any,
    language: str | None = "en",
    task: str = "transcribe",
    suppress_tokens: list[int] | None = None,
    begin_suppress_tokens: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Patch Whisper generation config across Transformers versions.

    English-only `.en` checkpoints do not need multilingual language IDs. Newer
    Transformers versions may still try to access `lang_to_id` if `language` is
    set, so this helper only sets language when token mappings are available.
    """
    generation_config = getattr(model, "generation_config", None)
    model_config = getattr(model, "config", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    summary: dict[str, Any] = {
        "language_requested": language,
        "task_requested": task,
        "language_set": None,
        "task_set": None,
        "has_lang_to_id": False,
        "forced_decoder_ids": None,
    }
    if generation_config is None:
        return summary

    lang_to_id = _get_token_mapping(tokenizer, "lang_to_id", ["<|en|>", "<|english|>"])
    task_to_id = _get_token_mapping(tokenizer, "task_to_id", ["<|transcribe|>", "<|translate|>"])
    if lang_to_id and not hasattr(generation_config, "lang_to_id"):
        setattr(generation_config, "lang_to_id", lang_to_id)
    if task_to_id and not hasattr(generation_config, "task_to_id"):
        setattr(generation_config, "task_to_id", task_to_id)
    summary["has_lang_to_id"] = hasattr(generation_config, "lang_to_id")

    forced_decoder_ids = _safe_forced_decoder_ids(tokenizer, language, task, bool(lang_to_id), bool(task_to_id))
    if model_config is not None:
        setattr(model_config, "forced_decoder_ids", forced_decoder_ids)
    setattr(generation_config, "forced_decoder_ids", forced_decoder_ids)
    summary["forced_decoder_ids"] = forced_decoder_ids

    if lang_to_id and language:
        safe_language = _safe_language(language, lang_to_id)
        setattr(generation_config, "language", safe_language)
        summary["language_set"] = safe_language
    else:
        # Safe English-only behavior for `.en` Whisper checkpoints.
        setattr(generation_config, "language", None)
        summary["language_set"] = None

    if task_to_id and task:
        setattr(generation_config, "task", task)
        summary["task_set"] = task
    else:
        setattr(generation_config, "task", None)
        summary["task_set"] = None

    if suppress_tokens is not None:
        setattr(generation_config, "suppress_tokens", suppress_tokens)
    elif not hasattr(generation_config, "suppress_tokens"):
        setattr(generation_config, "suppress_tokens", [])
    if begin_suppress_tokens is not None:
        setattr(generation_config, "begin_suppress_tokens", begin_suppress_tokens)
    elif not hasattr(generation_config, "begin_suppress_tokens"):
        setattr(generation_config, "begin_suppress_tokens", [])

    if verbose:
        print(
            "Whisper generation config: "
            f"language={summary['language_set']}, "
            f"task={summary['task_set']}, "
            f"has_lang_to_id={summary['has_lang_to_id']}, "
            f"forced_decoder_ids={summary['forced_decoder_ids']}"
        )
    return summary


def _get_token_mapping(tokenizer: Any, attr_name: str, fallback_tokens: list[str]) -> dict[str, int]:
    mapping = getattr(tokenizer, attr_name, None)
    if isinstance(mapping, dict) and mapping:
        return dict(mapping)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return {}
    built: dict[str, int] = {}
    for token in fallback_tokens:
        try:
            token_id = convert(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id >= 0:
            key = token.strip("<|>").lower()
            built[key] = token_id
            built[token] = token_id
    return built


def _safe_language(language: str, lang_to_id: dict[str, int]) -> str | None:
    candidates = [language, language.lower(), "english" if language.lower() in {"en", "<|en|>"} else language]
    for candidate in candidates:
        if candidate in lang_to_id or f"<|{candidate}|>" in lang_to_id:
            return candidate
    return None


def _safe_forced_decoder_ids(tokenizer: Any, language: str | None, task: str | None, has_lang: bool, has_task: bool) -> list[list[int]] | None:
    get_prompt_ids = getattr(tokenizer, "get_decoder_prompt_ids", None)
    if callable(get_prompt_ids) and (has_lang or has_task):
        try:
            prompt_language = language if has_lang else None
            return get_prompt_ids(language=prompt_language, task=task if has_task else None)
        except Exception:
            return None
    return None
