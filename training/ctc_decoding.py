from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


NEG_INF = float("-inf")


def _log_add(*values: float) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


@dataclass(frozen=True)
class DecodeSettings:
    decode_mode: str = "greedy"
    beam_width: int = 50
    alpha: float = 0.5
    beta: float = 1.0
    lm_path: str | None = None
    hotwords: tuple[str, ...] = ()
    hotword_weight: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hotwords"] = list(self.hotwords)
        return result


def tokenizer_id_maps(processor: Any) -> tuple[list[str], int, set[int]]:
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    labels = [""] * len(vocab)
    for token, token_id in vocab.items():
        labels[int(token_id)] = token

    blank_id = int(tokenizer.pad_token_id)
    delimiter_id = int(tokenizer.word_delimiter_token_id)
    labels[blank_id] = ""
    labels[delimiter_id] = " "

    suppressed_ids = {
        token_id
        for token_id in (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        )
        if token_id is not None and int(token_id) != blank_id
    }
    # pyctcdecode requires one label per logit column. Private-use characters
    # keep suppressed special-token columns valid without making them words.
    for offset, token_id in enumerate(sorted(suppressed_ids), start=1):
        labels[int(token_id)] = chr(0xE000 + offset)
    return labels, blank_id, {int(token_id) for token_id in suppressed_ids}


def mask_suppressed_logits(logits: np.ndarray, suppressed_ids: set[int]) -> np.ndarray:
    masked = np.asarray(logits, dtype=np.float64).copy()
    if masked.ndim != 2:
        raise ValueError(f"Expected [time, vocab] logits, received shape {masked.shape}")
    if suppressed_ids:
        masked[:, sorted(suppressed_ids)] = -1e9
    return masked


def validate_decoder_vocabulary(processor: Any, logits_vocab_size: int) -> dict[str, Any]:
    labels, blank_id, suppressed_ids = tokenizer_id_maps(processor)
    vocab = processor.tokenizer.get_vocab()
    token_ids = sorted(int(token_id) for token_id in vocab.values())
    if token_ids != list(range(len(vocab))):
        raise RuntimeError("Tokenizer IDs are not contiguous from 0 to vocab_size - 1.")
    if logits_vocab_size != len(labels):
        raise RuntimeError(
            f"Model logits contain {logits_vocab_size} tokens but tokenizer contains "
            f"{len(labels)}. Decoder vocabulary mismatch."
        )
    if labels[blank_id] != "":
        raise RuntimeError("CTC blank/pad token was not mapped to an empty label.")
    delimiter_id = int(processor.tokenizer.word_delimiter_token_id)
    if labels[delimiter_id] != " ":
        raise RuntimeError("Wav2Vec2 word delimiter was not mapped to a space.")
    return {
        "vocab_size": len(labels),
        "blank_id": blank_id,
        "word_delimiter_id": delimiter_id,
        "suppressed_token_ids": sorted(suppressed_ids),
    }


def greedy_decode(logits: np.ndarray, processor: Any) -> str:
    import torch

    predicted_ids = torch.from_numpy(np.asarray(logits)).argmax(dim=-1).unsqueeze(0)
    return processor.batch_decode(predicted_ids)[0]


def _prefix_score(
    prefix: tuple[int, ...],
    probabilities: tuple[float, float],
    delimiter_id: int,
    beta: float,
) -> float:
    acoustic = _log_add(*probabilities)
    completed_words = sum(token_id == delimiter_id for token_id in prefix)
    return acoustic + beta * completed_words


def pure_ctc_prefix_beam_search(
    logits: np.ndarray,
    processor: Any,
    *,
    beam_width: int,
    beta: float,
) -> str:
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    labels, blank_id, suppressed_ids = tokenizer_id_maps(processor)
    delimiter_id = int(processor.tokenizer.word_delimiter_token_id)
    masked = mask_suppressed_logits(logits, suppressed_ids)
    maximum = masked.max(axis=-1, keepdims=True)
    log_probs = masked - maximum
    log_probs -= np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))

    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, NEG_INF)
    }
    emitted_ids = [
        token_id
        for token_id in range(masked.shape[1])
        if token_id != blank_id and token_id not in suppressed_ids
    ]
    for frame in log_probs:
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(prefix: tuple[int, ...], blank: float = NEG_INF, nonblank: float = NEG_INF) -> None:
            old_blank, old_nonblank = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (
                _log_add(old_blank, blank),
                _log_add(old_nonblank, nonblank),
            )

        for prefix, (prob_blank, prob_nonblank) in beams.items():
            blank_probability = float(frame[blank_id])
            update(
                prefix,
                blank=_log_add(
                    prob_blank + blank_probability,
                    prob_nonblank + blank_probability,
                ),
            )
            last_token = prefix[-1] if prefix else None
            for token_id in emitted_ids:
                token_probability = float(frame[token_id])
                if token_id == last_token:
                    update(prefix, nonblank=prob_nonblank + token_probability)
                    update(prefix + (token_id,), nonblank=prob_blank + token_probability)
                else:
                    update(
                        prefix + (token_id,),
                        nonblank=_log_add(prob_blank, prob_nonblank) + token_probability,
                    )
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _prefix_score(
                    item[0], item[1], delimiter_id, beta
                ),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(
        beams.items(),
        key=lambda item: _prefix_score(item[0], item[1], delimiter_id, beta),
    )[0]
    return "".join(labels[token_id] for token_id in best_prefix)


class CTCTextDecoder:
    def __init__(self, processor: Any, settings: DecodeSettings):
        if settings.decode_mode not in {"greedy", "beam", "beam_lm"}:
            raise ValueError("decode_mode must be greedy, beam, or beam_lm")
        if settings.decode_mode == "beam_lm" and not settings.lm_path:
            raise ValueError("beam_lm requires --lm_path pointing to a KenLM .arpa or .bin file.")
        if settings.decode_mode != "beam_lm" and settings.lm_path:
            raise ValueError("--lm_path is only valid with --decode_mode beam_lm.")
        if settings.hotwords and settings.decode_mode == "greedy":
            raise ValueError("Hotwords require beam or beam_lm decoding.")
        self.processor = processor
        self.settings = settings
        self.backend = "transformers_greedy"
        self.fallback_reason: str | None = None
        self._pyctcdecode = None

        if settings.decode_mode in {"beam", "beam_lm"}:
            self._initialize_beam_backend()

    def _initialize_beam_backend(self) -> None:
        labels, _, _ = tokenizer_id_maps(self.processor)
        lm_path = Path(self.settings.lm_path).resolve() if self.settings.lm_path else None
        if lm_path is not None:
            if not lm_path.exists():
                raise FileNotFoundError(f"KenLM language model not found: {lm_path}")
            if not lm_path.is_file():
                raise FileNotFoundError(f"KenLM path is not a file: {lm_path}")
            if lm_path.suffix.lower() not in {".arpa", ".bin"}:
                raise ValueError(
                    f"Unsupported KenLM file extension '{lm_path.suffix}'. "
                    "Expected .arpa or .bin."
                )
            if lm_path.stat().st_size == 0:
                raise ValueError(f"KenLM language model is empty: {lm_path}")
        try:
            from pyctcdecode import build_ctcdecoder
        except Exception as exc:
            if self.settings.decode_mode == "beam_lm":
                raise RuntimeError(
                    "LM decoding requires pyctcdecode and KenLM Python bindings. "
                    "No no-LM or greedy fallback was used."
                ) from exc
            if self.settings.hotwords:
                raise RuntimeError(
                    "Hotwords require pyctcdecode. The pure Python no-LM fallback "
                    "does not implement hotword scoring."
                ) from exc
            self.fallback_reason = f"pyctcdecode import failed: {exc}"
            self.backend = "pure_python_ctc_prefix_beam"
            print(f"{self.fallback_reason}; using true pure-Python CTC beam search.")
            return

        kwargs: dict[str, Any] = {
            "labels": labels,
            "alpha": self.settings.alpha,
            "beta": self.settings.beta,
        }
        if lm_path is not None:
            kwargs["kenlm_model_path"] = str(lm_path)
        try:
            self._pyctcdecode = build_ctcdecoder(**kwargs)
            self.backend = "pyctcdecode_with_lm" if lm_path else "pyctcdecode_no_lm"
        except Exception as exc:
            if self.settings.decode_mode == "beam_lm":
                raise RuntimeError(
                    "pyctcdecode could not initialize the requested language model. "
                    f"LM path: {lm_path}. Verify that KenLM bindings are installed and "
                    "the file is a valid .arpa or .bin model. No fallback was used."
                ) from exc
            if self.settings.hotwords:
                raise RuntimeError(
                    "pyctcdecode initialization failed, and hotwords cannot be applied "
                    "by the pure Python no-LM fallback."
                ) from exc
            self.fallback_reason = f"pyctcdecode initialization failed: {exc}"
            self.backend = "pure_python_ctc_prefix_beam"
            print(f"{self.fallback_reason}; using true pure-Python CTC beam search.")

    @property
    def beam_search_used(self) -> bool:
        return self.settings.decode_mode in {"beam", "beam_lm"} and self.backend != "transformers_greedy"

    @property
    def language_model_used(self) -> bool:
        return self.settings.decode_mode == "beam_lm" and self.backend == "pyctcdecode_with_lm"

    def decode(self, logits: np.ndarray) -> str:
        validate_decoder_vocabulary(self.processor, int(np.asarray(logits).shape[-1]))
        if self.settings.decode_mode == "greedy":
            return greedy_decode(logits, self.processor)
        if self._pyctcdecode is not None:
            _, _, suppressed_ids = tokenizer_id_maps(self.processor)
            masked = mask_suppressed_logits(logits, suppressed_ids)
            return self._pyctcdecode.decode(
                masked,
                beam_width=self.settings.beam_width,
                hotwords=list(self.settings.hotwords) or None,
                hotword_weight=self.settings.hotword_weight,
            )
        return pure_ctc_prefix_beam_search(
            logits,
            self.processor,
            beam_width=self.settings.beam_width,
            beta=self.settings.beta,
        )

    def beam_candidates(self, logits: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        """Return compact real beam alternatives when the active backend exposes them."""
        if self._pyctcdecode is None or not hasattr(self._pyctcdecode, "decode_beams"):
            return []

        validate_decoder_vocabulary(self.processor, int(np.asarray(logits).shape[-1]))
        _, _, suppressed_ids = tokenizer_id_maps(self.processor)
        masked = mask_suppressed_logits(logits, suppressed_ids)
        try:
            beams = self._pyctcdecode.decode_beams(
                masked,
                beam_width=self.settings.beam_width,
                hotwords=list(self.settings.hotwords) or None,
                hotword_weight=self.settings.hotword_weight,
            )
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        for beam in list(beams or [])[: max(0, top_k)]:
            text = ""
            score: float | None = None
            confidence: float | None = None
            if isinstance(beam, dict):
                text = str(beam.get("text") or beam.get("candidate") or beam.get("transcript") or "").strip()
                raw_score = beam.get("score", beam.get("logit_score", beam.get("lm_score")))
                raw_confidence = beam.get("confidence")
            elif isinstance(beam, tuple) and beam:
                text = str(beam[0]).strip()
                numeric = [item for item in beam[1:] if isinstance(item, (int, float))]
                raw_score = numeric[0] if numeric else None
                raw_confidence = numeric[1] if len(numeric) > 1 else None
            else:
                text = str(beam).strip()
                raw_score = None
                raw_confidence = None

            if raw_score is not None:
                score = round(float(raw_score), 6)
            if raw_confidence is not None:
                confidence = round(float(raw_confidence), 6)
            if text:
                candidate = {"candidate": text}
                if score is not None:
                    candidate["score"] = score
                if confidence is not None:
                    candidate["confidence"] = confidence
                candidates.append(candidate)
        return candidates

    def metadata(self) -> dict[str, Any]:
        return {
            **self.settings.to_dict(),
            "beam_search": self.beam_search_used,
            "language_model_used": self.language_model_used,
            "decoder_backend": self.backend,
            "decoder_fallback_reason": self.fallback_reason,
            "external_language_model_used": self.language_model_used,
            "decoder_vocabulary": validate_decoder_vocabulary(
                self.processor, len(self.processor.tokenizer.get_vocab())
            ),
        }
