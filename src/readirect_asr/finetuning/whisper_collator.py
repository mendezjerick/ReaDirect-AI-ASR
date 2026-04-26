from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        decoder_start = self.decoder_start_token_id
        if decoder_start is None:
            decoder_start = getattr(self.processor.tokenizer, "bos_token_id", None)
        if decoder_start is not None and labels.shape[1] > 0 and (labels[:, 0] == decoder_start).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
