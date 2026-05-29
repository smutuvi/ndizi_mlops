# src/training/whisper_collator.py — batch padding for Whisper seq2seq training.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad log-mel inputs and decoder labels (matches common HF Whisper fine-tune recipes)."""

    processor: WhisperProcessor

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (
            labels.size(1) > 0
            and self.processor.tokenizer.bos_token_id is not None
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
