# src/training/collator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import Wav2Vec2BertProcessor, Wav2Vec2Processor

ASRProcessor = Union[Wav2Vec2Processor, Wav2Vec2BertProcessor]


@dataclass
class DataCollatorCTCWithPadding:
    processor: ASRProcessor
    padding: Union[bool, str] = True

    def __post_init__(self) -> None:
        self._is_w2vbert = isinstance(self.processor, Wav2Vec2BertProcessor)
        self._input_key = "input_features" if self._is_w2vbert else "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{self._input_key: f[self._input_key]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        label_ids = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_ids, padding=self.padding, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return batch
