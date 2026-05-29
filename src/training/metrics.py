# src/training/metrics.py
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import evaluate
from transformers.trainer_utils import PredictionOutput

from src.data.dataset import ASRProcessor

logger = logging.getLogger(__name__)


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> np.ndarray:
    return logits.argmax(axis=-1)


class ASRMetrics:
    def __init__(
        self,
        processor: ASRProcessor,
        wer_metric,
        cer_metric,
        output_dir: str | None = None,
    ):
        self.processor = processor
        self.wer_metric = wer_metric
        self.cer_metric = cer_metric
        self.output_dir = output_dir
        self._call_count = 0

    def compute_metrics(self, pred: PredictionOutput) -> Dict[str, float]:
        k = self._call_count
        self._call_count += 1
        pred_ids = np.argmax(pred.predictions, axis=-1)
        lab = pred.label_ids.copy()
        lab[lab == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.tokenizer.batch_decode(pred_ids)
        try:
            label_str = self.processor.tokenizer.batch_decode(lab, group_tokens=False)
        except TypeError:
            label_str = self.processor.tokenizer.batch_decode(lab)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"predictions_{k}.json")
            data = {f"ID_{i}": {"prediction": pred_str[i], "reference": label_str[i]} for i in range(len(pred_str))}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Wrote %s", path)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        wer_v = float(wer["wer"]) if isinstance(wer, dict) else float(wer)
        cer_v = float(cer["cer"]) if isinstance(cer, dict) else float(cer)
        score = (1.0 - (0.5 * wer_v + 0.5 * cer_v)) * 100.0
        return {"wer": wer_v, "cer": cer_v, "score": score}
