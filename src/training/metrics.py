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
from src.data.text_format import combined_asr_score, mean_punctuation_recall

logger = logging.getLogger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    """Host numpy array from logits/predictions (Trainer may pass CUDA tensors)."""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """
    Store greedy token ids (batch, time); avoids saving full vocab logits on disk.

    Must return a **CPU torch.Tensor** — Trainer accumulates predictions with ``.cpu()``,
    not numpy arrays.
    """
    import torch

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    logits = logits.detach()
    if logits.ndim >= 3:
        return logits.argmax(dim=-1).cpu()
    return logits.cpu()


def _prediction_token_ids(predictions: Any) -> np.ndarray:
    """
    Convert Trainer ``predictions`` to per-frame token ids ``(batch, time)``.

    Do **not** argmax along time — that collapses each utterance to one token id
    (the max id in the row), which decodes as a single letter (e.g. ``\"n\"``).
    """
    arr = _to_numpy(predictions)
    if arr.ndim >= 3:
        return arr.argmax(axis=-1)
    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)
    raise ValueError(f"Unexpected prediction shape for CTC metrics: {arr.shape}")


def _batch_decode_ctc(processor: ASRProcessor, token_ids: np.ndarray) -> list[str]:
    if hasattr(processor, "batch_decode"):
        return list(processor.batch_decode(token_ids))
    return list(processor.tokenizer.batch_decode(token_ids))


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
        pred_ids = _prediction_token_ids(pred.predictions)
        lab = pred.label_ids.copy()
        lab[lab == -100] = self.processor.tokenizer.pad_token_id
        pred_str = _batch_decode_ctc(self.processor, pred_ids)
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
        punct_rec = mean_punctuation_recall(label_str, pred_str)
        score = combined_asr_score(wer_v, cer_v, punct_rec)
        return {
            "wer": wer_v,
            "cer": cer_v,
            "punct_recall": punct_rec,
            "score": score,
        }
