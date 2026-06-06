# src/data/dataset_encoders.py
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List

from datasets import Dataset
from transformers import WhisperProcessor, Wav2Vec2BertProcessor, Wav2Vec2Processor

ASRProcessor = Wav2Vec2Processor | Wav2Vec2BertProcessor


def whisper_prepare_example_row(
    example: Dict[str, Any],
    *,
    processor: WhisperProcessor,
    text_column: str,
    label_max_length: int,
) -> Dict[str, Any]:
    """
    One-row Whisper features + labels (same logic as ``ndizi_finetune_whisper.build_prepare_fn``,
    without ``_drop`` — duration filtering is done earlier in ``load_datasets``).
    """
    audio = example["audio"]
    enc = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    )
    feat0 = enc.input_features[0]
    text = example.get(text_column)
    if text is None:
        text = ""
    labels = processor.tokenizer(
        str(text),
        truncation=True,
        max_length=int(label_max_length),
    ).input_ids
    return {
        "input_features": feat0,
        "length": int(feat0.shape[-1]),
        "labels": labels,
    }


class DatasetEncoder(ABC):
    def __init__(self, processor: ASRProcessor):
        self.processor = processor
        self._is_w2vbert = isinstance(processor, Wav2Vec2BertProcessor)
        self._feature_key = "input_features" if self._is_w2vbert else "input_values"

    def _extract_features(self, features):
        if self._is_w2vbert:
            return features.input_features
        return features.input_values

    @abstractmethod
    def batch_encode(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    def encode_dataset(
        self,
        dataset: Dataset,
        batched: bool = True,
        batch_size: int = 32,
        remove_columns: bool = True,
        num_proc: int = 4,
    ) -> Dataset:
        return dataset.map(
            self.batch_encode,
            batched=batched,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names if remove_columns else None,
            desc="Encoding dataset",
        )


class ASRDatasetEncoder(DatasetEncoder):
    def __init__(self, processor: ASRProcessor, text_column: str = "clean_transcription"):
        super().__init__(processor)
        self.text_column = text_column

    def batch_encode(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"]
        features = self.processor(audio_arrays, sampling_rate=sampling_rate)
        batch[self._feature_key] = self._extract_features(features)
        batch["length"] = [len(f) for f in batch[self._feature_key]]
        batch["labels"] = self.processor.tokenizer(
            batch[self.text_column],
            add_special_tokens=False,
        ).input_ids
        return batch


class WhisperDatasetEncoder:
    """Map raw audio + ``clean_transcription`` to Whisper ``input_features`` + token ``labels``."""

    def __init__(
        self,
        processor: WhisperProcessor,
        text_column: str = "clean_transcription",
        label_max_length: int = 444,
    ):
        self.processor = processor
        self.text_column = text_column
        self.label_max_length = int(label_max_length)

    def encode_dataset(
        self,
        dataset: Dataset,
        *,
        remove_columns: bool = True,
        num_proc: int = 1,
        writer_batch_size: int = 256,
    ) -> Dataset:
        """
        Row-wise ``Dataset.map`` (``batched=False``), matching ``ndizi_finetune_whisper.map_dataset``:
        avoids batched multiprocessing stalls at 0%% on many setups.

        ``num_proc`` above 1 is only safe when the map function pickles cleanly; default is 1.
        """
        fn = partial(
            whisper_prepare_example_row,
            processor=self.processor,
            text_column=self.text_column,
            label_max_length=self.label_max_length,
        )
        cols = list(dataset.column_names) if remove_columns else None
        return dataset.map(
            fn,
            remove_columns=cols,
            num_proc=max(1, int(num_proc)),
            desc="Encoding dataset (Whisper)",
            writer_batch_size=max(16, int(writer_batch_size)),
        )
