# src/data/dataset.py — load Hub / local ASR splits for bundled training.
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from src.data.preprocessing import (
    add_may6_text_norm_batch,
    clean_text_batch,
    hub_ctc_identity_clean_batch,
)
from src.data.text_format import format_transcription_batch
from src.data.mms_fa_chunk import expand_long_examples_mms_fa, load_mms_fa_context
from src.data.qc import apply_qc_filter, qc_config_for_training
from src.utils.config import ASRConfig
from src.utils.whisper_config import WhisperTrainingConfig

ASRProcessor = Wav2Vec2Processor | Wav2Vec2BertProcessor

logger = logging.getLogger(__name__)


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def resolve_columns(column_names: List[str]) -> tuple[str, str]:
    audio_col = pick_col(column_names, ["audio", "audio_path", "path", "file", "wav", "speech"])
    text_col = pick_col(
        column_names,
        ["text", "transcript", "sentence", "transcription", "normalized_text"],
    )
    if audio_col is None:
        raise ValueError(f"No audio column in {column_names}")
    if text_col is None:
        raise ValueError(f"No text column in {column_names}")
    return audio_col, text_col


def revision_kw(rev: Optional[str]) -> Dict[str, Any]:
    return {"revision": rev.strip()} if rev else {}


def load_split_audio(
    dataset_id: str,
    split: str,
    audio_col: str,
    dataset_revision: Optional[str],
) -> Dataset:
    kw = revision_kw(dataset_revision) if dataset_revision else {}
    ds = load_dataset(dataset_id, split=split, **kw)
    return ds.cast_column(audio_col, Audio(sampling_rate=16000))


def merge_splits(
    hub_ids: List[str],
    splits: List[str],
    audio_col: str,
    dataset_revision: Optional[str],
    weights: Optional[List[float]],
) -> Dataset:
    assert len(hub_ids) == len(splits)
    parts = [load_split_audio(did, sp, audio_col, dataset_revision) for did, sp in zip(hub_ids, splits)]
    if len(parts) == 1:
        return parts[0]
    if weights is not None:
        if len(weights) != len(parts):
            raise ValueError("train_weights length must match train_datasets")
        total = sum(weights)
        probs = [w / total for w in weights]
        return interleave_datasets(parts, probabilities=probs, seed=42, stopping_strategy="all_exhausted")
    return concatenate_datasets(parts)


def _ensure_transcription_column(dataset: Dataset, split_name: str) -> Dataset:
    if "transcription" in dataset.column_names:
        return dataset
    if "transcript" in dataset.column_names:
        return dataset.rename_column("transcript", "transcription")
    if "text" in dataset.column_names:
        return dataset.rename_column("text", "transcription")
    raise ValueError(
        f"No transcription column in {split_name}; expected text/transcript/transcription. "
        f"Columns: {dataset.column_names}"
    )


def _ensure_audio_duration_column(dataset: Dataset, split_name: str) -> Dataset:
    if "audio_duration" in dataset.column_names:
        return _fill_missing_audio_duration(dataset, split_name)
    if "duration" in dataset.column_names:
        dataset = dataset.rename_column("duration", "audio_duration")
        return _fill_missing_audio_duration(dataset, split_name)
    durations: List[float] = []
    for audio in tqdm(dataset["audio"], total=len(dataset), desc=f"audio_duration {split_name}"):
        try:
            durations.append(len(audio["array"]) / max(int(audio["sampling_rate"]), 1))
        except Exception as e:  # noqa: BLE001
            logger.warning("duration error: %s", e)
            durations.append(0.0)
    return dataset.add_column("audio_duration", durations)


def _fill_missing_audio_duration(dataset: Dataset, split_name: str) -> Dataset:
    """
    Some datasets have a duration/audio_duration column with None values.
    Fill those from the audio arrays so duration filtering doesn't crash.

    Do **not** use ``Dataset.map`` returning full rows here: HuggingFace will
    re-encode the ``Audio`` column through soundfile, and mono 1D arrays can
    trigger ``IndexError: tuple index out of range`` in ``sf.write``.
    Rebuild only the ``audio_duration`` column instead.
    """
    if "audio_duration" not in dataset.column_names:
        return dataset

    col = dataset["audio_duration"]
    if not any(d is None for d in col):
        return dataset

    new_durs: List[float] = []
    for i in tqdm(range(len(dataset)), desc=f"Filling missing audio_duration ({split_name})"):
        d = col[i]
        if d is not None:
            new_durs.append(float(d))
            continue
        row = dataset[i]
        a = row.get("audio")
        try:
            if isinstance(a, dict) and a.get("array") is not None and a.get("sampling_rate") is not None:
                new_durs.append(len(a["array"]) / max(int(a["sampling_rate"]), 1))
            else:
                new_durs.append(0.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("audio_duration fill row %s: %s", i, e)
            new_durs.append(0.0)

    dataset = dataset.remove_columns(["audio_duration"])
    return dataset.add_column("audio_duration", new_durs)


def _filter_max_duration(dataset: Dataset, max_sec: float) -> Dataset:
    return dataset.filter(lambda x: (x.get("audio_duration") is not None) and float(x["audio_duration"]) <= max_sec)


def _apply_qc_mms_fa_to_split(
    dataset: Dataset,
    config: Any,
    qc_cfg: Any,
    *,
    split_label: str,
    apply_mms_fa: bool,
    fa_ctx: Optional[Dict[str, Any]],
) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
    """
    Optional May6 norm, optional MMS_FA chunking, then QC.

    When MMS_FA is on: QC full utterances first (no ``dur_high`` cap so long clips are kept),
    chunk, then audio-only QC on chunks (avoids ``weird_high`` / ``text_long`` on slice labels).
    """
    audio_col_qc, text_col_src = resolve_columns(list(dataset.column_names))
    use_may6_qc_text = bool(getattr(config, "qc_use_may6_text_norm", False))
    if use_may6_qc_text:
        dataset = dataset.map(
            lambda batch: add_may6_text_norm_batch(batch, text_col=text_col_src),
            batched=True,
            batch_size=64,
            desc=f"may6 __text_norm ({split_label})",
        )
        text_col_qc = "__text_norm"
    else:
        text_col_qc = "clean_transcription"

    if apply_mms_fa:
        # Pre-chunk: full text+audio QC, but do not drop for duration (long → MMS_FA next).
        qc_pre = replace(qc_cfg, max_dur=3600.0)
        logger.info(
            "[%s] Pre-MMS_FA QC (full gates, max_dur=3600s — long utterances kept for chunking)",
            split_label,
        )
        dataset = apply_qc_filter(
            dataset, audio_col_qc, text_col_qc, qc_pre, split_label=f"{split_label} pre-mms"
        )
        if fa_ctx is None:
            fa_ctx = load_mms_fa_context(str(getattr(config, "qc_fa_device", "auto")))
        dataset = expand_long_examples_mms_fa(
            dataset,
            audio_col=audio_col_qc,
            text_col=text_col_qc,
            label_col="clean_transcription",
            chunk_seconds=float(getattr(config, "qc_chunk_seconds", 30.0)),
            fa_ctx=fa_ctx,
            split_label=split_label,
        )
        logger.info(
            "[%s] Post-MMS_FA QC (audio gates only; text gates skipped on chunks)",
            split_label,
        )
        dataset = apply_qc_filter(
            dataset,
            audio_col_qc,
            text_col_qc,
            qc_cfg,
            split_label=f"{split_label} post-mms",
            audio_only=True,
        )
    else:
        dataset = apply_qc_filter(dataset, audio_col_qc, text_col_qc, qc_cfg, split_label=split_label)

    if use_may6_qc_text and "__text_norm" in dataset.column_names:
        dataset = dataset.remove_columns(["__text_norm"])
    return dataset, fa_ctx


def _strip_columns(dataset: Dataset) -> Dataset:
    keep = ["audio", "transcription", "audio_duration"]
    if "language" in dataset.column_names:
        keep.append("language")
    if "clean_transcription" in dataset.column_names:
        keep.append("clean_transcription")
    remove = [c for c in dataset.column_names if c not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)
    return dataset


def load_hub_processor(config: ASRConfig) -> ASRProcessor:
    mid = config.get_pretrained_model_path()
    proc = AutoProcessor.from_pretrained(mid)
    if getattr(proc, "tokenizer", None) is None:
        raise RuntimeError(f"AutoProcessor for {mid!r} has no tokenizer; use use_hub_ctc_checkpoint=false + custom vocab.")
    return proc  # type: ignore[return-value]


def build_vocabulary(
    character_set: str,
    add_language_tags: bool = False,
    language_tags: Optional[List[str]] = None,
) -> Dict[str, int]:
    vocab_dict = {v: k for k, v in enumerate(sorted(set(character_set)))}
    if " " not in vocab_dict:
        raise ValueError("character_set must include space for CTC word delimiter mapping")
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    next_idx = max(vocab_dict.values()) + 1
    vocab_dict["<unk>"] = next_idx
    vocab_dict["<pad>"] = next_idx + 1
    if add_language_tags and language_tags:
        for i, tag in enumerate(language_tags):
            vocab_dict[f"[{str(tag).upper()}]"] = next_idx + 2 + i
    return vocab_dict


def create_processor(config: ASRConfig, ctc_dir: str) -> ASRProcessor:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        ctc_dir,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    resolved = config.get_pretrained_model_path()
    if resolved == "facebook/w2v-bert-2.0" or config.pretrained_model == "w2v-BERT":
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        return Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def load_datasets(config: ASRConfig | WhisperTrainingConfig) -> Tuple[Dataset, Dataset]:
    revision = config.dataset_revision.strip() if config.dataset_revision else None

    if config.train_datasets:
        kw0 = revision_kw(revision) if revision else {}
        probe = load_dataset(config.train_datasets[0], split=(config.train_splits or ["train"] * len(config.train_datasets))[0], **kw0)
        audio_col, text_col = resolve_columns(list(probe.column_names))
        train_ids = list(config.train_datasets)
        train_splits = config.train_splits or ["train"] * len(train_ids)
        if len(train_splits) != len(train_ids):
            raise ValueError("train_splits must match train_datasets length")
        train_raw = merge_splits(train_ids, train_splits, audio_col, revision, config.train_weights)
        if text_col != "transcription" and text_col in train_raw.column_names:
            train_raw = train_raw.rename_column(text_col, "transcription")
        eval_ids = list(config.eval_datasets) if config.eval_datasets else train_ids
        eval_splits = config.eval_splits or ["validation"] * len(eval_ids)
        if len(eval_splits) != len(eval_ids):
            raise ValueError("eval_splits must match eval_datasets length")
        eval_raw = merge_splits(eval_ids, eval_splits, audio_col, revision, None)
        if text_col != "transcription" and text_col in eval_raw.column_names:
            eval_raw = eval_raw.rename_column(text_col, "transcription")
    elif config.use_custom_dataset:
        if not config.dataset_path:
            raise ValueError("dataset_path required when use_custom_dataset is True")
        ddict = DatasetDict.load_from_disk(config.dataset_path)
        train_raw = ddict[config.train_split]
        eval_raw = ddict[config.eval_split]
        train_raw = _ensure_transcription_column(train_raw, "train")
        eval_raw = _ensure_transcription_column(eval_raw, "validation")
    else:
        if not config.dataset_path:
            raise ValueError("Either train_datasets or dataset_path must be set")
        ds = load_dataset(config.dataset_path, verification_mode="no_checks")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        if config.language != "all":
            train_raw = ds[config.train_split].filter(lambda x: x["language"] == config.language, batch_size=32)
            eval_raw = ds[config.eval_split].filter(lambda x: x["language"] == config.language, batch_size=32)
        else:
            train_raw = ds[config.train_split]
            eval_raw = ds[config.eval_split]
        train_raw = _ensure_transcription_column(train_raw, "train")
        eval_raw = _ensure_transcription_column(eval_raw, "validation")

    train_raw = _ensure_audio_duration_column(train_raw, "train")
    eval_raw = _ensure_audio_duration_column(eval_raw, "validation")
    if config.max_input_seconds is not None:
        train_raw = _filter_max_duration(train_raw, float(config.max_input_seconds))
        eval_raw = _filter_max_duration(eval_raw, float(config.max_input_seconds))
    train_raw = _strip_columns(train_raw)
    eval_raw = _strip_columns(eval_raw)

    if getattr(config, "format_transcripts", True):
        fmt_kw = dict(
            normalize_oral=bool(getattr(config, "normalize_oral_tokens", False)),
        )
        train_raw = train_raw.map(
            lambda b: format_transcription_batch(b, **fmt_kw),
            batched=True,
            batch_size=64,
            desc="format transcripts (train)",
        )
        eval_raw = eval_raw.map(
            lambda b: format_transcription_batch(b, **fmt_kw),
            batched=True,
            batch_size=64,
            desc="format transcripts (eval)",
        )

    if config.sample:
        train_raw = train_raw.shuffle(seed=config.seed).select(range(min(config.sample_size, len(train_raw))))
        eval_raw = eval_raw.shuffle(seed=config.seed).select(range(min(2000, len(eval_raw))))

    if isinstance(config, WhisperTrainingConfig):
        train_raw = train_raw.map(
            hub_ctc_identity_clean_batch,
            batched=True,
            batch_size=64,
            desc="text (Whisper: minimal strip)",
        )
        eval_raw = eval_raw.map(
            hub_ctc_identity_clean_batch,
            batched=True,
            batch_size=64,
            desc="text (Whisper: minimal strip)",
        )
    elif config.use_hub_ctc_checkpoint:
        train_raw = train_raw.map(hub_ctc_identity_clean_batch, batched=True, batch_size=64, desc="text (hub CTC)")
        eval_raw = eval_raw.map(hub_ctc_identity_clean_batch, batched=True, batch_size=64, desc="text (hub CTC)")
    else:
        lc = bool(getattr(config, "lowercase_ctc_labels", True))

        def _clean(batch):
            return clean_text_batch(
                batch,
                config.character_set,
                config.apply_accent_replacements,
                lowercase=lc,
            )

        train_raw = train_raw.map(_clean, batched=True, batch_size=64, desc="clean train text")
        eval_raw = eval_raw.map(_clean, batched=True, batch_size=64, desc="clean eval text")

    if getattr(config, "apply_data_qc", False):
        qc_cfg = qc_config_for_training(config)
        chunk_train = bool(getattr(config, "qc_chunk_long_with_mms_fa", False))
        chunk_eval = chunk_train and bool(getattr(config, "qc_chunk_long_with_mms_fa_eval", False))

        if chunk_train and config.max_input_seconds is not None:
            logger.warning(
                "max_input_seconds=%s drops long clips before MMS_FA chunking; "
                "set max_input_seconds to null or run with --no-max-input-filter.",
                config.max_input_seconds,
            )
        if chunk_train:
            logger.info(
                "MMS_FA train chunking (chunk_seconds=%.1f, device=%s)",
                float(getattr(config, "qc_chunk_seconds", 30.0)),
                getattr(config, "qc_fa_device", "auto"),
            )
        if chunk_eval:
            logger.info(
                "MMS_FA eval/validation chunking enabled (same chunk_seconds=%.1f) for early stopping",
                float(getattr(config, "qc_chunk_seconds", 30.0)),
            )
        elif chunk_train:
            logger.warning(
                "qc_chunk_long_with_mms_fa_eval is false: validation stays unchunked while train uses MMS_FA; "
                "eval WER may not match chunked test decode."
            )

        if getattr(config, "qc_use_may6_text_norm", False):
            logger.info("QC text: May 6 normalize_text → __text_norm (labels stay on clean_transcription)")

        fa_ctx: Optional[Dict[str, Any]] = None
        if chunk_train:
            fa_ctx = load_mms_fa_context(str(getattr(config, "qc_fa_device", "auto")))

        train_raw, fa_ctx = _apply_qc_mms_fa_to_split(
            train_raw,
            config,
            qc_cfg,
            split_label="train",
            apply_mms_fa=chunk_train,
            fa_ctx=fa_ctx,
        )
        if chunk_eval:
            eval_raw, _ = _apply_qc_mms_fa_to_split(
                eval_raw,
                config,
                qc_cfg,
                split_label="validation",
                apply_mms_fa=True,
                fa_ctx=fa_ctx,
            )

    return train_raw, eval_raw
