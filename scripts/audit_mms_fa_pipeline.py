#!/usr/bin/env python3
"""Audit train/val row counts through MMS-FA + QC (no training)."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("audit_mms_fa")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-datasets",
        nargs="+",
        default=["smutuvi/ndizi-2", "smutuvi/ndizi-1-2025"],
    )
    p.add_argument("--train-splits", nargs="+", default=["train", "train"])
    p.add_argument("--eval-dataset", default="smutuvi/ndizi-2")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--qc-skip-pre-mms", action="store_true", default=True)
    p.add_argument("--no-qc-skip-pre-mms", action="store_false", dest="qc_skip_pre_mms")
    p.add_argument("--skip-mms-fa", action="store_true", help="Count only through Hub load + text clean")
    args = p.parse_args()

    from src.data.dataset import (
        _apply_qc_mms_fa_to_split,
        _ensure_audio_duration_column,
        _ensure_transcription_column,
        _filter_max_duration,
        _strip_columns,
        hub_ctc_identity_clean_batch,
        merge_splits,
        resolve_columns,
    )
    from src.data.qc import apply_qc_filter, qc_config_for_training
    from src.utils.whisper_config import WhisperTrainingConfig

    if len(args.train_splits) != len(args.train_datasets):
        raise SystemExit("train-splits length must match train-datasets")

    cfg = WhisperTrainingConfig(
        stack="whisper",
        train_datasets=args.train_datasets,
        train_splits=args.train_splits,
        apply_data_qc=True,
        qc_skip_pre_mms=bool(args.qc_skip_pre_mms),
        qc_chunk_long_with_mms_fa=not args.skip_mms_fa,
        qc_chunk_long_with_mms_fa_eval=not args.skip_mms_fa,
        max_input_seconds=None,
    )
    log.info("qc_skip_pre_mms=%s | qc_chunk_long_with_mms_fa=%s", cfg.qc_skip_pre_mms, cfg.qc_chunk_long_with_mms_fa)

    kw = {}
    probe = merge_splits(args.train_datasets[:1], args.train_splits[:1], "audio", None, None)
    audio_col, text_col = resolve_columns(list(probe.column_names))

    parts = [
        merge_splits([did], [sp], audio_col, None, None)
        for did, sp in zip(args.train_datasets, args.train_splits)
    ]
    from datasets import concatenate_datasets

    train = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    log.info("TRAIN raw Hub merge: %d rows (%s)", len(train), list(zip(args.train_datasets, args.train_splits)))

    eval_ds = merge_splits([args.eval_dataset], [args.eval_split], audio_col, None, None)
    log.info("EVAL raw Hub: %d rows (%s:%s)", len(eval_ds), args.eval_dataset, args.eval_split)

    train = _ensure_transcription_column(train, "train")
    eval_ds = _ensure_transcription_column(eval_ds, "validation")
    train = _ensure_audio_duration_column(train, "train")
    eval_ds = _ensure_audio_duration_column(eval_ds, "validation")

    if cfg.max_input_seconds is not None:
        n0 = len(train)
        train = _filter_max_duration(train, float(cfg.max_input_seconds))
        log.info("TRAIN after max_input_seconds=%.1f: %d -> %d", cfg.max_input_seconds, n0, len(train))

    train = _strip_columns(train)
    eval_ds = _strip_columns(eval_ds)
    train = train.map(hub_ctc_identity_clean_batch, batched=True, batch_size=64, desc="clean train")
    eval_ds = eval_ds.map(hub_ctc_identity_clean_batch, batched=True, batch_size=64, desc="clean eval")

    empty_train = sum(1 for x in train["clean_transcription"] if not str(x or "").strip())
    log.info("TRAIN empty clean_transcription: %d / %d", empty_train, len(train))

    if args.skip_mms_fa:
        log.info("Skipping MMS-FA/QC (--skip-mms-fa)")
        return

    qc_cfg = qc_config_for_training(cfg)
    log.info("QC max_dur after MMS_FA bump: %.2f", qc_cfg.max_dur)

    if cfg.qc_skip_pre_mms and cfg.apply_data_qc:
        qc_pre = replace(qc_cfg, max_dur=3600.0)
        n_pre = len(train)
        train_pre = apply_qc_filter(
            train,
            "audio",
            "clean_transcription",
            qc_pre,
            split_label="train PRE (should not run)",
        )
        if len(train_pre) < n_pre:
            log.warning(
                "BUG? Pre-MMS QC ran despite qc_skip_pre_mms=true: %d -> %d",
                n_pre,
                len(train_pre),
            )

    train, _ = _apply_qc_mms_fa_to_split(
        train, cfg, qc_cfg, split_label="train", apply_mms_fa=True, fa_ctx=None
    )
    eval_ds, _ = _apply_qc_mms_fa_to_split(
        eval_ds, cfg, qc_cfg, split_label="validation", apply_mms_fa=True, fa_ctx=None
    )

    log.info("=" * 60)
    log.info("FINAL train rows: %d", len(train))
    log.info("FINAL eval rows: %d", len(eval_ds))
    raw_n = sum(len(p) for p in parts)
    log.info("Train retention vs raw merge: %.1f%% (%d / %d)", 100.0 * len(train) / max(raw_n, 1), len(train), raw_n)


if __name__ == "__main__":
    main()
