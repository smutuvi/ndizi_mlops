#!/usr/bin/env python3
"""
Whisper seq2seq fine-tuning entry (same Hub dataset manifests as CTC; different stack).

Requires a JSON/YAML config with ``"stack": "whisper"``. See ``config_files/whisper/``.

Usage:
  python3 scripts/train_whisper.py --config config_files/whisper/ndizi_whisper_small_merged.json
  python3 scripts/train_whisper.py --config path/to/config.json --max-input-seconds 30
  python3 scripts/train_whisper.py --config path/to/config.json --encode-num-proc 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v
    logging.info("Loaded environment from %s", env_path)


def setup_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    setup_logging()
    logger = logging.getLogger("train_whisper")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    load_env_file(project_root / ".env")
    logger.info("Project root: %s", project_root)
    logger.info("Whisper stack: bundled Seq2Seq trainer (src.training.whisper_trainer).")
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description="Train Whisper ASR from JSON/YAML (stack: whisper)")
    parser.add_argument("--config", type=str, required=True)
    dur = parser.add_mutually_exclusive_group()
    dur.add_argument(
        "--max-input-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Override config: drop train/eval rows with audio_duration > SEC.",
    )
    dur.add_argument(
        "--no-max-input-filter",
        action="store_true",
        help="Override config: keep all clip lengths (set max_input_seconds to null).",
    )
    parser.add_argument(
        "--encode-num-proc",
        type=int,
        default=1,
        metavar="N",
        help="Workers for HF Dataset.map during Whisper encoding. Default 1 matches "
        "ndizi_finetune_whisper (multiprocessing often stalls with audio + processor).",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

    from huggingface_hub import login as hf_login
    from transformers import set_seed as huggingface_set_seed

    from src.data.dataset import load_datasets
    from src.data.dataset_encoders import WhisperDatasetEncoder
    from src.models.whisper_factory import (
        apply_whisper_trainable_scope,
        create_whisper_processor_and_limits,
        load_whisper_model_for_training,
        load_whisper_teacher_for_distillation,
        snapshot_trainable_params_cpu,
    )
    from src.training.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
    from src.training.whisper_trainer import create_whisper_seq2seq_trainer
    from src.utils.whisper_config import load_whisper_training_config

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    config = load_whisper_training_config(cfg_path)

    if args.no_max_input_filter:
        config.max_input_seconds = None
        logger.info("CLI --no-max-input-filter: max_input_seconds=None (keep all clip lengths).")
    elif args.max_input_seconds is not None:
        config.max_input_seconds = float(args.max_input_seconds)
        logger.info("CLI --max-input-seconds: max_input_seconds=%s", config.max_input_seconds)

    setup_seed(config.seed)
    huggingface_set_seed(config.seed)

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if tok:
        hf_login(token=tok)
        logger.info("Logged in to Hugging Face Hub")

    logger.info("Loading datasets (shared loader with CTC stack)…")
    train_raw, eval_raw = load_datasets(config)
    logger.info("Train rows: %s | Eval rows: %s", len(train_raw), len(eval_raw))

    experiment_name = config.get_experiment_name()
    model_dir = Path(config.output_dir) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading Whisper processor + label limits (full model loads after dataset encoding)…"
    )
    processor, label_max_len, gen_max_len = create_whisper_processor_and_limits(config)
    logger.info("label_max_len=%s | eval generation_max_length=%s", label_max_len, gen_max_len)

    encode_num_proc = max(1, int(args.encode_num_proc))
    logger.info(
        "Encoding datasets (row-wise map like ndizi_finetune_whisper, num_proc=%s)…",
        encode_num_proc,
    )
    encoder = WhisperDatasetEncoder(processor, text_column="clean_transcription", label_max_length=label_max_len)
    train_ds = encoder.encode_dataset(train_raw, num_proc=encode_num_proc)
    eval_ds = encoder.encode_dataset(eval_raw, num_proc=encode_num_proc)

    logger.info("Loading Whisper model weights from %s", config.get_pretrained_model_path())
    model = load_whisper_model_for_training(config, processor)

    distill_w = float(getattr(config, "distill_kl_weight", 0.0) or 0.0)
    teacher_model = None
    if distill_w > 0.0:
        tp = getattr(config, "teacher_model_path", None)
        tpath = (tp.strip() if isinstance(tp, str) and tp.strip() else None) or config.get_pretrained_model_path()
        logger.info("KL distillation weight=%s; loading frozen teacher from %s", distill_w, tpath)
        teacher_model = load_whisper_teacher_for_distillation(tpath, config, processor)

    scope = str(getattr(config, "trainable_scope", "full") or "full")
    logger.info("Applying trainable_scope=%s", scope)
    model = apply_whisper_trainable_scope(model, config)

    anchor_ref = None
    anchor_w = float(getattr(config, "anchor_to_init_weight", 0.0) or 0.0)
    if anchor_w > 0.0:
        anchor_ref = snapshot_trainable_params_cpu(model)
        logger.info("anchor_to_init_weight=%s (%s trainable parameter tensors snapshotted)", anchor_w, len(anchor_ref))

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = create_whisper_seq2seq_trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processor=processor,
        experiment_name=experiment_name,
        config=config,
        generation_max_length=gen_max_len,
        teacher_model=teacher_model,
        anchor_param_reference=anchor_ref,
    )

    esp = int(getattr(config, "early_stopping_patience", 0) or 0)
    if esp > 0:
        logger.info(
            "Early stopping enabled (patience=%s); best checkpoint selected by lowest eval WER",
            esp,
        )
    if getattr(config, "qc_chunk_long_with_mms_fa_eval", False):
        logger.info("Validation uses MMS_FA chunks (qc_chunk_long_with_mms_fa_eval=true)")

    logger.info("Starting Whisper training…")
    trainer.train()

    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    if best_ckpt:
        logger.info(
            "Best checkpoint: %s | best eval metric (wer, lower=better)=%s",
            best_ckpt,
            best_metric,
        )

    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    with open(model_dir / "training_config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, default=str, indent=2)

    scope = str(getattr(config, "trainable_scope", "full") or "full")
    if scope == "lora":
        logger.info(
            "LoRA checkpoint saved under %s. Eval with evaluate_asr_batch/single "
            "--model_path %s --backend auto (requires peft; adapters are merged at load time).",
            model_dir,
            model_dir,
        )

    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    metrics_out = dict(metrics)
    if getattr(trainer.state, "best_model_checkpoint", None):
        metrics_out["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    if getattr(trainer.state, "best_metric", None) is not None:
        metrics_out["best_metric"] = trainer.state.best_metric
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    train_ds.cleanup_cache_files()
    eval_ds.cleanup_cache_files()
    logger.info("Done. Artifacts under %s", model_dir)


if __name__ == "__main__":
    main()
