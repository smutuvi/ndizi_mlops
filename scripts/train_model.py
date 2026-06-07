#!/usr/bin/env python3
"""
In-bundle **CTC** ASR training entry (w2v-BERT / wav2vec2 Hub CTC workflows).

For **Whisper** fine-tuning, use ``scripts/train_whisper.py`` with a config that sets
``"stack": "whisper"``.

Usage:
  python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json
  python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged_10epoch.json --max-input-seconds 30
  python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json --no-max-input-filter
  python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json --no-apply-data-qc
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
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
    logger = logging.getLogger("train_model")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    load_env_file(project_root / ".env")
    logger.info("Project root: %s", project_root)
    logger.info(
        "Bundled CTC trainer (uses src/ package). No external ndizi_finetune_w2vbert.py is required."
    )
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description="Train Ndizi CTC ASR from JSON/YAML config")
    parser.add_argument("--config", type=str, required=True)
    dur = parser.add_mutually_exclusive_group()
    dur.add_argument(
        "--max-input-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Override config: drop train/eval rows with audio_duration > SEC (e.g. 30).",
    )
    dur.add_argument(
        "--no-max-input-filter",
        action="store_true",
        help="Override config: keep all clip lengths (set max_input_seconds to null).",
    )
    qc_flags = parser.add_mutually_exclusive_group()
    qc_flags.add_argument(
        "--no-apply-data-qc",
        "--skip-data-qc",
        action="store_true",
        dest="skip_data_qc",
        help="Disable multi-gate QC on train/eval (on by default unless config sets apply_data_qc: false).",
    )
    qc_flags.add_argument(
        "--apply-data-qc",
        "--aggressive-qc",
        action="store_true",
        dest="force_data_qc",
        help="Force QC on even when config sets apply_data_qc: false.",
    )
    parser.add_argument(
        "--encode-num-proc",
        type=int,
        default=None,
        metavar="N",
        help="Parallel workers for dataset feature encoding (default: 1 for w2v-BERT, else up to 4).",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

    from huggingface_hub import login as hf_login
    from transformers import set_seed as huggingface_set_seed

    from dataclasses import asdict

    from src.data.dataset import (
        build_vocabulary,
        create_processor,
        load_datasets,
        load_hub_processor,
    )
    from src.data.dataset_encoders import ASRDatasetEncoder
    from src.models.ctc_factory import apply_ctc_trainable_scope
    from src.models.factory import create_asr_model, create_asr_model_for_custom_vocab, sync_ctc_model_with_tokenizer
    from src.training.collator import DataCollatorCTCWithPadding
    from src.training.trainer import create_asr_trainer
    from src.utils.config import load_config

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    config = load_config(cfg_path)

    if args.no_max_input_filter:
        config.max_input_seconds = None
        logger.info("CLI --no-max-input-filter: max_input_seconds=None (keep all clip lengths).")
    elif args.max_input_seconds is not None:
        config.max_input_seconds = float(args.max_input_seconds)
        logger.info("CLI --max-input-seconds: max_input_seconds=%s", config.max_input_seconds)

    if args.skip_data_qc:
        config.apply_data_qc = False
        config.qc_use_may6_text_norm = False
        config.qc_chunk_long_with_mms_fa = False
        config.qc_chunk_long_with_mms_fa_eval = False
        logger.info("CLI --no-apply-data-qc: disabling training QC filters.")
    elif args.force_data_qc:
        config.apply_data_qc = True
        logger.info("CLI --apply-data-qc: forcing training QC filters on.")

    setup_seed(config.seed)
    huggingface_set_seed(config.seed)

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if tok:
        hf_login(token=tok)
        logger.info("Logged in to Hugging Face Hub")

    logger.info("Loading datasets…")
    train_raw, eval_raw = load_datasets(config)
    logger.info("Train rows: %s | Eval rows: %s", len(train_raw), len(eval_raw))

    experiment_name = config.get_experiment_name()
    model_dir = Path(config.output_dir) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if config.use_hub_ctc_checkpoint:
        processor = load_hub_processor(config)
        model = create_asr_model(config, processor)
    else:
        lang_tags = None
        if config.add_language_tokens and "language" in train_raw.column_names:
            try:
                lang_tags = sorted(str(x) for x in train_raw.unique("language"))
            except Exception:  # noqa: BLE001
                lang_tags = sorted({str(x) for x in train_raw["language"]})
            logger.info("Language tags for vocab: %s", lang_tags)
        vocab_dict = build_vocabulary(
            config.character_set,
            config.add_language_tokens,
            lang_tags,
        )
        ctc_dir = model_dir / "ctc_tokenizer"
        ctc_dir.mkdir(parents=True, exist_ok=True)
        (ctc_dir / "vocab.json").write_text(json.dumps(vocab_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        (ctc_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "word_delimiter_token": "|",
                    "tokenizer_class": "Wav2Vec2CTCTokenizer",
                    "do_lower_case": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote CTC vocab (%d entries) to %s", len(vocab_dict), ctc_dir)
        processor = create_processor(config, str(ctc_dir))
        model = create_asr_model_for_custom_vocab(config, processor)

    scope = str(getattr(config, "trainable_scope", "full") or "full")
    logger.info("Applying trainable_scope=%s", scope)
    model = apply_ctc_trainable_scope(model, config)
    sync_ctc_model_with_tokenizer(model, processor)
    logger.info(
        "CTC tokenizer: vocab=%s pad_id=%s (blank); bos/eos disabled",
        len(processor.tokenizer),
        processor.tokenizer.pad_token_id,
    )
    if scope == "lora":
        try:
            model.print_trainable_parameters()
        except Exception:  # noqa: BLE001
            pass

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Smaller map batches use less RAM when encoding long audio (HF map batch, not train micro-batch).
    encode_map_bs = max(2, min(32, config.batch_size * 2))
    from transformers import Wav2Vec2BertProcessor

    if args.encode_num_proc is not None:
        encode_num_proc = max(1, int(args.encode_num_proc))
    elif isinstance(processor, Wav2Vec2BertProcessor):
        # w2v-BERT fbank + HF Dataset multiprocessing often stalls at 0%% with many workers.
        encode_num_proc = 1
    elif config.sample:
        encode_num_proc = 1
    else:
        encode_num_proc = min(4, os.cpu_count() or 2)
    logger.info(
        "Encoding datasets (map batch_size=%s, num_proc=%s; w2v-BERT fbank is CPU-heavy — "
        "0%% for several minutes is normal with num_proc=1)…",
        encode_map_bs,
        encode_num_proc,
    )
    encoder = ASRDatasetEncoder(processor, text_column="clean_transcription")
    train_ds = encoder.encode_dataset(train_raw, batch_size=encode_map_bs, num_proc=encode_num_proc)
    eval_ds = encoder.encode_dataset(eval_raw, batch_size=encode_map_bs, num_proc=encode_num_proc)

    if len(train_raw) and "clean_transcription" in train_raw.column_names:
        logger.info("Sample clean_transcription[0]=%r", train_raw[0]["clean_transcription"])
    if len(train_ds):
        sample_labels = train_ds[0]["labels"]
        try:
            decoded = processor.tokenizer.decode(sample_labels, group_tokens=False)
        except TypeError:
            decoded = processor.batch_decode([sample_labels], group_tokens=False)[0]
        logger.info("Round-trip decoded label[0]=%r", decoded)

    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = create_asr_trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processor=processor,
        experiment_name=experiment_name,
        config=config,
    )

    logger.info("Starting training…")
    trainer.train()
    if scope == "lora":
        from src.models.ctc_factory import save_ctc_lora_checkpoint

        save_ctc_lora_checkpoint(model, str(model_dir))
    else:
        trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    with open(model_dir / "training_config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, default=str, indent=2)

    if scope == "lora":
        logger.info(
            "LoRA CTC checkpoint saved under %s. Eval with evaluate_asr_batch/single "
            "--model_path %s (requires peft; adapters merge at load by default).",
            model_dir,
            model_dir,
        )

    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    train_ds.cleanup_cache_files()
    eval_ds.cleanup_cache_files()
    logger.info("Done. Artifacts under %s", model_dir)


if __name__ == "__main__":
    main()
