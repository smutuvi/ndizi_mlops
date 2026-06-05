# src/training/trainer.py
from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import evaluate
import torch
from transformers import Trainer, TrainingArguments

from src.data.dataset import ASRProcessor
from src.training.metrics import ASRMetrics
from src.utils.config import ASRConfig

logger = logging.getLogger(__name__)


def create_training_args(config: ASRConfig, experiment_name: str) -> TrainingArguments:
    output_dir = os.path.join(config.output_dir, experiment_name)

    if getattr(config, "max_steps", -1) and config.max_steps > 0:
        num_train_epochs = 9999.0
        max_steps = config.max_steps
    else:
        num_train_epochs = float(config.num_epochs)
        max_steps = -1

    if getattr(config, "warmup_steps", 0) and config.warmup_steps > 0:
        warmup_steps = int(config.warmup_steps)
        warmup_ratio = 0.0
    else:
        warmup_steps = 0
        warmup_ratio = float(config.warmup_ratio)

    eval_strategy = str(config.eval_strategy)
    if eval_strategy == "epoch":
        save_strategy = "epoch"
        eval_steps = None
        save_steps = None
    else:
        save_strategy = "steps"
        eval_steps = int(config.eval_steps)
        save_steps = int(config.save_steps)

    use_cuda = torch.cuda.is_available()
    fp16 = bool(config.fp16 and use_cuda and not config.bf16)
    bf16 = bool(config.bf16 and use_cuda)

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        group_by_length=bool(config.group_by_length),
        per_device_train_batch_size=int(config.batch_size),
        per_device_eval_batch_size=int(config.per_device_eval_batch_size or config.batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        dataloader_num_workers=4,
        fp16=fp16,
        bf16=bf16,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        gradient_checkpointing=bool(config.gradient_checkpointing),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=int(config.logging_steps),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        save_total_limit=int(config.save_total_limit),
        push_to_hub=bool(config.push_to_hub),
        hub_model_id=config.hub_model_id,
        report_to=str(config.report_to),
        load_best_model_at_end=True,
        metric_for_best_model=(
            "score" if getattr(config, "use_formatting_score_for_best", False) else "wer"
        ),
        greater_is_better=bool(getattr(config, "use_formatting_score_for_best", False)),
        remove_unused_columns=False,
        seed=int(config.seed),
        data_seed=int(config.seed),
        length_column_name="length",
    )

    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in ta_sig.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy

    if eval_strategy == "steps":
        kwargs["eval_steps"] = eval_steps
        kwargs["save_steps"] = save_steps

    if "save_strategy" in ta_sig.parameters:
        kwargs["save_strategy"] = save_strategy

    if "optim" in ta_sig.parameters and use_cuda:
        kwargs["optim"] = "adamw_torch_fused"

    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in ta_sig.parameters})


def create_asr_trainer(
    model: torch.nn.Module,
    train_dataset,
    eval_dataset,
    data_collator,
    processor: ASRProcessor,
    experiment_name: str,
    config: ASRConfig,
) -> Trainer:
    training_args = create_training_args(config, experiment_name)
    pred_dir = os.path.join(training_args.output_dir, "predictions_json")
    asr_metrics = ASRMetrics(
        processor=processor,
        wer_metric=evaluate.load("wer"),
        cer_metric=evaluate.load("cer"),
        output_dir=pred_dir,
    )

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=asr_metrics.compute_metrics,
    )
    tr_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in tr_sig.parameters:
        trainer_kw["processing_class"] = processor
    elif "tokenizer" in tr_sig.parameters:
        trainer_kw["tokenizer"] = processor

    return Trainer(**{k: v for k, v in trainer_kw.items() if k in tr_sig.parameters})
