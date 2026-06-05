# src/training/whisper_trainer.py — Seq2SeqTrainer setup for Whisper fine-tuning.
from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.training.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.utils.whisper_config import WhisperTrainingConfig


def _whisper_encoder_feature_dtype(module: torch.nn.Module) -> torch.dtype:
    wm: torch.nn.Module = getattr(module, "module", module)
    if hasattr(wm, "module"):
        wm = wm.module  # type: ignore[assignment]
    for mod_name, sm in wm.named_modules():
        if mod_name.endswith("encoder.conv1") and hasattr(sm, "weight"):
            return sm.weight.dtype
    return next(wm.parameters()).dtype


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """Aligns ``input_features`` dtype with encoder weights during generate-based eval."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if (
            not prediction_loss_only
            and self.args.predict_with_generate
            and inputs.get("input_features") is not None
            and getattr(inputs["input_features"], "dtype", None) is not None
        ):
            feats = inputs["input_features"]
            enc_dtype = _whisper_encoder_feature_dtype(model)
            if enc_dtype.is_floating_point and feats.dtype != enc_dtype:
                inputs = dict(inputs)
                inputs["input_features"] = feats.to(enc_dtype)
            # Hub checkpoints often set both max_new_tokens and max_length; Trainer passes
            # generation_max_length — clear max_new_tokens so only max_length applies.
            gen_max = int(getattr(self.args, "generation_max_length", 0) or 0)
            if gen_max > 0:
                from src.models.whisper_factory import apply_whisper_decoder_length_cap

                apply_whisper_decoder_length_cap(model, gen_max)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class ContinualWhisperSeq2SeqTrainer(WhisperSeq2SeqTrainer):
    """
    Optional KL distillation vs a frozen teacher, L2 anchor toward init weights,
    and AdamW param groups for encoder vs decoder learning-rate multipliers.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.teacher_model = kwargs.pop("teacher_model", None)
        self.distill_kl_weight = float(kwargs.pop("distill_kl_weight", 0.0))
        self.distill_temperature = float(kwargs.pop("distill_temperature", 2.0))
        self.anchor_to_init_weight = float(kwargs.pop("anchor_to_init_weight", 0.0))
        self.anchor_param_reference = kwargs.pop("anchor_param_reference", None) or {}
        self.encoder_lr_multiplier = float(kwargs.pop("encoder_lr_multiplier", 1.0))
        self.decoder_lr_multiplier = float(kwargs.pop("decoder_lr_multiplier", 1.0))
        super().__init__(**kwargs)
        self._teacher_device_ready = False

    def _move_teacher_to_model_device(self) -> None:
        if self.teacher_model is None or self._teacher_device_ready:
            return
        device = self.args.device
        if device is not None:
            self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device_ready = True

    @staticmethod
    def _whisper_training_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """Whisper expects ``input_features`` + ``labels``; drop seq2seq extras PEFT may mishandle."""
        allowed = ("input_features", "labels", "attention_mask", "decoder_attention_mask")
        out = {k: inputs[k] for k in allowed if k in inputs}
        return out if out else dict(inputs)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs: Any):
        kwargs.pop("num_items_in_batch", None)
        if (
            self.teacher_model is None
            and self.distill_kl_weight <= 0
            and self.anchor_to_init_weight <= 0
        ):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(result, tuple):
            loss, outputs = result[0], result[1]
        else:
            loss, outputs = result, None

        if loss is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if self.distill_kl_weight > 0 and self.teacher_model is not None and outputs is not None:
            self._move_teacher_to_model_device()
            with torch.no_grad():
                t_out = self.teacher_model(**self._whisper_training_inputs(inputs))
            s_logits = outputs.logits
            t_logits = t_out.logits
            labels = inputs.get("labels")
            if labels is not None and s_logits is not None and t_logits is not None:
                shift_logits_s = s_logits[:, :-1, :].contiguous()
                shift_logits_t = t_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                mask = shift_labels.ne(-100)
                if mask.any():
                    T = max(self.distill_temperature, 1e-6)
                    s_sel = shift_logits_s[mask]
                    t_sel = shift_logits_t[mask]
                    log_p_s = F.log_softmax(s_sel / T, dim=-1)
                    p_t = F.softmax(t_sel / T, dim=-1)
                    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
                    loss = loss + self.distill_kl_weight * kl

        if self.anchor_to_init_weight > 0 and self.anchor_param_reference:
            anchor_sum = loss.new_tensor(0.0)
            for name, p in model.named_parameters():
                if not p.requires_grad or name not in self.anchor_param_reference:
                    continue
                pref = self.anchor_param_reference[name].to(device=p.device, dtype=p.dtype)
                anchor_sum = anchor_sum + (p - pref).pow(2).mean()
            loss = loss + self.anchor_to_init_weight * anchor_sum

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        enc_mul = self.encoder_lr_multiplier
        dec_mul = self.decoder_lr_multiplier
        if abs(enc_mul - 1.0) < 1e-12 and abs(dec_mul - 1.0) < 1e-12:
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        from torch.optim import AdamW

        lr = float(self.args.learning_rate)
        wd = float(self.args.weight_decay)
        encoder_params: list[torch.nn.Parameter] = []
        decoder_params: list[torch.nn.Parameter] = []
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            if "model.encoder" in n:
                encoder_params.append(p)
            else:
                decoder_params.append(p)

        param_groups: list[dict[str, Any]] = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": lr * enc_mul, "weight_decay": wd})
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": lr * dec_mul, "weight_decay": wd})

        if not param_groups:
            return super().create_optimizer()

        beta1 = float(getattr(self.args, "adam_beta1", 0.9))
        beta2 = float(getattr(self.args, "adam_beta2", 0.999))
        eps = float(getattr(self.args, "adam_epsilon", 1e-8))
        self.optimizer = AdamW(param_groups, betas=(beta1, beta2), eps=eps)
        return self.optimizer


def create_whisper_seq2seq_training_args(
    config: WhisperTrainingConfig,
    experiment_name: str,
    generation_max_length: int,
) -> Seq2SeqTrainingArguments:
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
        predict_with_generate=True,
        generation_max_length=int(generation_max_length),
        generation_num_beams=int(config.generation_num_beams),
        remove_unused_columns=False,
        seed=int(config.seed),
        data_seed=int(config.seed),
        length_column_name="length",
    )

    ta_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
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

    return Seq2SeqTrainingArguments(**{k: v for k, v in kwargs.items() if k in ta_sig.parameters})


def build_whisper_compute_metrics(processor, wer_metric, cer_metric, normalize_wer: bool):
    from src.data.preprocessing import wer_normalize as _norm
    from src.data.text_format import combined_asr_score, mean_punctuation_recall

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        if normalize_wer:
            pred_str = [_norm(s) for s in pred_str]
            label_str = [_norm(s) for s in label_str]
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer_v = float(wer["wer"]) if isinstance(wer, dict) else float(wer)
        cer_v = float(cer["cer"]) if isinstance(cer, dict) else float(cer)
        punct_rec = mean_punctuation_recall(label_str, pred_str)
        score = combined_asr_score(wer_v, cer_v, punct_rec)
        return {"wer": wer_v, "cer": cer_v, "punct_recall": punct_rec, "score": score}

    return compute_metrics


def create_whisper_seq2seq_trainer(
    model: torch.nn.Module,
    train_dataset,
    eval_dataset,
    data_collator: DataCollatorSpeechSeq2SeqWithPadding,
    processor,
    experiment_name: str,
    config: WhisperTrainingConfig,
    generation_max_length: int,
    *,
    teacher_model: Optional[torch.nn.Module] = None,
    anchor_param_reference: Optional[dict[str, torch.Tensor]] = None,
) -> ContinualWhisperSeq2SeqTrainer:
    training_args = create_whisper_seq2seq_training_args(config, experiment_name, generation_max_length)

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")
    compute_metrics = build_whisper_compute_metrics(
        processor,
        wer_m,
        cer_m,
        bool(config.normalize_wer),
    )

    callbacks: list[Any] = []
    esp = int(getattr(config, "early_stopping_patience", 0) or 0)
    if esp > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=esp,
                early_stopping_threshold=0.0,
            )
        )
        logger.info(
            "Early stopping: patience=%d on eval WER (load_best_model_at_end=True, "
            "greater_is_better=False)",
            esp,
        )

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        teacher_model=teacher_model,
        distill_kl_weight=float(getattr(config, "distill_kl_weight", 0.0)),
        distill_temperature=float(getattr(config, "distill_temperature", 2.0)),
        anchor_to_init_weight=float(getattr(config, "anchor_to_init_weight", 0.0)),
        anchor_param_reference=anchor_param_reference,
        encoder_lr_multiplier=float(getattr(config, "encoder_lr_multiplier", 1.0)),
        decoder_lr_multiplier=float(getattr(config, "decoder_lr_multiplier", 1.0)),
    )
    if callbacks:
        trainer_kw["callbacks"] = callbacks

    ws_sig = inspect.signature(WhisperSeq2SeqTrainer.__init__)
    if "processing_class" in ws_sig.parameters:
        trainer_kw["processing_class"] = processor
    elif "tokenizer" in ws_sig.parameters:
        trainer_kw["tokenizer"] = processor.tokenizer

    return ContinualWhisperSeq2SeqTrainer(**trainer_kw)
