# src/models/ctc_factory.py — w2v-BERT / Hub CTC load + LoRA adaptation.
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModelForCTC

from src.models.whisper_factory import (
    is_peft_adapter_checkpoint,
    read_adapter_base_model_name,
)
from src.utils.config import ASRConfig

_DEFAULT_W2VBERT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "intermediate_dense",
    "output_dense",
]


def apply_ctc_trainable_scope(
    model: torch.nn.Module,
    config: ASRConfig,
) -> torch.nn.Module:
    """
    Full fine-tune, frozen encoder, or LoRA adapters on the acoustic encoder.

    For ``trainable_scope="lora"``, returns a PEFT-wrapped model (requires ``peft``).
    """
    scope = (getattr(config, "trainable_scope", None) or "full").strip().lower()
    if scope in ("freeze_encoder", "decoder_only"):
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        for name, p in model.named_parameters():
            if "encoder" in name and "lm_head" not in name:
                p.requires_grad = False
        return model
    if scope == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise SystemExit(
                'trainable_scope="lora" requires the `peft` package. Install with: pip install peft'
            ) from e
        targets = config.lora_target_modules
        if not targets:
            targets = list(_DEFAULT_W2VBERT_LORA_TARGETS)
        lora_config = LoraConfig(
            r=int(config.lora_r),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=list(targets),
            bias="none",
            modules_to_save=["lm_head"],
        )
        return get_peft_model(model, lora_config)
    if scope != "full":
        raise SystemExit(
            f'Unknown trainable_scope={scope!r}. Use "full", "freeze_encoder", "decoder_only", or "lora".'
        )
    return model


def resolve_ctc_pretrained_for_eval(
    model_path: str,
    text_settings: Optional[dict[str, Any]] = None,
) -> tuple[str, bool]:
    """
    Returns ``(base_or_checkpoint_path, is_lora_adapter_dir)``.

    Hub ids and full local checkpoints load from ``model_path``. LoRA dirs load
    the base from ``adapter_config.json`` or ``pretrained_model`` in eval settings.
    """
    from src.models.whisper_factory import is_huggingface_hub_model_id

    settings = text_settings or {}
    raw = str(model_path).strip()
    if is_huggingface_hub_model_id(raw):
        return raw, False
    mp = __import__("pathlib").Path(raw).expanduser().resolve()
    if not is_peft_adapter_checkpoint(mp):
        return str(mp), False
    base = read_adapter_base_model_name(mp) or str(settings.get("pretrained_model") or "").strip()
    if not base:
        raise SystemExit(
            f"LoRA CTC checkpoint at {mp} needs a base model: set pretrained_model in "
            "training_config_resolved.json or ensure adapter_config.json has base_model_name_or_path."
        )
    return base, True


def load_ctc_model_for_eval(
    model_path: str,
    *,
    text_settings: Optional[dict[str, Any]] = None,
    merge_lora: bool = True,
) -> torch.nn.Module:
    """Load Hub CTC / w2v-BERT checkpoint for inference (full weights or PEFT LoRA)."""
    from pathlib import Path

    base_path, is_adapter = resolve_ctc_pretrained_for_eval(model_path, text_settings)
    adapter_dir = str(Path(str(model_path).strip()).expanduser().resolve())

    if is_adapter:
        model = AutoModelForCTC.from_pretrained(base_path)
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit(
                "LoRA CTC checkpoint requires peft. Install with: pip install peft"
            ) from e
        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        model = AutoModelForCTC.from_pretrained(base_path)
    return model
