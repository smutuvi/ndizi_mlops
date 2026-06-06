# src/models/factory.py
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

from src.utils.config import ASRConfig


def sync_ctc_model_with_tokenizer(model: torch.nn.Module, processor: AutoProcessor | Wav2Vec2Processor) -> None:
    """Align CTC config with custom tokenizer (no BOS/EOS; pad is CTC blank)."""
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None  # type: ignore[misc, assignment]
    base = model.get_base_model() if PeftModel is not None and isinstance(model, PeftModel) else model
    cfg = base.config
    tok = processor.tokenizer
    cfg.vocab_size = len(tok)
    cfg.pad_token_id = tok.pad_token_id
    for name in ("bos_token_id", "eos_token_id"):
        if hasattr(cfg, name):
            setattr(cfg, name, None)
    gen = getattr(base, "generation_config", None)
    if gen is not None:
        gen.pad_token_id = tok.pad_token_id
        for name in ("bos_token_id", "eos_token_id"):
            if hasattr(gen, name):
                setattr(gen, name, None)


def create_asr_model(config: ASRConfig, processor: AutoProcessor | Wav2Vec2Processor) -> torch.nn.Module:
    """Load a Hub CTC checkpoint (processor vocab matches checkpoint)."""
    pretrained_model_path = config.get_pretrained_model_path()
    model = AutoModelForCTC.from_pretrained(pretrained_model_path)
    if getattr(config, "freeze_feature_encoder", False) and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    return model


def create_asr_model_for_custom_vocab(config: ASRConfig, processor: AutoProcessor | Wav2Vec2Processor) -> torch.nn.Module:
    """
    facebook/w2v-bert-2.0 (and similar) with a freshly built CTC head sized to ``len(processor.tokenizer)``.
    """
    path = config.get_pretrained_model_path()
    tok = processor.tokenizer
    model_config = AutoConfig.from_pretrained(path)
    for name, val in [
        ("attention_dropout", 0.0),
        ("hidden_dropout", 0.0),
        ("feat_proj_dropout", 0.0),
        ("mask_time_prob", 0.0),
        ("layerdrop", 0.0),
    ]:
        if hasattr(model_config, name):
            setattr(model_config, name, val)
    model_config.ctc_loss_reduction = "mean"
    model_config.ctc_zero_infinity = bool(config.ctc_zero_infinity)
    model_config.pad_token_id = tok.pad_token_id
    model_config.vocab_size = len(tok)
    if hasattr(model_config, "add_adapter"):
        model_config.add_adapter = bool(config.add_final_layer_adapter)
    model = AutoModelForCTC.from_pretrained(
        path,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    if getattr(config, "freeze_feature_encoder", False) and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    return model
