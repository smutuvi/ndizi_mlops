# src/models/ctc_factory.py — w2v-BERT / Hub CTC load + LoRA adaptation.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoModelForCTC

from src.models.whisper_factory import (
    is_peft_adapter_checkpoint,
    read_adapter_base_model_name,
)
from src.utils.config import ASRConfig

logger = logging.getLogger(__name__)

_DEFAULT_W2VBERT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "intermediate_dense",
    "output_dense",
]

_CTC_LM_HEAD_FILENAME = "ctc_lm_head.bin"


def _enable_ctc_lm_head_training(model: torch.nn.Module) -> None:
    """Train CTC head alongside LoRA (not via modules_to_save — breaks w2v-BERT save)."""
    for name, param in model.named_parameters():
        if "lm_head" in name:
            param.requires_grad = True


def _lm_head_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lm_head" in k
    }


def _load_lm_head_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    if not state:
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        logger.debug("ctc_lm_head unexpected keys: %s", unexpected)
    if missing:
        logger.debug("ctc_lm_head missing keys (ok if LoRA-only): %s", missing)


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
        # Do not use modules_to_save=["lm_head"]: PEFT save calls get_input_embeddings()
        # on Wav2Vec2BertForCTC and raises NotImplementedError after vocab resize.
        lora_config = LoraConfig(
            r=int(config.lora_r),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=list(targets),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        _enable_ctc_lm_head_training(model)
        return model
    if scope != "full":
        raise SystemExit(
            f'Unknown trainable_scope={scope!r}. Use "full", "freeze_encoder", "decoder_only", or "lora".'
        )
    return model


def save_ctc_lora_checkpoint(
    model: torch.nn.Module,
    output_dir: str,
    *,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    """
    Save PEFT LoRA adapters + separate CTC ``lm_head`` weights.

    Works around Wav2Vec2BertForCTC lacking ``get_input_embeddings`` for PEFT save.
    """
    from peft import PeftModel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, PeftModel):
        model.save_pretrained(output_dir, state_dict=state_dict)
        return

    if state_dict is None:
        state_dict = model.state_dict()

    lm_head_state = _lm_head_state_dict(model)
    if lm_head_state:
        torch.save(lm_head_state, out / _CTC_LM_HEAD_FILENAME)

    try:
        model.save_pretrained(str(out), state_dict=state_dict, save_embedding_layers=False)
    except TypeError:
        from peft.utils import get_peft_model_state_dict

        peft_state = get_peft_model_state_dict(
            model, state_dict=state_dict, save_embedding_layers=False
        )
        model.peft_config["default"].save_pretrained(str(out))
        try:
            import safetensors.torch

            safetensors.torch.save_file(peft_state, out / "adapter_model.safetensors")
        except Exception:
            torch.save(peft_state, out / "adapter_model.bin")


def _load_ctc_base_for_eval(
    base_path: str,
    adapter_dir: Path,
) -> torch.nn.Module:
    """Load base CTC weights, resizing ``lm_head`` when the checkpoint has a custom vocab."""
    vocab_size: Optional[int] = None
    pad_token_id: Optional[int] = None
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(str(adapter_dir))
        tok = getattr(proc, "tokenizer", None)
        if tok is not None:
            vocab_size = len(tok)
            pad_token_id = getattr(tok, "pad_token_id", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read processor from %s for vocab resize: %s", adapter_dir, exc)

    if vocab_size is None:
        return AutoModelForCTC.from_pretrained(base_path)

    cfg = AutoConfig.from_pretrained(base_path)
    base_vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if base_vocab == vocab_size:
        return AutoModelForCTC.from_pretrained(base_path)

    cfg.vocab_size = vocab_size
    if pad_token_id is not None:
        cfg.pad_token_id = pad_token_id
    logger.info(
        "Resizing CTC head for eval: base vocab %s -> checkpoint vocab %s",
        base_vocab,
        vocab_size,
    )
    return AutoModelForCTC.from_pretrained(
        base_path,
        config=cfg,
        ignore_mismatched_sizes=True,
    )


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
    mp = Path(raw).expanduser().resolve()
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
    base_path, is_adapter = resolve_ctc_pretrained_for_eval(model_path, text_settings)
    adapter_dir = Path(str(model_path).strip()).expanduser().resolve()

    if is_adapter:
        model = _load_ctc_base_for_eval(base_path, adapter_dir)
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit(
                "LoRA CTC checkpoint requires peft. Install with: pip install peft"
            ) from e
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        lm_path = adapter_dir / _CTC_LM_HEAD_FILENAME
        if lm_path.is_file():
            try:
                lm_state = torch.load(lm_path, map_location="cpu", weights_only=True)
            except TypeError:
                lm_state = torch.load(lm_path, map_location="cpu")
            _load_lm_head_state(model, lm_state)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        model = AutoModelForCTC.from_pretrained(base_path)
    return model
