# src/models/whisper_factory.py — load Whisper model + processor for fine-tuning.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor

from src.utils.whisper_config import WhisperTrainingConfig


def cap_label_length(model: WhisperForConditionalGeneration, requested: int) -> int:
    max_pos = int(getattr(model.config, "max_target_positions", 448))
    return min(int(requested), max_pos)


def eval_generation_max_length(model: WhisperForConditionalGeneration, requested: int) -> int:
    max_pos = int(getattr(model.config, "max_target_positions", 448))
    return max(min(int(requested), max_pos), 1)


def _label_and_gen_max_from_whisper_config(
    whisper_config: WhisperConfig,
    generation_max_length: int,
) -> tuple[int, int]:
    max_pos = int(getattr(whisper_config, "max_target_positions", 448))
    label_max = min(int(generation_max_length), max_pos)
    gen_max = min(int(generation_max_length), max_pos)
    return max(label_max, 1), max(gen_max, 1)


def apply_whisper_decoder_length_cap(
    model: WhisperForConditionalGeneration,
    decoder_max_length: int,
) -> int:
    """
    Decoder ``generate`` uses **only** ``max_length`` (clears ``max_new_tokens``) so HuggingFace
    does not warn that both knobs are set (common on hub Whisper checkpoints).
    """
    max_pos = int(getattr(model.config, "max_target_positions", 448))
    cap = max(2, min(int(decoder_max_length), max_pos))
    gc = model.generation_config
    if hasattr(gc, "max_new_tokens"):
        try:
            gc.max_new_tokens = None
        except (AttributeError, TypeError):
            pass
    gc.max_length = int(cap)
    return int(cap)


def apply_whisper_generation_config(
    model: WhisperForConditionalGeneration,
    config: WhisperTrainingConfig,
    processor: WhisperProcessor,
    *,
    decoder_max_length: Optional[int] = None,
) -> None:
    model.generation_config.language = config.whisper_language
    model.generation_config.task = config.whisper_task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.whisper_language,
        task=config.whisper_task,
    )
    if decoder_max_length is not None:
        apply_whisper_decoder_length_cap(model, int(decoder_max_length))


def apply_whisper_trainable_scope(
    model: WhisperForConditionalGeneration,
    config: WhisperTrainingConfig,
) -> torch.nn.Module:
    """
    Adjust which parameters are trainable: full model, frozen encoder, or LoRA adapters.

    For ``trainable_scope="lora"``, returns a PEFT-wrapped model (requires ``peft``).
    """
    scope = (getattr(config, "trainable_scope", None) or "full").strip().lower()
    if scope in ("decoder_only", "freeze_encoder"):
        for name, p in model.named_parameters():
            if "model.encoder" in name:
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
            # Match HF peft Whisper examples (no task_type — SEQ_2_SEQ_LM breaks input_features).
            targets = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        lora_config = LoraConfig(
            r=int(config.lora_r),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=list(targets),
            bias="none",
        )
        return get_peft_model(model, lora_config)
    if scope != "full":
        raise SystemExit(
            f'Unknown trainable_scope={scope!r}. Use "full", "freeze_encoder", "decoder_only", or "lora".'
        )
    return model


def load_whisper_teacher_for_distillation(
    path: str,
    config: WhisperTrainingConfig,
    processor: WhisperProcessor,
) -> WhisperForConditionalGeneration:
    """Load a frozen copy of Whisper for KL distillation (separate forward per step when enabled)."""
    teacher = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32)
    _, gen_max = _label_and_gen_max_from_whisper_config(teacher.config, config.generation_max_length)
    apply_whisper_generation_config(teacher, config, processor, decoder_max_length=gen_max)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def snapshot_trainable_params_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU clones of parameters that require grad (for anchor loss)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = p.detach().float().cpu().clone()
    return out


def create_whisper_processor_and_limits(
    config: WhisperTrainingConfig,
) -> tuple[WhisperProcessor, int, int]:
    """
    Load only the processor + ``WhisperConfig`` (no full model weights).

    Used **before** ``Dataset.map`` so encoding does not fork after a large GPU model is
    loaded and so RAM is lower during the map phase.
    """
    path = config.get_pretrained_model_path()
    wcfg = WhisperConfig.from_pretrained(path)
    processor = WhisperProcessor.from_pretrained(
        path,
        language=config.whisper_language,
        task=config.whisper_task,
    )
    label_max, gen_max = _label_and_gen_max_from_whisper_config(wcfg, config.generation_max_length)
    return processor, label_max, gen_max


def is_huggingface_hub_model_id(path: str | Path) -> bool:
    """
    True for Hub ids like ``msingiai/sauti-asr`` (not an existing local path).

    Relative paths such as ``runs/foo/bar`` are local even if they contain ``/``.
    """
    s = str(path).strip()
    if not s or s.startswith((".", "/")) or "\\" in s:
        return False
    if len(s) > 1 and s[1] == ":":
        return False
    if Path(s).expanduser().exists():
        return False
    parts = [p for p in s.split("/") if p]
    return len(parts) == 2 and all(parts)


def is_peft_adapter_checkpoint(model_path: str | Path) -> bool:
    """True when ``model_path`` contains a PEFT LoRA adapter (not a full merged checkpoint)."""
    if is_huggingface_hub_model_id(model_path):
        return False
    p = Path(model_path).expanduser().resolve()
    return (p / "adapter_config.json").is_file()


def read_adapter_base_model_name(model_path: str | Path) -> Optional[str]:
    cfg_path = Path(model_path).expanduser().resolve() / "adapter_config.json"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    base = data.get("base_model_name_or_path")
    return str(base).strip() if base else None


def resolve_whisper_pretrained_for_eval(
    model_path: str | Path,
    text_settings: Optional[dict[str, Any]] = None,
) -> tuple[str, bool]:
    """
    Returns ``(weights_path, is_lora_adapter_dir)``.

    Full checkpoints load from ``model_path``. LoRA dirs load the base from
    ``adapter_config.json`` or ``pretrained_model`` in ``training_config_resolved.json``.
    """
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
            f"LoRA checkpoint at {mp} needs a base model: set pretrained_model in "
            "training_config_resolved.json or ensure adapter_config.json has base_model_name_or_path."
        )
    return base, True


def load_whisper_model_for_eval(
    model_path: str | Path,
    *,
    processor: WhisperProcessor,
    whisper_language: str,
    whisper_task: str,
    generation_max_length: int,
    text_settings: Optional[dict[str, Any]] = None,
    merge_lora: bool = True,
) -> WhisperForConditionalGeneration:
    """
    Load a Whisper checkpoint for inference (full fine-tune or PEFT LoRA from continual training).

    LoRA runs saved by ``train_whisper.py`` are merged into a plain
    ``WhisperForConditionalGeneration`` by default (``merge_lora=True``).
    """
    base_path, is_adapter = resolve_whisper_pretrained_for_eval(model_path, text_settings)
    adapter_dir = str(Path(str(model_path).strip()).expanduser().resolve())

    if is_adapter:
        model = WhisperForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.float32)
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit(
                "LoRA checkpoint requires peft. Install with: pip install peft"
            ) from e
        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.float32)

    gen_cap = eval_generation_max_length(model, int(generation_max_length))
    apply_whisper_decoder_length_cap(model, gen_cap)
    model.generation_config.language = whisper_language
    model.generation_config.task = whisper_task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=whisper_language,
        task=whisper_task,
    )
    return model


def load_whisper_model_for_training(
    config: WhisperTrainingConfig,
    processor: WhisperProcessor,
    pretrained_path: Optional[str] = None,
) -> WhisperForConditionalGeneration:
    """Load weights after dataset encoding (see ``create_whisper_processor_and_limits``)."""
    path = pretrained_path if pretrained_path is not None else config.get_pretrained_model_path()
    model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32)
    _, gen_max = _label_and_gen_max_from_whisper_config(model.config, config.generation_max_length)
    apply_whisper_generation_config(model, config, processor, decoder_max_length=gen_max)
    return model


def create_whisper_model_bundle(
    config: WhisperTrainingConfig,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, int, int]:
    """
    Returns ``(model, processor, label_max_length, generation_max_length_for_eval)``.

    Prefer the split API in ``train_whisper.py`` (processor + limits first, encode, then
    ``load_whisper_model_for_training``) to avoid multiprocessing stalls during ``map``.
    """
    processor, label_max, gen_max = create_whisper_processor_and_limits(config)
    model = load_whisper_model_for_training(config, processor)
    return model, processor, label_max, gen_max
