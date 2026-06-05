# src/utils/whisper_config.py — Whisper seq2seq training config (stack: whisper).
from __future__ import annotations

import datetime
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import read_raw_training_config


@dataclass
class WhisperTrainingConfig:
    """
    Training configuration for bundled Whisper fine-tuning.

    Use JSON/YAML with ``"stack": "whisper"`` and run ``scripts/train_whisper.py``.
    Dataset fields mirror ``ASRConfig`` so the same Hub manifests work for CTC and Whisper.
    """

    stack: str = "whisper"
    project: str = "Ndizi-ASR-Whisper"
    output_dir: str = "inprogress/ndizi-whisper"
    seed: int = 42

    pretrained_model: str = "openai/whisper-small"
    whisper_language: str = "sw"
    whisper_task: str = "transcribe"
    generation_max_length: int = 444
    generation_num_beams: int = 1

    batch_size: int = 2
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 2
    num_epochs: int = 10
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.0
    warmup_steps: int = 500
    weight_decay: float = 0.0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 2
    max_steps: int = -1
    group_by_length: bool = True
    eval_strategy: str = "epoch"
    normalize_wer: bool = False

    max_input_seconds: Optional[float] = 30.0
    dataset_revision: Optional[str] = None

    train_datasets: Optional[List[str]] = None
    train_splits: Optional[List[str]] = None
    train_weights: Optional[List[float]] = None
    eval_datasets: Optional[List[str]] = None
    eval_splits: Optional[List[str]] = None

    use_custom_dataset: bool = False
    dataset_path: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    language: str = "all"
    sample: bool = False
    sample_size: int = 1000

    report_to: str = "none"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Continual / ablation: encoder–decoder LRs, freezing, LoRA, KD, anchor, early stop.
    # trainable_scope: "full" | "freeze_encoder" | "decoder_only" (same as freeze_encoder) | "lora"
    trainable_scope: str = "full"
    encoder_lr_multiplier: float = 1.0
    decoder_lr_multiplier: float = 1.0
    # Optional Hub/local path for a frozen teacher; if distill_kl_weight > 0 and this is null,
    # a second load from pretrained_model is used (same init weights as the student at load time).
    teacher_model_path: Optional[str] = None
    distill_kl_weight: float = 0.0
    distill_temperature: float = 2.0
    # L2 anchor toward initial trainable weights (after LoRA); 0 disables.
    anchor_to_init_weight: float = 0.0
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    # 0 = disabled; otherwise HF EarlyStoppingCallback(patience=…)
    early_stopping_patience: int = 0

    # Multi-gate audio/text QC filter (ported from cleaned_ndizi_may_6.py).
    # Off by default; set true + optional "qc_*" overrides to enable.
    apply_data_qc: bool = False
    # When true with apply_data_qc, QC gates use May 6 ``__text_norm`` (punct/number norm);
    # training labels still use ``clean_transcription`` (minimal strip for Whisper).
    qc_use_may6_text_norm: bool = False
    qc_chunk_long_with_mms_fa: bool = False
    # When true (and train MMS_FA is on), validation is MMS_FA-chunked too for eval WER / early stopping.
    qc_chunk_long_with_mms_fa_eval: bool = False
    qc_chunk_seconds: float = 30.0
    qc_fa_device: str = "auto"
    format_transcripts: bool = True
    normalize_oral_tokens: bool = False
    use_formatting_score_for_best: bool = False
    training_config_raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get_pretrained_model_path(self) -> str:
        return self.pretrained_model

    def get_experiment_name(self) -> str:
        ts = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        return f"{self.pretrained_model.replace('/', '-')}-{ts}"


def load_whisper_training_config(config_path: str | Path) -> WhisperTrainingConfig:
    path = Path(config_path).expanduser().resolve()
    raw: Dict[str, Any] = dict(read_raw_training_config(path))
    if str(raw.get("stack", "")).lower() != "whisper":
        raise SystemExit(
            'Whisper training configs must set "stack": "whisper". '
            "For CTC / w2v-BERT, use scripts/train_model.py with a CTC config (omit stack or use stack: ctc)."
        )
    if "num_train_epochs" in raw and "num_epochs" not in raw:
        raw["num_epochs"] = int(float(raw["num_train_epochs"]))
    field_names = {f.name for f in fields(WhisperTrainingConfig)}
    kwargs = {k: raw[k] for k in field_names if k in raw}
    kwargs["training_config_raw"] = dict(raw)
    cfg = WhisperTrainingConfig(**kwargs)
    if cfg.qc_chunk_long_with_mms_fa and not cfg.apply_data_qc:
        raise SystemExit("qc_chunk_long_with_mms_fa requires apply_data_qc: true")
    if cfg.qc_use_may6_text_norm and not cfg.apply_data_qc:
        raise SystemExit("qc_use_may6_text_norm requires apply_data_qc: true")
    if cfg.qc_chunk_long_with_mms_fa_eval and not cfg.qc_chunk_long_with_mms_fa:
        raise SystemExit("qc_chunk_long_with_mms_fa_eval requires qc_chunk_long_with_mms_fa: true")
    if cfg.qc_chunk_long_with_mms_fa_eval and not cfg.apply_data_qc:
        raise SystemExit("qc_chunk_long_with_mms_fa_eval requires apply_data_qc: true")
    # Default: chunk validation when train uses MMS_FA (unless explicitly set false in JSON).
    if cfg.qc_chunk_long_with_mms_fa and "qc_chunk_long_with_mms_fa_eval" not in raw:
        cfg.qc_chunk_long_with_mms_fa_eval = True
    return cfg
