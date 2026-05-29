# src/utils/config.py — dataclass config + YAML/JSON loader (layout inspired by common ASR training repos).
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_pretrained_model_map() -> Dict[str, str]:
    return {
        "xlsr-128": "facebook/wav2vec2-xls-r-300m",
        "xlsr-53": "facebook/wav2vec2-large-xlsr-53",
        "w2v-BERT": "facebook/w2v-bert-2.0",
        "mms-300m": "facebook/mms-300m",
    }


@dataclass
class ASRConfig:
    """Training configuration for bundled CTC ASR (stack: ctc)."""

    stack: str = "ctc"
    project: str = "Ndizi-ASR"
    output_dir: str = "inprogress/ndizi-w2vbert"
    seed: int = 42

    pretrained_model: str = "facebook/wav2vec2-bert-rel-pos-large"
    freeze_feature_encoder: bool = False
    add_final_layer_adapter: bool = False
    ctc_zero_infinity: bool = True

    batch_size: int = 4
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
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

    # None = do not drop rows by duration (long clips stay in train/eval; lower batch_size / gradient_checkpointing).
    max_input_seconds: Optional[float] = 30.0
    dataset_revision: Optional[str] = None

    # Hub CTC checkpoint with bundled tokenizer/processor (recommended for Ndizi).
    use_hub_ctc_checkpoint: bool = True

    # Multi-dataset (merged train / pooled eval) — overrides single dataset_path when set.
    train_datasets: Optional[List[str]] = None
    train_splits: Optional[List[str]] = None
    train_weights: Optional[List[float]] = None
    eval_datasets: Optional[List[str]] = None
    eval_splits: Optional[List[str]] = None

    # Single Hub dataset (legacy-style)
    use_custom_dataset: bool = False
    dataset_path: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    language: str = "all"
    sample: bool = False
    sample_size: int = 1000

    # Custom CTC vocab path (when use_hub_ctc_checkpoint is False)
    add_language_tokens: bool = False
    character_set: str = (
        "abcdefghijklmnopqrstuvwxyz0123456789 -'"
    )
    apply_accent_replacements: bool = True

    pretrained_model_map: Dict[str, str] = field(default_factory=_default_pretrained_model_map)

    report_to: str = "none"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Multi-gate audio/text QC filter (ported from cleaned_ndizi_may_6.py).
    # Off by default; set true + optional "qc_*" overrides to enable.
    apply_data_qc: bool = False
    # When true with apply_data_qc, QC uses May 6-style ``__text_norm`` (see preprocessing.add_may6_text_norm_batch).
    qc_use_may6_text_norm: bool = False
    # MMS_FA word-aligned chunking for long train clips (only when apply_data_qc is true).
    qc_chunk_long_with_mms_fa: bool = False
    qc_chunk_long_with_mms_fa_eval: bool = False
    qc_chunk_seconds: float = 30.0
    qc_fa_device: str = "auto"
    # Full JSON/YAML dict for qc_* threshold overrides (populated by load_config).
    training_config_raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get_pretrained_model_path(self) -> str:
        return self.pretrained_model_map.get(self.pretrained_model, self.pretrained_model)

    def get_experiment_name(self) -> str:
        ts = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        return f"{self.get_pretrained_model_path().replace('/', '-')}-{ts}"


def _load_raw_dict(config_path: Path) -> dict[str, Any]:
    suf = config_path.suffix.lower()
    if suf == ".json":
        with open(config_path, encoding="utf-8") as f:
            return json.load(f) or {}
    if suf in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "Reading .yaml requires PyYAML (`pip install PyYAML`) or use a .json config."
            ) from e
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise ValueError(f"Unsupported config extension: {config_path.suffix}")


def read_raw_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load JSON/YAML as a plain dict (shared by CTC and Whisper config loaders)."""
    return _load_raw_dict(Path(config_path).expanduser().resolve())


def load_config(config_path: str | Path) -> ASRConfig:
    path = Path(config_path).expanduser().resolve()
    raw = _load_raw_dict(path)
    if str(raw.get("stack", "ctc")).lower() == "whisper":
        raise SystemExit(
            "This file is a Whisper training config (stack: whisper). Run:\n"
            "  python3 scripts/train_whisper.py --config "
            f"{path}\n"
            "CTC / w2v-BERT training uses scripts/train_model.py."
        )
    if "num_train_epochs" in raw and "num_epochs" not in raw:
        raw["num_epochs"] = int(float(raw["num_train_epochs"]))
    field_names = {f.name for f in fields(ASRConfig)}
    kwargs = {k: raw[k] for k in field_names if k in raw}
    kwargs["training_config_raw"] = dict(raw)
    cfg = ASRConfig(**kwargs)
    if cfg.qc_chunk_long_with_mms_fa and not cfg.apply_data_qc:
        raise SystemExit("qc_chunk_long_with_mms_fa requires apply_data_qc: true")
    if cfg.qc_use_may6_text_norm and not cfg.apply_data_qc:
        raise SystemExit("qc_use_may6_text_norm requires apply_data_qc: true")
    if cfg.qc_chunk_long_with_mms_fa_eval and not cfg.qc_chunk_long_with_mms_fa:
        raise SystemExit("qc_chunk_long_with_mms_fa_eval requires qc_chunk_long_with_mms_fa: true")
    return cfg
