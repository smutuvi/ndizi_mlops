#!/usr/bin/env python3
"""
Batch ASR evaluation for checkpoints from this bundle (CTC or Whisper).

**CTC (default):** loads ``AutoModelForCTC`` + processor (w2v-BERT uses ``input_features``;
wav2vec2-style uses ``input_values`` — same as ``scripts/train_model.py``).

**Whisper:** pass ``--backend whisper`` or use ``--backend auto`` with a
``training_config_resolved.json`` that contains ``"stack": "whisper"`` (written by
``scripts/train_whisper.py``). Decoding uses ``model.generate`` with forced decoder ids.
**LoRA continual** checkpoints (``trainable_scope: lora``) save adapter weights plus
``training_config_resolved.json``; eval loads the base ``pretrained_model``, applies the
adapter, and merges for inference (requires ``peft``). Same for **CTC / w2v-BERT LoRA**
from ``scripts/train_model.py``.

**Audio length:** by default **no** max-duration filter (all utterances are scored),
matching common batch eval drivers that only resample to 16 kHz. To align with
training’s ``max_input_seconds`` (typically 30), pass ``--max_audio_seconds 30`` or
the alias ``--max-input-seconds 30``; use ``--no-max-input-filter`` for no cap
(same as omitting both).

**Aggressive QC:** pass ``--aggressive-qc`` to drop eval rows through the same multi-gate
pipeline as training (``src/data/qc.py`` / ``check_example``). Thresholds and
``qc_use_may6_text_norm`` come from ``--training_config`` or
``<model_path>/training_config_resolved.json`` when present; otherwise default
:class:`~src.data.qc.QCConfig` thresholds apply (log warns when no config file).

**Memory:** decoding streams features per batch (no giant precomputed feature list). If you
still hit CUDA OOM from a single very long clip in a batch, use ``--chunk_long_audio_seconds 30``
(non-overlapping chunks, predictions joined with spaces) and/or ``--fp16``, or cap length with
``--max_audio_seconds 30``. A pre-streaming snapshot lives in
``evaluate_asr_batch_backup_precompute_all_rows.py``.

**Output schema** matches ``ndizi_mlops_offshelf`` batch eval: ``predictions.json`` / ``predictions.csv``
with ``reference``, ``prediction``, timing fields, and optional ``*_normalized`` columns when
``--normalize`` is not ``none``. ``metrics.json`` uses shared ``pooled`` / ``per_set`` plus
``run_info`` for checkpoint-specific metadata. Default WER/CER use raw ``reference`` and ``prediction``.

Examples (from ``ndizi_mlops/``):

  python3 scripts/evaluate_asr_batch.py \\
    --model_path inprogress/.../facebook-w2v-bert-2.0-12052026-090000 \\
    --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \\
    --output_dir eval/run1

  # Match training clip cap (30 s) and explicit training YAML/JSON:
  python3 scripts/evaluate_asr_batch.py \\
    --model_path ... --training_config config_files/w2vbert/ndizi_w2vbert_merged_1epoch.json \\
    --max_audio_seconds 30 \\
    --test_datasets smutuvi/ndizi-1:test --output_dir eval/run2

  # Whisper checkpoint (auto picks whisper if resolved config has stack: whisper):
  python3 scripts/evaluate_asr_batch.py \\
    --model_path inprogress/.../openai-whisper-small-... \\
    --backend auto --test_datasets smutuvi/ndizi-1:test --output_dir eval/whisper1

  # Hub baseline (e.g. msingiai/sauti-asr — pass Hub id, not a local path):
  python3 scripts/evaluate_asr_batch.py \\
    --model_path msingiai/sauti-asr --backend whisper \\
    --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test --output_dir eval/sauti_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")


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


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def resolve_columns(column_names: List[str]) -> Tuple[str, str]:
    audio_col = pick_col(column_names, ["audio", "audio_path", "path", "file", "wav", "speech"])
    text_col = pick_col(
        column_names,
        ["text", "transcript", "sentence", "transcription", "normalized_text"],
    )
    if audio_col is None:
        raise ValueError(f"No audio column in {column_names}")
    if text_col is None:
        raise ValueError(f"No text column in {column_names}")
    return audio_col, text_col


def revision_kw(rev: Optional[str]) -> Dict[str, Any]:
    return {"revision": rev.strip()} if rev else {}


@dataclass
class SplitSpec:
    dataset_id: str
    split: str

    @classmethod
    def parse(cls, raw: str) -> SplitSpec:
        if ":" not in raw:
            return cls(raw, "test")
        ds, sp = raw.split(":", 1)
        return cls(ds, sp)


def probe_model_input_name(processor: Any) -> str:
    import numpy as np

    dummy = np.zeros(16000, dtype=np.float32)
    out = processor(dummy, sampling_rate=16000)
    if isinstance(out, dict):
        for k in ("input_features", "input_values"):
            if k in out:
                return k
        raise RuntimeError(f"Processor keys not usable: {list(out.keys())}")
    for k in ("input_features", "input_values"):
        if hasattr(out, k):
            return k
    raise RuntimeError("Processor output has no input_features / input_values")


def wer_normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def resolve_max_audio_seconds_for_eval(args: argparse.Namespace) -> Optional[float]:
    """``--no-max-input-filter`` wins; else ``--max-input-seconds``; else ``--max_audio_seconds``."""
    if getattr(args, "no_max_audio_cap", False):
        return None
    if getattr(args, "max_input_seconds", None) is not None:
        return float(args.max_input_seconds)
    return getattr(args, "max_audio_seconds", None)


def resolve_aggressive_qc_bundle(
    aggressive_qc: bool,
    text_settings: Dict[str, Any],
    log: Optional[logging.Logger] = None,
) -> Optional[Tuple[Any, bool]]:
    """
    When ``aggressive_qc`` is true, return ``(QCConfig, qc_use_may6_text_norm)`` for eval filtering.
    Thresholds are merged from ``text_settings['training_config_raw']`` when set, else defaults.
    """
    if not aggressive_qc:
        return None
    from src.data.qc import QCConfig, qc_config_from_training_json

    raw = text_settings.get("training_config_raw")
    if not raw:
        if log:
            log.warning(
                "Aggressive QC enabled but no training config JSON was loaded; using default "
                "QCConfig thresholds and qc_use_may6_text_norm=false. Pass --training_config or "
                "keep training_config_resolved.json next to the checkpoint to mirror training."
            )
        return (QCConfig(), False)
    cfg = qc_config_from_training_json(raw)
    use_may6 = bool(raw.get("qc_use_may6_text_norm", False))
    return (cfg, use_may6)


def try_build_jiwer_transforms() -> Tuple[Any, Any]:
    try:
        import jiwer
    except ImportError as e:
        raise SystemExit(
            'text_normalize="jiwer_default" requires jiwer. Install with: python3 -m pip install jiwer'
        ) from e
    tr_w = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    tr_c = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )
    return tr_w, tr_c


def pooled_wer_cer(
    preds: List[str],
    refs: List[str],
    mode: str,
    wer_m: Any,
    cer_m: Any,
    jiwer_tr_w: Any,
    jiwer_tr_c: Any,
) -> Tuple[Optional[float], Optional[float]]:
    pairs = [(p, r) for p, r in zip(preds, refs) if str(r).strip()]
    if not pairs:
        return None, None
    p2, r2 = zip(*pairs)
    pl, rl = list(p2), list(r2)
    if mode == "none":
        return (
            float(wer_m.compute(predictions=pl, references=rl)),
            float(cer_m.compute(predictions=pl, references=rl)),
        )
    if mode == "simple":
        pl2 = [wer_normalize(p) for p in pl]
        rl2 = [wer_normalize(r) for r in rl]
        return (
            float(wer_m.compute(predictions=pl2, references=rl2)),
            float(cer_m.compute(predictions=pl2, references=rl2)),
        )
    import jiwer

    return (
        float(jiwer.wer(rl, pl, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w)),
        float(jiwer.cer(rl, pl, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c)),
    )


def compute_split_quality_metrics(
    pred_raw: List[str],
    ref_raw_col: List[str],
    text_mode: str,
    wer_m: Any,
    cer_m: Any,
    jiwer_tr_w: Any,
    jiwer_tr_c: Any,
) -> Dict[str, Any]:
    """WER/CER (raw, no jiwer strip) plus optional normalized and punctuation recall."""
    from src.data.text_format import mean_punctuation_recall

    wer_v, cer_v = pooled_wer_cer(
        list(pred_raw),
        list(ref_raw_col),
        "none",
        wer_m,
        cer_m,
        jiwer_tr_w,
        jiwer_tr_c,
    )
    out: Dict[str, Any] = {
        "wer": wer_v,
        "cer": cer_v,
        "wer_raw": wer_v,
        "cer_raw": cer_v,
        "punct_recall": mean_punctuation_recall(list(ref_raw_col), list(pred_raw)),
    }
    if text_mode != "none":
        wn, cn = pooled_wer_cer(
            list(pred_raw),
            list(ref_raw_col),
            text_mode,
            wer_m,
            cer_m,
            jiwer_tr_w,
            jiwer_tr_c,
        )
        out["wer_normalized"] = wn
        out["cer_normalized"] = cn
        if text_mode == "jiwer_default":
            out["wer_jiwer"] = wn
            out["cer_jiwer"] = cn
    return out


def utterance_wer_cer(
    ref: str,
    hyp: str,
    mode: str,
    wer_m: Any,
    cer_m: Any,
    jiwer_tr_w: Any,
    jiwer_tr_c: Any,
) -> Tuple[Optional[float], Optional[float]]:
    if not str(ref).strip():
        return None, None
    if mode == "none":
        return (
            float(wer_m.compute(predictions=[hyp], references=[ref])),
            float(cer_m.compute(predictions=[hyp], references=[ref])),
        )
    if mode == "simple":
        r2, p2 = wer_normalize(ref), wer_normalize(hyp)
        return (
            float(wer_m.compute(predictions=[p2], references=[r2])),
            float(cer_m.compute(predictions=[p2], references=[r2])),
        )
    import jiwer

    wo = jiwer.process_words(ref, hyp, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w)
    co = jiwer.process_characters(ref, hyp, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c)
    return float(wo.wer), float(co.cer)


def extra_normalized_fields_for_row(
    ref_raw: str,
    pred: str,
    mode: str,
    wer_m: Any,
    cer_m: Any,
    jiwer_tr_w: Any,
    jiwer_tr_c: Any,
) -> Dict[str, Any]:
    """Extra JSON/CSV fields when ``--normalize`` is set (``reference`` / ``prediction`` stay raw)."""
    if mode == "none":
        return {}
    if mode == "simple":
        wn, cn = utterance_wer_cer(ref_raw, pred, mode, wer_m, cer_m, jiwer_tr_w, jiwer_tr_c)
        return {
            "text_normalized": wer_normalize(ref_raw),
            "prediction_normalized": wer_normalize(pred),
            "wer_normalized": wn,
            "cer_normalized": cn,
        }
    import jiwer

    wo = jiwer.process_words(
        ref_raw, pred, reference_transform=jiwer_tr_w, hypothesis_transform=jiwer_tr_w
    )
    co = jiwer.process_characters(
        ref_raw, pred, reference_transform=jiwer_tr_c, hypothesis_transform=jiwer_tr_c
    )
    rn = " ".join(wo.references[0]) if wo.references and wo.references[0] else ""
    pn = " ".join(wo.hypotheses[0]) if wo.hypotheses and wo.hypotheses[0] else ""
    return {
        "text_normalized": rn,
        "prediction_normalized": pn,
        "wer_normalized": float(wo.wer),
        "cer_normalized": float(co.cer),
    }


def rtfx_from_times(audio_duration_s: float, decode_wall_s: float) -> Optional[float]:
    """RTFx = audio_duration / decode_wall (higher ⇒ faster than real time). Undefined if decode_wall ≤ 0."""
    if decode_wall_s is None or decode_wall_s <= 0.0:
        return None
    return float(audio_duration_s) / float(decode_wall_s)


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.8f}".rstrip("0").rstrip(".")
    return str(v)


def _write_predictions_csv(path: Path, rows: List[Dict[str, Any]], text_mode: str) -> None:
    """Tabular mirror of ``predictions.json`` (UTF-8 CSV)."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    norm = text_mode != "none"
    fieldnames = [
        "dataset",
        "split",
        "row_idx",
        "audio_path",
        "reference",
        "prediction",
        "wer",
        "cer",
        "decode_wall_s",
        "rtfx",
    ]
    if norm:
        fieldnames += [
            "text_normalized",
            "prediction_normalized",
            "wer_normalized",
            "cer_normalized",
            "rtfx_normalized",
        ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})


def resolve_processor_path(model_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    from src.models.whisper_factory import is_huggingface_hub_model_id

    if is_huggingface_hub_model_id(model_path):
        return str(model_path).strip()
    p = Path(model_path).expanduser().resolve()
    if (p / "tokenizer_config.json").is_file() or (p / "preprocessor_config.json").is_file():
        return str(p)
    tok = p / "ctc_tokenizer"
    if tok.is_dir() and (tok / "vocab.json").is_file():
        if (p / "preprocessor_config.json").is_file():
            return str(p)
    for cand in [p] + list(p.parents):
        t = cand / "ctc_tokenizer"
        if t.is_dir() and (t / "vocab.json").is_file():
            if (cand / "preprocessor_config.json").is_file():
                return str(cand)
    return str(p)


@dataclass
class RowCollator:
    processor: Any
    model_input_name: str

    def __call__(self, features: List[dict]) -> dict:
        feats = [{self.model_input_name: f[self.model_input_name]} for f in features]
        return self.processor.pad(feats, padding=True, return_tensors="pt")


def decode_pred_ids(processor: Any, pred_ids: Any) -> List[str]:
    """Greedy CTC token ids → text (same as ``ASRMetrics``: ``batch_decode(pred_ids)``)."""
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok.batch_decode(pred_ids)
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(pred_ids)
    raise RuntimeError("Processor cannot decode token ids")


def reference_for_wer_like_training(processor: Any, clean_transcription: str) -> str:
    """
    Match ``src/training/metrics.ASRMetrics``: label side uses
    ``tokenizer.batch_decode(..., group_tokens=False)`` on **input_ids**, i.e.
    round-trip encode then decode.
    """
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return str(clean_transcription or "")
    ids = tok(str(clean_transcription or ""), add_special_tokens=False).input_ids
    if not ids:
        return ""
    try:
        return tok.decode(ids, group_tokens=False)
    except TypeError:
        return tok.decode(ids)


def clean_transcription_like_training(
    raw: str,
    *,
    use_hub_ctc_checkpoint: bool,
    character_set: str,
    apply_accent_replacements: bool,
    lowercase_ctc_labels: bool = True,
) -> str:
    """Same cleaning as ``src/data/dataset.py`` in ``load_datasets`` before encoding."""
    from src.data.preprocessing import clean_text_batch, hub_ctc_identity_clean_batch

    batch = {"transcription": [str(raw or "")]}
    if use_hub_ctc_checkpoint:
        return hub_ctc_identity_clean_batch(batch)["clean_transcription"][0]
    return clean_text_batch(
        batch,
        character_set,
        apply_accent_replacements,
        lowercase=lowercase_ctc_labels,
    )["clean_transcription"][0]


def eval_reference_like_training(raw: str, text_settings: Dict[str, Any]) -> str:
    """Format + clean references the same way as training labels."""
    from src.data.text_format import format_transcript

    fmt = bool(text_settings.get("format_transcripts", True))
    s = str(raw or "")
    if fmt:
        s = format_transcript(
            s,
            normalize_oral=bool(text_settings.get("normalize_oral_tokens", False)),
            discourse_commas=bool(text_settings.get("enrich_discourse_punctuation", False)),
        )
    return clean_transcription_like_training(
        s,
        use_hub_ctc_checkpoint=bool(text_settings.get("use_hub_ctc_checkpoint", True)),
        character_set=str(text_settings.get("character_set", "")),
        apply_accent_replacements=bool(text_settings.get("apply_accent_replacements", True)),
        lowercase_ctc_labels=bool(text_settings.get("lowercase_ctc_labels", True)),
    )


def load_eval_text_settings(
    model_path: str,
    training_config: Optional[str],
    ignore_resolved_training_config: bool,
) -> Dict[str, Any]:
    """
    Load text/reference settings from ``--training_config`` and/or
    ``<model_path>/training_config_resolved.json``. Whisper configs (``stack: whisper``)
    are detected via raw JSON/YAML so ``load_config`` (CTC-only) is not used for them.
    """
    from src.data.preprocessing import DEFAULT_CTC_CHARACTER_SET

    defaults: Dict[str, Any] = {
        "stack": "ctc",
        "use_hub_ctc_checkpoint": False,
        "character_set": DEFAULT_CTC_CHARACTER_SET,
        "apply_accent_replacements": True,
        "format_transcripts": True,
        "normalize_oral_tokens": False,
        "enrich_discourse_punctuation": False,
        "lowercase_ctc_labels": True,
        "config_path": None,
        "whisper_language": "sw",
        "whisper_task": "transcribe",
        "training_config_raw": None,
    }
    from src.utils.config import load_config, read_raw_training_config

    candidates: List[Path] = []
    if training_config:
        candidates.append(Path(training_config).expanduser().resolve())
    if not ignore_resolved_training_config:
        candidates.append(Path(model_path).resolve() / "training_config_resolved.json")

    for p in candidates:
        if p.is_file():
            raw = read_raw_training_config(p)
            if str(raw.get("stack", "ctc")).lower() == "whisper":
                return {
                    "stack": "whisper",
                    "whisper_language": str(raw.get("whisper_language", "sw")),
                    "whisper_task": str(raw.get("whisper_task", "transcribe")),
                    "generation_max_length": int(raw.get("generation_max_length", 444)),
                    "trainable_scope": str(raw.get("trainable_scope", "full")),
                    "pretrained_model": str(raw.get("pretrained_model", "")),
                    "config_path": str(p),
                    "use_hub_ctc_checkpoint": True,
                    "character_set": defaults["character_set"],
                    "apply_accent_replacements": True,
                    "format_transcripts": bool(raw.get("format_transcripts", True)),
                    "normalize_oral_tokens": bool(raw.get("normalize_oral_tokens", False)),
                    "enrich_discourse_punctuation": bool(raw.get("enrich_discourse_punctuation", False)),
                    "lowercase_ctc_labels": True,
                    "training_config_raw": dict(raw),
                }
            cfg = load_config(p)
            return {
                "stack": "ctc",
                "use_hub_ctc_checkpoint": bool(cfg.use_hub_ctc_checkpoint),
                "character_set": str(cfg.character_set),
                "apply_accent_replacements": bool(cfg.apply_accent_replacements),
                "format_transcripts": bool(cfg.format_transcripts),
                "normalize_oral_tokens": bool(cfg.normalize_oral_tokens),
                "enrich_discourse_punctuation": bool(cfg.enrich_discourse_punctuation),
                "lowercase_ctc_labels": bool(cfg.lowercase_ctc_labels),
                "pretrained_model": str(cfg.pretrained_model or raw.get("pretrained_model", "")),
                "trainable_scope": str(getattr(cfg, "trainable_scope", "full") or "full"),
                "config_path": str(p),
                "whisper_language": defaults["whisper_language"],
                "whisper_task": defaults["whisper_task"],
                "training_config_raw": dict(cfg.training_config_raw),
            }
    return defaults


def _classify_hub_config_dict(cfg: Dict[str, Any]) -> Optional[str]:
    """Return ``whisper`` or ``ctc`` from a HuggingFace ``config.json`` object."""
    mt = str(cfg.get("model_type") or "").lower()
    arch_list = cfg.get("architectures") or []
    arch_s = " ".join(str(a) for a in arch_list).lower()
    if mt == "whisper" or "whisper" in arch_s:
        return "whisper"
    compact = arch_s.replace("_", "")
    if "forctc" in compact or mt in (
        "wav2vec2",
        "wav2vec2_bert",
        "wav2vec2-conformer",
        "wavlm",
        "hubert",
        "data2vec-audio",
        "unispeech",
        "unispeech-sat",
        "sew",
        "sew-d",
    ):
        return "ctc"
    return None


def _infer_backend_from_adapter_dir(root: Path) -> Optional[str]:
    """
    PEFT LoRA dirs may be Whisper or CTC/w2v-BERT — do not assume Whisper.
    """
    resolved = root / "training_config_resolved.json"
    if resolved.is_file():
        try:
            with open(resolved, encoding="utf-8") as f:
                raw = json.load(f)
            if str(raw.get("stack") or "ctc").lower() == "whisper":
                return "whisper"
            return "ctc"
        except (OSError, json.JSONDecodeError):
            pass

    cfg_path = root / "config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            kind = _classify_hub_config_dict(cfg)
            if kind:
                return kind
        except (OSError, json.JSONDecodeError):
            pass

    adapter_path = root / "adapter_config.json"
    if adapter_path.is_file():
        try:
            with open(adapter_path, encoding="utf-8") as f:
                ac = json.load(f)
            base = str(ac.get("base_model_name_or_path") or "").lower()
            if "whisper" in base or "sauti" in base:
                return "whisper"
            if any(tok in base for tok in ("w2v", "wav2vec", "badrex", "hubert", "wavlm", "ctc")):
                return "ctc"
        except (OSError, json.JSONDecodeError):
            pass
    return None


def infer_decode_backend_from_checkpoint(model_path: str) -> Optional[str]:
    """
    Read ``<model_path>/config.json`` and return ``\"whisper\"`` or ``\"ctc\"`` when unambiguous,
    else ``None``. Used so ``--backend auto`` matches the saved weights even if
    ``training_config_resolved.json`` is missing or wrong.
    """
    from src.models.whisper_factory import is_huggingface_hub_model_id

    raw = str(model_path).strip()
    if is_huggingface_hub_model_id(raw):
        low = raw.lower()
        if "whisper" in low or "sauti" in low:
            return "whisper"
        try:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(raw, "config.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            mt = str(cfg.get("model_type") or "").lower()
            arch_s = " ".join(str(a) for a in (cfg.get("architectures") or [])).lower()
            if mt == "whisper" or "whisper" in arch_s:
                return "whisper"
        except Exception:
            pass
        return None

    root = Path(raw).expanduser().resolve()
    if (root / "adapter_config.json").is_file():
        return _infer_backend_from_adapter_dir(root)
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _classify_hub_config_dict(cfg)


def build_eval_meta(
    ds,
    audio_col: str,
    text_col: str,
    max_audio_seconds: Optional[float],
    path_labels: Optional[List[str]] = None,
    *,
    qc_bundle: Optional[Tuple[Any, bool]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[dict], int, int]:
    """Row metadata only (no processor); features are built per batch during decode.

    Returns ``(rows, dropped_long, dropped_qc)``. When ``qc_bundle`` is ``(qc_cfg, use_may6)``,
    rows failing :func:`~src.data.qc.check_example` are skipped (training-style aggressive QC).
    """
    from collections import Counter

    import numpy as np

    from src.data.eval_paths import extract_audio_path_label

    rows: List[dict] = []
    dropped = 0
    dropped_qc = 0
    qc_cfg: Any = None
    qc_use_may6 = False
    if qc_bundle is not None:
        qc_cfg, qc_use_may6 = qc_bundle
    qc_reasons: Counter[str] = Counter()

    if qc_cfg is not None:
        from src.data.preprocessing import normalize_text_may6
        from src.data.qc import check_example

    for i in range(len(ds)):
        ex = ds[i]
        audio = ex[audio_col]
        arr = audio["array"]
        sr = int(audio["sampling_rate"])
        dur = float(len(arr) / max(sr, 1))
        if max_audio_seconds is not None and dur > float(max_audio_seconds):
            dropped += 1
            continue
        ref = str(ex.get(text_col) or "")
        if qc_cfg is not None:
            text_for_qc = normalize_text_may6(ref) if qc_use_may6 else str(ref).strip()
            keep, reason = check_example(np.asarray(arr, dtype=np.float32), sr, text_for_qc, qc_cfg)
            qc_reasons[reason] += 1
            if not keep:
                dropped_qc += 1
                continue
        if path_labels is not None and i < len(path_labels):
            audio_path = path_labels[i]
        else:
            audio_path = extract_audio_path_label(ex, audio_col, i)
        rows.append(
            {
                "row_idx": i,
                "reference": ref,
                "audio_duration_s": dur,
                "audio_path": audio_path,
            }
        )

    if qc_cfg is not None and log is not None and len(ds) > 0:
        n_pass_duration = len(rows) + dropped_qc
        log.info(
            "Aggressive QC: kept %d / %d rows after max-duration filter (dropped_long=%d, dropped_qc=%d; "
            "dataset_rows=%d)",
            len(rows),
            n_pass_duration,
            dropped,
            dropped_qc,
            len(ds),
        )
        log.info(
            "  QC first-fail histogram (duration-passing rows only): %s",
            dict(sorted(qc_reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        )

    return rows, dropped, dropped_qc


def _processor_feat_row(processor: Any, model_input_name: str, arr: Any, sr: int) -> dict:
    out = processor(arr, sampling_rate=sr)
    if isinstance(out, dict):
        feat = out[model_input_name][0]
    else:
        feat = getattr(out, model_input_name)[0]
    return {model_input_name: feat}


def _numpy_float_waveform(x: Any) -> Any:
    import numpy as np

    a = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(a)


def _chunk_waveform_for_decode(arr: Any, sr: int, chunk_seconds: float) -> List[Any]:
    """
    Split ``arr`` into fixed-size chunks for long-utterance decode.

    A very short **final** slice (common when ``len(arr)`` is not a multiple of the chunk)
    yields empty or degenerate log-mel frames and the w2v-BERT encoder fails with
    ``Kernel size can't be greater than actual input size``. We merge such tails into the
    previous chunk and/or pad to a minimum length so the feature extractor always sees
    enough samples.
    """
    import numpy as np

    wav = _numpy_float_waveform(arr)
    if wav.size == 0:
        return []

    chunk_samples = max(1, int(float(chunk_seconds) * sr))
    # ~250 ms at native rate; w2v-BERT front-end + stride stacks need non-trivial length.
    min_standalone = max(4000, int(0.25 * max(sr, 1)))

    segs: List[Any] = []
    for off in range(0, wav.size, chunk_samples):
        segs.append(np.copy(wav[off : off + chunk_samples]))
    while len(segs) >= 2 and segs[-1].size < min_standalone:
        segs[-2] = np.concatenate([segs[-2], segs[-1]])
        segs.pop()

    out: List[Any] = []
    for s in segs:
        if s.size == 0:
            continue
        if s.size < min_standalone:
            s = np.pad(s, (0, min_standalone - int(s.size)), mode="constant")
        out.append(np.ascontiguousarray(s, dtype=np.float32))
    return out


def transcribe_batches_streaming(
    ds,
    audio_col: str,
    model: Any,
    processor: Any,
    model_input_name: str,
    rows_meta: List[dict],
    device: Any,
    batch_size: int,
    *,
    chunk_long_audio_seconds: Optional[float],
    fp16: bool,
    cuda_empty_cache: bool,
) -> Tuple[List[str], List[str], List[float]]:
    """Load audio from ``ds`` per batch so we never store all log-mel / features in RAM.

    Returns ``(predictions, references, decode_wall_seconds_per_utterance)``. Batch time is split
    evenly across utterances in the batch; chunked clips sum per-chunk decode times.
    """
    import time

    import torch
    from tqdm import tqdm

    collator = RowCollator(processor=processor, model_input_name=model_input_name)
    model.eval()
    use_amp = bool(fp16 and device.type == "cuda")
    decode_times: List[float] = []

    def forward_logits(feat_rows: List[dict]) -> Any:
        batch = collator(feat_rows)
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**batch).logits
            else:
                logits = model(**batch).logits
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        del logits, batch
        if cuda_empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()
        return pred_ids

    def decode_one_segment(arr: Any, sr: int) -> Tuple[str, float]:
        import numpy as np

        if np.asarray(arr).size == 0:
            return "", 0.0
        t0 = time.perf_counter()
        fr = [_processor_feat_row(processor, model_input_name, arr, sr)]
        pred_ids = forward_logits(fr)
        t1 = time.perf_counter()
        return decode_pred_ids(processor, pred_ids)[0], float(t1 - t0)

    preds: List[str] = []
    refs: List[str] = []
    pending_feats: List[dict] = []
    pending_refs: List[str] = []

    def flush_pending() -> None:
        nonlocal pending_feats, pending_refs
        if not pending_feats:
            return
        t0 = time.perf_counter()
        pred_ids = forward_logits(pending_feats)
        t1 = time.perf_counter()
        n = max(len(pending_feats), 1)
        dt_each = float(t1 - t0) / float(n)
        decode_times.extend([dt_each] * len(pending_feats))
        preds.extend(decode_pred_ids(processor, pred_ids))
        refs.extend(pending_refs)
        pending_feats = []
        pending_refs = []

    chunk_s = chunk_long_audio_seconds
    pbar = tqdm(rows_meta, desc="decode")
    for m in pbar:
        i = int(m["row_idx"])
        ref = str(m["reference"])
        dur = float(m["audio_duration_s"])
        ex = ds[i]
        audio = ex[audio_col]
        arr = audio["array"]
        sr = int(audio["sampling_rate"])

        if chunk_s is not None and dur > float(chunk_s):
            flush_pending()
            parts: List[str] = []
            total_decode = 0.0
            for seg in _chunk_waveform_for_decode(arr, sr, float(chunk_s)):
                seg_txt, seg_dt = decode_one_segment(seg, sr)
                parts.append(seg_txt)
                total_decode += seg_dt
            from src.data.text_format import join_chunk_predictions

            preds.append(join_chunk_predictions(parts))
            refs.append(ref)
            decode_times.append(total_decode)
            continue

        pending_feats.append(_processor_feat_row(processor, model_input_name, arr, sr))
        pending_refs.append(ref)
        if len(pending_feats) >= batch_size:
            flush_pending()

    flush_pending()
    return preds, refs, decode_times


@dataclass
class WhisperRowCollator:
    processor: Any

    def __call__(self, features: List[dict]) -> dict:
        feats = [{"input_features": f["input_features"]} for f in features]
        return self.processor.feature_extractor.pad(feats, return_tensors="pt")


def _whisper_feat_row(processor: Any, arr: Any, sr: int) -> dict:
    import numpy as np

    wav = np.asarray(arr, dtype=np.float32).reshape(-1)
    out = processor(wav, sampling_rate=int(sr), return_tensors="pt")
    feat = out.input_features[0]
    return {"input_features": feat}


def transcribe_whisper_batches_streaming(
    ds,
    audio_col: str,
    model: Any,
    processor: Any,
    rows_meta: List[dict],
    device: Any,
    batch_size: int,
    *,
    forced_decoder_ids: Any,
    decoder_max_length: int,
    chunk_long_audio_seconds: Optional[float],
    fp16: bool,
    cuda_empty_cache: bool,
) -> Tuple[List[str], List[str], List[float]]:
    """Whisper ``generate`` decoding with batched log-mel inputs (streaming over rows).

    Returns ``(predictions, references, decode_wall_seconds_per_utterance)``.
    """
    import time

    import torch
    from tqdm import tqdm

    collator = WhisperRowCollator(processor=processor)
    model.eval()
    use_amp = bool(fp16 and device.type == "cuda")
    decode_times: List[float] = []

    def forward_generate(feat_rows: List[dict]) -> List[str]:
        batch = collator(feat_rows)
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        gen_kw: Dict[str, Any] = dict(
            forced_decoder_ids=forced_decoder_ids,
            max_length=int(decoder_max_length),
        )
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    tok_ids = model.generate(**batch, **gen_kw)
            else:
                tok_ids = model.generate(**batch, **gen_kw)
        texts = processor.tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
        del batch, tok_ids
        if cuda_empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()
        return texts

    preds: List[str] = []
    refs: List[str] = []
    pending_feats: List[dict] = []
    pending_refs: List[str] = []

    def flush_pending() -> None:
        nonlocal pending_feats, pending_refs
        if not pending_feats:
            return
        t0 = time.perf_counter()
        texts = forward_generate(pending_feats)
        t1 = time.perf_counter()
        n = max(len(pending_feats), 1)
        dt_each = float(t1 - t0) / float(n)
        decode_times.extend([dt_each] * len(pending_feats))
        preds.extend(texts)
        refs.extend(pending_refs)
        pending_feats = []
        pending_refs = []

    def decode_one_segment(arr: Any, sr: int) -> Tuple[str, float]:
        import numpy as np

        if np.asarray(arr).size == 0:
            return "", 0.0
        t0 = time.perf_counter()
        fr = [_whisper_feat_row(processor, arr, sr)]
        texts = forward_generate(fr)
        t1 = time.perf_counter()
        return texts[0], float(t1 - t0)

    chunk_s = chunk_long_audio_seconds
    pbar = tqdm(rows_meta, desc="decode (Whisper)")
    for m in pbar:
        i = int(m["row_idx"])
        ref = str(m["reference"])
        dur = float(m["audio_duration_s"])
        ex = ds[i]
        audio = ex[audio_col]
        arr = audio["array"]
        sr = int(audio["sampling_rate"])

        if chunk_s is not None and dur > float(chunk_s):
            flush_pending()
            parts: List[str] = []
            total_decode = 0.0
            for seg in _chunk_waveform_for_decode(arr, sr, float(chunk_s)):
                seg_txt, seg_dt = decode_one_segment(seg, sr)
                parts.append(seg_txt)
                total_decode += seg_dt
            from src.data.text_format import join_chunk_predictions

            preds.append(join_chunk_predictions(parts))
            refs.append(ref)
            decode_times.append(total_decode)
            continue

        pending_feats.append(_whisper_feat_row(processor, arr, sr))
        pending_refs.append(ref)
        if len(pending_feats) >= batch_size:
            flush_pending()

    flush_pending()
    return preds, refs, decode_times


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ASR evaluation on Hub splits (CTC w2v-BERT/wav2vec or Whisper).",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Saved checkpoint dir or Hub model id.")
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Override processor directory (default: infer from checkpoint layout).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "ctc", "whisper"],
        default="auto",
        help='Decode path: "auto" uses training_config stack when present (Whisper vs CTC), else CTC.',
    )
    parser.add_argument(
        "--whisper_language",
        type=str,
        default=None,
        help="Whisper forced-decoder language id (default: from training config or sw).",
    )
    parser.add_argument(
        "--whisper_task",
        type=str,
        default=None,
        help='Whisper task for forced decoder ids: "transcribe" or "translate" (default from config or transcribe).',
    )
    parser.add_argument(
        "--test_datasets",
        nargs="+",
        required=True,
        help='Hub splits, e.g. smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test (default split is "test").',
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=None,
        help="If set, skip clips longer than this many seconds (use 30 to match training "
        "``max_input_seconds``). Default: no cap (score all lengths).",
    )
    parser.add_argument(
        "--max-input-seconds",
        type=float,
        default=None,
        dest="max_input_seconds",
        metavar="SEC",
        help="Alias for --max_audio_seconds (same name as train_whisper.py).",
    )
    parser.add_argument(
        "--no-max-input-filter",
        action="store_true",
        dest="no_max_audio_cap",
        help="Do not skip clips by duration (same as omitting --max_audio_seconds).",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default=None,
        help="Training JSON/YAML (e.g. the config you passed to train_model.py). "
        "If omitted, uses <model_path>/training_config_resolved.json when present.",
    )
    parser.add_argument(
        "--ignore_resolved_training_config",
        action="store_true",
        help="Do not load <model_path>/training_config_resolved.json (still honors --training_config if set).",
    )
    parser.add_argument(
        "--aggressive-qc",
        action="store_true",
        help="Drop eval rows that fail the training multi-gate QC (src/data/qc.py). "
        "qc_* thresholds and qc_use_may6_text_norm are read from --training_config or "
        "training_config_resolved.json when present; otherwise defaults apply (see log warning).",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Cap utterances per split (debug).")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--audio_column", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "simple", "jiwer_default"],
        default="none",
        dest="text_normalize",
        help="Reference/hypothesis preprocessing: none (default; raw text and prediction for "
        "wer/cer), simple (lowercase + collapse whitespace), or jiwer_default (jiwer Compose; "
        "requires pip install jiwer). When not none, also emits *_normalized columns.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--chunk_long_audio_seconds",
        type=float,
        default=None,
        help="If set, clips longer than this are decoded in non-overlapping chunks (preds joined with spaces). "
        "Very short final slices are merged into the previous chunk or zero-padded so the log-mel frontend "
        "never sees an empty frame sequence. Typical: 30.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use torch.autocast(float16) on CUDA for smaller activation footprint.",
    )
    parser.add_argument(
        "--cuda_empty_cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each forward (slower; can help fragmentation).",
    )
    parser.add_argument(
        "--no-format-decode",
        action="store_true",
        help="Skip post-decode transcript formatting (spacing after .?!, etc.).",
    )
    parser.add_argument(
        "--discourse-commas",
        action="store_true",
        help="Insert commas before common Swahili discourse markers in decode output (lakini, vile vile, …).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("evaluate_asr_batch")

    script_dir = Path(__file__).resolve().parent
    bundle_root = script_dir.parent
    sys.path.insert(0, str(bundle_root))
    load_env_file(bundle_root / ".env")

    import torch
    import evaluate
    from datasets import Audio, load_dataset
    from huggingface_hub import login as hf_login

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if tok:
        hf_login(token=tok)

    text_settings = load_eval_text_settings(
        args.model_path,
        args.training_config,
        bool(args.ignore_resolved_training_config),
    )
    if text_settings.get("config_path"):
        log.info("Training settings from %s (stack=%s)", text_settings["config_path"], text_settings.get("stack"))
    else:
        log.info(
            "No training config found; using CTC defaults for text. "
            "Pass --training_config or keep training_config_resolved.json next to the checkpoint."
        )

    qc_bundle = resolve_aggressive_qc_bundle(bool(args.aggressive_qc), text_settings, log)
    if qc_bundle is not None:
        log.info("Aggressive QC on (qc_use_may6_text_norm=%s)", qc_bundle[1])

    if args.backend == "auto":
        decode_backend = "whisper" if text_settings.get("stack") == "whisper" else "ctc"
    else:
        decode_backend = args.backend

    ck_kind = infer_decode_backend_from_checkpoint(args.model_path)
    if ck_kind == "whisper":
        if decode_backend != "whisper":
            log.warning(
                "Checkpoint config.json is Whisper but decode backend was %s; using whisper.",
                decode_backend,
            )
        decode_backend = "whisper"
    elif ck_kind == "ctc":
        if decode_backend != "ctc":
            log.warning(
                "Checkpoint config.json is CTC-style but decode backend was %s; using ctc.",
                decode_backend,
            )
        decode_backend = "ctc"

    if decode_backend == "ctc" and text_settings.get("stack") == "whisper":
        log.warning(
            "Training config has stack=whisper but decode backend is CTC; "
            "use --backend whisper or --backend auto for Whisper checkpoints."
        )

    wh_lang = args.whisper_language or text_settings.get("whisper_language") or "sw"
    wh_task = args.whisper_task or text_settings.get("whisper_task") or "transcribe"
    if wh_task not in ("transcribe", "translate"):
        raise SystemExit(f"Invalid whisper task {wh_task!r}; use transcribe or translate.")

    proc_path = resolve_processor_path(args.model_path, args.processor_path)
    log.info("Loading processor from %s", proc_path)

    if decode_backend == "whisper":
        from transformers import WhisperProcessor

        from src.models.whisper_factory import is_peft_adapter_checkpoint, load_whisper_model_for_eval

        processor = WhisperProcessor.from_pretrained(proc_path)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=wh_lang, task=wh_task)
        model_input_name = "input_features"
        log.info("Whisper decode: language=%s task=%s", wh_lang, wh_task)
        gen_req = int(text_settings.get("generation_max_length") or 444)
        if is_peft_adapter_checkpoint(args.model_path):
            log.info(
                "Loading LoRA Whisper checkpoint from %s (base=%s)",
                args.model_path,
                text_settings.get("pretrained_model") or "adapter_config.json",
            )
        else:
            log.info("Loading Whisper model from %s", args.model_path)
        model = load_whisper_model_for_eval(
            args.model_path,
            processor=processor,
            whisper_language=wh_lang,
            whisper_task=wh_task,
            generation_max_length=gen_req,
            text_settings=text_settings,
        )
        whisper_decoder_cap = int(model.generation_config.max_length)
        log.info(
            "Whisper generate cap: max_length=%s (config generation_max_length=%s)",
            whisper_decoder_cap,
            gen_req,
        )
    else:
        from transformers import AutoProcessor

        from src.models.ctc_factory import load_ctc_model_for_eval
        from src.models.whisper_factory import is_peft_adapter_checkpoint

        processor = AutoProcessor.from_pretrained(proc_path)
        if getattr(processor, "tokenizer", None) is None and not hasattr(processor, "batch_decode"):
            raise RuntimeError("Processor has no tokenizer; cannot decode CTC outputs.")
        forced_decoder_ids = None
        model_input_name = probe_model_input_name(processor)
        if is_peft_adapter_checkpoint(args.model_path):
            log.info(
                "Loading LoRA CTC checkpoint from %s (base=%s)",
                args.model_path,
                text_settings.get("pretrained_model") or "adapter_config.json",
            )
        else:
            log.info("Loading CTC model from %s", args.model_path)
        model = load_ctc_model_for_eval(args.model_path, text_settings=text_settings)

    model.to(device)
    model.eval()

    log.info("model_input_name=%s decode_backend=%s device=%s", model_input_name, decode_backend, device)

    max_audio_seconds = resolve_max_audio_seconds_for_eval(args)
    text_mode = args.text_normalize
    log.info("max_audio_seconds=%s normalize=%s", max_audio_seconds, text_mode)
    jiwer_tr_w: Any = None
    jiwer_tr_c: Any = None
    if text_mode == "jiwer_default":
        jiwer_tr_w, jiwer_tr_c = try_build_jiwer_transforms()

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    revision = args.dataset_revision.strip() if args.dataset_revision else None
    kw = revision_kw(revision) if revision else {}

    all_pred_raw: List[str] = []
    all_ref_raw: List[str] = []
    per_set: Dict[str, Any] = {}
    predictions_out: List[Dict[str, Any]] = []

    for raw in args.test_datasets:
        spec = SplitSpec.parse(raw)
        log.info("Loading %s split=%s", spec.dataset_id, spec.split)
        ds = load_dataset(spec.dataset_id, split=spec.split, **kw)

        if args.audio_column and args.text_column:
            audio_col, text_col = args.audio_column, args.text_column
        else:
            audio_col, text_col = resolve_columns(list(ds.column_names))

        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        from src.data.eval_paths import collect_audio_path_labels

        path_labels = collect_audio_path_labels(ds, audio_col)
        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))

        rows_meta, dropped, dropped_qc = build_eval_meta(
            ds,
            audio_col,
            text_col,
            max_audio_seconds,
            path_labels=path_labels,
            qc_bundle=qc_bundle,
            log=log,
        )
        if rows_meta and args.chunk_long_audio_seconds is None and max_audio_seconds is None:
            mx_dur = max(float(r["audio_duration_s"]) for r in rows_meta)
            if mx_dur > 45.0:
                log.warning(
                    "Longest clip is %.1fs with no duration cap (--max_audio_seconds / --max-input-seconds) "
                    "or --chunk_long_audio_seconds; GPU may OOM (try --chunk_long_audio_seconds 30 or --fp16).",
                    mx_dur,
                )
        if dropped and max_audio_seconds is not None:
            log.info("Dropped %d clips longer than %.1fs", dropped, max_audio_seconds)
        if dropped_qc:
            log.info("Aggressive QC dropped %d clips for %s:%s", dropped_qc, spec.dataset_id, spec.split)
        if not rows_meta:
            log.warning("No examples left for %s:%s", spec.dataset_id, spec.split)
            per_set[f"{spec.dataset_id}:{spec.split}"] = {
                "wer": None,
                "cer": None,
                "n": 0,
                "dropped_long": dropped,
                "dropped_qc": dropped_qc,
            }
            continue

        for r in rows_meta:
            r["reference"] = eval_reference_like_training(r["reference"], text_settings)

        if decode_backend == "whisper":
            pred_raw, ref_raw_col, decode_times = transcribe_whisper_batches_streaming(
                ds,
                audio_col,
                model,
                processor,
                rows_meta,
                device,
                args.batch_size,
                forced_decoder_ids=forced_decoder_ids,
                decoder_max_length=whisper_decoder_cap,
                chunk_long_audio_seconds=args.chunk_long_audio_seconds,
                fp16=bool(args.fp16),
                cuda_empty_cache=bool(args.cuda_empty_cache),
            )
        else:
            pred_raw, ref_raw_col, decode_times = transcribe_batches_streaming(
                ds,
                audio_col,
                model,
                processor,
                model_input_name,
                rows_meta,
                device,
                args.batch_size,
                chunk_long_audio_seconds=args.chunk_long_audio_seconds,
                fp16=bool(args.fp16),
                cuda_empty_cache=bool(args.cuda_empty_cache),
            )

        if not args.no_format_decode:
            from src.data.text_format import format_decode_output

            discourse_commas = bool(args.discourse_commas) or bool(
                text_settings.get("enrich_discourse_punctuation", False)
            )
            pred_raw = [
                format_decode_output(p, discourse_commas=discourse_commas) for p in pred_raw
            ]

        qm = compute_split_quality_metrics(
            pred_raw, ref_raw_col, text_mode, wer_m, cer_m, jiwer_tr_w, jiwer_tr_c
        )
        wer_v, cer_v = qm["wer"], qm["cer"]

        key = f"{spec.dataset_id}:{spec.split}"
        per_set[key] = {
            **{k: v for k, v in qm.items() if k not in ("wer_raw", "cer_raw")},
            "n": len(rows_meta),
            "dropped_long": dropped,
            "dropped_qc": dropped_qc,
        }
        log.info("%s WER=%s CER=%s n=%d", key, wer_v, cer_v, len(rows_meta))

        all_pred_raw.extend(pred_raw)
        all_ref_raw.extend(ref_raw_col)
        for j, rrow in enumerate(rows_meta):
            pred_j = pred_raw[j]
            ref_j = ref_raw_col[j]
            dur = float(rrow.get("audio_duration_s") or 0.0)
            dwall = float(decode_times[j]) if j < len(decode_times) else 0.0
            rx = rtfx_from_times(dur, dwall)
            wu, cu = utterance_wer_cer(ref_j, pred_j, "none", wer_m, cer_m, jiwer_tr_w, jiwer_tr_c)
            from src.data.text_format import punctuation_recall

            rec: Dict[str, Any] = {
                "dataset": spec.dataset_id,
                "split": spec.split,
                "row_idx": rrow["row_idx"],
                "audio_path": rrow.get("audio_path", ""),
                "audio_duration_s": rrow.get("audio_duration_s"),
                "reference": ref_j,
                "prediction": pred_j,
                "wer": wu,
                "cer": cu,
                "punct_recall": punctuation_recall(ref_j, pred_j),
                "decode_wall_s": dwall if dwall > 0.0 else None,
                "rtfx": rx,
            }
            if text_mode != "none":
                rec.update(
                    extra_normalized_fields_for_row(
                        ref_j, pred_j, text_mode, wer_m, cer_m, jiwer_tr_w, jiwer_tr_c
                    )
                )
                rec["rtfx_normalized"] = rx
            predictions_out.append(rec)

    pool_pairs = [(p, r) for p, r in zip(all_pred_raw, all_ref_raw) if str(r).strip()]

    pooled_qm = compute_split_quality_metrics(
        all_pred_raw, all_ref_raw, text_mode, wer_m, cer_m, jiwer_tr_w, jiwer_tr_c
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled: Dict[str, Any] = {
        **{k: v for k, v in pooled_qm.items() if k not in ("wer_raw", "cer_raw")},
        "n_utterances": len(pool_pairs),
    }

    metrics: Dict[str, Any] = {
        "text_normalize": text_mode,
        "pooled": pooled,
        "per_set": per_set,
        "run_info": {
            "model_path": args.model_path,
            "processor_path": proc_path,
            "model_input_name": model_input_name,
            "max_audio_seconds": max_audio_seconds,
            "chunk_long_audio_seconds": args.chunk_long_audio_seconds,
            "fp16": bool(args.fp16),
            "cuda_empty_cache": bool(args.cuda_empty_cache),
            "decode_mode": f"streaming_per_batch_{decode_backend}",
            "decode_backend": decode_backend,
            "test_datasets": list(args.test_datasets),
            "dataset_revision": args.dataset_revision,
            "training_text_settings": {k: v for k, v in text_settings.items() if k != "training_config_raw"},
            "aggressive_qc": bool(qc_bundle),
            "qc_use_may6_text_norm": qc_bundle[1] if qc_bundle else None,
            "format_decode": not bool(args.no_format_decode),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(
        json.dumps(predictions_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_predictions_csv(out_dir / "predictions.csv", predictions_out, text_mode)
    log.info("Wrote %s, %s, and %s", out_dir / "metrics.json", out_dir / "predictions.json", out_dir / "predictions.csv")
    log.info("Pooled WER=%s CER=%s (n=%d)", pooled_qm.get("wer"), pooled_qm.get("cer"), len(pool_pairs))


if __name__ == "__main__":
    main()
