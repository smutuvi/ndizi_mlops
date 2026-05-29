#!/usr/bin/env python3
"""
Backup copy (pre-streaming): precomputes processor outputs for **every** row before decode.

High host RAM use on large splits and CUDA OOM if long clips are batched together. Prefer
``evaluate_asr_batch.py`` (streaming + optional chunking). Kept for small tests / comparison.

---

Batch greedy CTC evaluation for checkpoints produced by ``scripts/train_model.py``.

Loads ``AutoModelForCTC`` + ``AutoProcessor`` (w2v-BERT uses ``input_features``;
wav2vec2-style uses ``input_values`` — same as training).

**Audio length:** by default **no** max-duration filter (all utterances are scored),
matching common batch eval drivers that only resample to 16 kHz. To align with
training’s ``max_input_seconds`` (typically 30), pass ``--max_audio_seconds 30``.

**References vs training:** if ``training_config_resolved.json`` exists next to the
checkpoint (written by ``train_model.py``), it is loaded to apply the same text
cleaning as ``load_datasets`` (hub identity vs ``character_set``), then the same
tokenizer round-trip as ``src/training/metrics.py`` (``group_tokens=False`` on labels).

Examples (from ``ndizi_mlops/``):

  python3 scripts/evaluate_asr_batch_backup_precompute_all_rows.py \\
    --model_path inprogress/.../facebook-w2v-bert-2.0-12052026-090000 \\
    --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \\
    --output_dir eval/run1

  # Match training clip cap (30 s) and explicit training YAML/JSON:
  python3 scripts/evaluate_asr_batch_backup_precompute_all_rows.py \\
    --model_path ... --training_config config_files/w2vbert/ndizi_w2vbert_merged_1epoch.json \\
    --max_audio_seconds 30 \\
    --test_datasets smutuvi/ndizi-1:test --output_dir eval/run2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


def resolve_processor_path(model_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    p = Path(model_path).resolve()
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
) -> str:
    """Same cleaning as ``src/data/dataset.py`` in ``load_datasets`` before encoding."""
    from src.data.preprocessing import clean_text_batch, hub_ctc_identity_clean_batch

    batch = {"transcription": [str(raw or "")]}
    if use_hub_ctc_checkpoint:
        return hub_ctc_identity_clean_batch(batch)["clean_transcription"][0]
    return clean_text_batch(batch, character_set, apply_accent_replacements)["clean_transcription"][0]


def load_eval_text_settings(
    model_path: str,
    training_config: Optional[str],
    ignore_resolved_training_config: bool,
) -> Dict[str, Any]:
    """
    Resolve ``use_hub_ctc_checkpoint``, ``character_set``, ``apply_accent_replacements``
    from ``--training_config`` and/or ``<model_path>/training_config_resolved.json``.
    Explicit ``--training_config`` is tried first; then the resolved JSON next to the checkpoint
    (unless ``--ignore_resolved_training_config``).
    """
    defaults = {
        "use_hub_ctc_checkpoint": True,
        "character_set": "abcdefghijklmnopqrstuvwxyz0123456789 -'",
        "apply_accent_replacements": True,
        "config_path": None,
    }
    from src.utils.config import load_config

    candidates: List[Path] = []
    if training_config:
        candidates.append(Path(training_config).expanduser().resolve())
    if not ignore_resolved_training_config:
        candidates.append(Path(model_path).resolve() / "training_config_resolved.json")

    for p in candidates:
        if p.is_file():
            cfg = load_config(p)
            return {
                "use_hub_ctc_checkpoint": bool(cfg.use_hub_ctc_checkpoint),
                "character_set": str(cfg.character_set),
                "apply_accent_replacements": bool(cfg.apply_accent_replacements),
                "config_path": str(p),
            }
    return defaults


def build_eval_rows(
    ds,
    processor: Any,
    model_input_name: str,
    audio_col: str,
    text_col: str,
    max_audio_seconds: Optional[float],
) -> Tuple[List[dict], int]:
    rows: List[dict] = []
    dropped = 0
    for i in range(len(ds)):
        ex = ds[i]
        audio = ex[audio_col]
        arr = audio["array"]
        sr = int(audio["sampling_rate"])
        dur = float(len(arr) / max(sr, 1))
        if max_audio_seconds is not None and dur > float(max_audio_seconds):
            dropped += 1
            continue
        out = processor(arr, sampling_rate=sr)
        if isinstance(out, dict):
            feat = out[model_input_name][0]
        else:
            feat = getattr(out, model_input_name)[0]
        ref = str(ex.get(text_col) or "")
        rows.append({model_input_name: feat, "reference": ref, "row_idx": i, "audio_duration_s": dur})
    return rows, dropped


def transcribe_batches(
    model: Any,
    processor: Any,
    model_input_name: str,
    rows: List[dict],
    device: Any,
    batch_size: int,
) -> Tuple[List[str], List[str]]:
    import torch
    from tqdm import tqdm

    collator = RowCollator(processor=processor, model_input_name=model_input_name)
    model.eval()
    preds: List[str] = []
    refs: List[str] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="decode"):
        chunk = rows[start : start + batch_size]
        batch = collator(chunk)
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        with torch.inference_mode():
            logits = model(**batch).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        preds.extend(decode_pred_ids(processor, pred_ids))
        refs.extend(str(r["reference"]) for r in chunk)
    return preds, refs


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch CTC ASR evaluation on Hub splits.")
    parser.add_argument("--model_path", type=str, required=True, help="Saved checkpoint dir or Hub model id.")
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Override processor directory (default: infer from checkpoint layout).",
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
    parser.add_argument("--max_samples", type=int, default=None, help="Cap utterances per split (debug).")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--audio_column", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--normalize_wer", action="store_true", help="Lowercase + collapse whitespace for WER/CER.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
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
    from transformers import AutoModelForCTC, AutoProcessor

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        log.info("Text cleaning settings from %s", text_settings["config_path"])
    else:
        log.info(
            "No training config loaded; using defaults (Hub CTC-style strip for text). "
            "For custom-vocab runs, pass --training_config or keep training_config_resolved.json next to the checkpoint."
        )

    proc_path = resolve_processor_path(args.model_path, args.processor_path)
    log.info("Loading processor from %s", proc_path)
    processor = AutoProcessor.from_pretrained(proc_path)
    if getattr(processor, "tokenizer", None) is None and not hasattr(processor, "batch_decode"):
        raise RuntimeError("Processor has no tokenizer; cannot decode CTC outputs.")

    log.info("Loading model from %s", args.model_path)
    model = AutoModelForCTC.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    model_input_name = probe_model_input_name(processor)
    log.info("model_input_name=%s device=%s", model_input_name, device)

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    revision = args.dataset_revision.strip() if args.dataset_revision else None
    kw = revision_kw(revision) if revision else {}

    all_pred: List[str] = []
    all_ref: List[str] = []
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

        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))
        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        rows, dropped = build_eval_rows(
            ds,
            processor,
            model_input_name,
            audio_col,
            text_col,
            args.max_audio_seconds,
        )
        if dropped and args.max_audio_seconds is not None:
            log.info("Dropped %d clips longer than %.1fs", dropped, args.max_audio_seconds)
        if not rows:
            log.warning("No examples left for %s:%s", spec.dataset_id, spec.split)
            per_set[f"{spec.dataset_id}:{spec.split}"] = {
                "wer": None,
                "cer": None,
                "n": 0,
                "dropped_long": dropped,
            }
            continue

        pred_raw, ref_raw_col = transcribe_batches(
            model, processor, model_input_name, rows, device, args.batch_size
        )

        ref_clean = [
            clean_transcription_like_training(
                r,
                use_hub_ctc_checkpoint=text_settings["use_hub_ctc_checkpoint"],
                character_set=text_settings["character_set"],
                apply_accent_replacements=text_settings["apply_accent_replacements"],
            )
            for r in ref_raw_col
        ]
        ref_for_wer = [reference_for_wer_like_training(processor, c) for c in ref_clean]

        pred = list(pred_raw)
        ref = list(ref_for_wer)
        if args.normalize_wer:
            pred = [wer_normalize(p) for p in pred]
            ref = [wer_normalize(r) for r in ref]

        pairs = [(p, r) for p, r in zip(pred, ref) if str(r).strip()]
        if not pairs:
            wer_v = cer_v = None
        else:
            p2, r2 = zip(*pairs)
            wer_v = float(wer_m.compute(predictions=list(p2), references=list(r2)))
            cer_v = float(cer_m.compute(predictions=list(p2), references=list(r2)))

        key = f"{spec.dataset_id}:{spec.split}"
        per_set[key] = {"wer": wer_v, "cer": cer_v, "n": len(rows), "dropped_long": dropped}
        log.info("%s WER=%s CER=%s n=%d", key, wer_v, cer_v, len(rows))

        all_pred.extend(pred)
        all_ref.extend(ref)
        for j, rrow in enumerate(rows):
            predictions_out.append(
                {
                    "dataset": spec.dataset_id,
                    "split": spec.split,
                    "row_idx": rrow["row_idx"],
                    "audio_duration_s": rrow.get("audio_duration_s"),
                    "reference_raw": ref_raw_col[j],
                    "reference_clean": ref_clean[j],
                    "reference_for_wer": ref_for_wer[j],
                    "prediction": pred[j],
                }
            )

    pool_pairs = [(p, r) for p, r in zip(all_pred, all_ref) if str(r).strip()]
    if pool_pairs:
        pp, rr = zip(*pool_pairs)
        pooled_wer = float(wer_m.compute(predictions=list(pp), references=list(rr)))
        pooled_cer = float(cer_m.compute(predictions=list(pp), references=list(rr)))
    else:
        pooled_wer = pooled_cer = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_path": args.model_path,
        "processor_path": proc_path,
        "model_input_name": model_input_name,
        "max_audio_seconds": args.max_audio_seconds,
        "training_text_settings": text_settings,
        "normalize_wer": bool(args.normalize_wer),
        "pooled": {"wer": pooled_wer, "cer": pooled_cer, "n_utterances": len(pool_pairs)},
        "per_set": per_set,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(
        json.dumps(predictions_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote %s and %s", out_dir / "metrics.json", out_dir / "predictions.json")
    log.info("Pooled WER=%s CER=%s (n=%d)", pooled_wer, pooled_cer, len(pool_pairs))


if __name__ == "__main__":
    main()
