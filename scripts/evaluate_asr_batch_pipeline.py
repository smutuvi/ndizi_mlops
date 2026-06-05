#!/usr/bin/env python3
"""
Batch ASR evaluation via Hugging Face ``transformers.pipeline`` (automatic-speech-recognition).

Same Hub splits and output files as ``scripts/evaluate_asr_batch.py`` (``metrics.json``,
``predictions.json``, ``predictions.csv``). Long audio uses the pipeline's built-in
``chunk_length_s`` / ``stride_length_s`` chunking (not the custom waveform splitter in
``evaluate_asr_batch.py``).

Examples (from ``ndizi_mlops/``):

  python3 scripts/evaluate_asr_batch_pipeline.py \\
    --model_path runs/.../checkpoint \\
    --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \\
    --output_dir eval/pipeline_whisper \\
    --chunk_length_s 30 --batch_size 8 --fp16

  python3 scripts/evaluate_asr_batch_pipeline.py \\
    --model_path inprogress/.../facebook-w2v-bert-2.0-... \\
    --backend ctc \\
    --test_datasets smutuvi/ndizi-1:test --output_dir eval/pipeline_ctc
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")


def _load_eval_batch_module():
    bundle = Path(__file__).resolve().parent.parent
    path = bundle / "scripts" / "evaluate_asr_batch.py"
    name = "ndizi_evaluate_asr_batch"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pipeline_device_arg(device: Any) -> Any:
    """Map ``torch.device`` to a value accepted by ``transformers.pipeline(..., device=...)``."""
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    if device.type == "mps":
        return "mps"
    return -1


def extract_pipeline_text(result: Any) -> str:
    """Normalize pipeline return value to a single hypothesis string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        text = result.get("text")
        if text is not None:
            return str(text).strip()
        chunks = result.get("chunks")
        if isinstance(chunks, list):
            from src.data.text_format import join_chunk_predictions

            parts = [str(c.get("text", "")).strip() for c in chunks if isinstance(c, dict)]
            return join_chunk_predictions(parts)
        return ""
    if isinstance(result, list):
        if not result:
            return ""
        from src.data.text_format import join_chunk_predictions

        if isinstance(result[0], dict) and "text" in result[0]:
            parts = [str(x.get("text", "")).strip() for x in result if isinstance(x, dict)]
            return join_chunk_predictions(parts)
        if isinstance(result[0], str):
            return join_chunk_predictions([str(x).strip() for x in result])
        return extract_pipeline_text(result[0])
    return str(result).strip()


def build_asr_pipeline(
    model: Any,
    processor: Any,
    *,
    device: Any,
    torch_dtype: Any,
    batch_size: int,
    chunk_length_s: Optional[float],
    stride_length_s: Optional[float],
    return_timestamps: bool,
) -> Any:
    from transformers import pipeline

    tok = getattr(processor, "tokenizer", None)
    fe = getattr(processor, "feature_extractor", None)
    if fe is None and callable(processor) and tok is not None:
        fe = processor

    pipe_kw: Dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": model,
        "tokenizer": tok,
        "feature_extractor": fe,
        "batch_size": int(batch_size),
        "torch_dtype": torch_dtype,
        "device": pipeline_device_arg(device),
    }
    if chunk_length_s is not None:
        pipe_kw["chunk_length_s"] = float(chunk_length_s)
    if stride_length_s is not None:
        pipe_kw["stride_length_s"] = float(stride_length_s)
    if return_timestamps:
        pipe_kw["return_timestamps"] = True

    return pipeline(**pipe_kw)


def transcribe_split_with_pipeline(
    ds: Any,
    audio_col: str,
    pipe: Any,
    rows_meta: List[dict],
    *,
    generate_kwargs: Optional[Dict[str, Any]],
    return_timestamps: bool,
) -> Tuple[List[str], List[str], List[float]]:
    import numpy as np
    from tqdm import tqdm

    preds: List[str] = []
    refs: List[str] = []
    decode_times: List[float] = []
    gen_kw = dict(generate_kwargs or {})

    for m in tqdm(rows_meta, desc="pipeline decode"):
        i = int(m["row_idx"])
        ref = str(m["reference"])
        ex = ds[i]
        audio = ex[audio_col]
        arr = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
        sr = int(audio["sampling_rate"])
        inputs = {"array": arr, "sampling_rate": sr}

        t0 = time.perf_counter()
        if return_timestamps:
            out = pipe(inputs, generate_kwargs=gen_kw, return_timestamps=True)
        else:
            out = pipe(inputs, generate_kwargs=gen_kw)
        t1 = time.perf_counter()

        preds.append(extract_pipeline_text(out))
        refs.append(ref)
        decode_times.append(float(t1 - t0))

    return preds, refs, decode_times


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ASR eval using transformers.pipeline (automatic-speech-recognition).",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "ctc", "whisper"],
        default="auto",
    )
    parser.add_argument("--whisper_language", type=str, default=None)
    parser.add_argument("--whisper_task", type=str, default=None)
    parser.add_argument("--test_datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--chunk_length_s",
        "--chunk_long_audio_seconds",
        type=float,
        default=None,
        dest="chunk_length_s",
        metavar="SEC",
        help="Pipeline chunk_length_s for long audio (e.g. 30). Omit to decode each clip as one piece.",
    )
    parser.add_argument(
        "--stride_length_s",
        type=float,
        default=None,
        help="Pipeline stride between chunks (seconds). Default: pipeline default (often no overlap).",
    )
    parser.add_argument(
        "--return_timestamps",
        action="store_true",
        help="Request timestamp chunks from the pipeline (Whisper).",
    )
    parser.add_argument("--max_audio_seconds", type=float, default=None)
    parser.add_argument(
        "--max-input-seconds",
        type=float,
        default=None,
        dest="max_input_seconds",
        metavar="SEC",
    )
    parser.add_argument(
        "--no-max-input-filter",
        action="store_true",
        dest="no_max_audio_cap",
    )
    parser.add_argument("--training_config", type=str, default=None)
    parser.add_argument("--ignore_resolved_training_config", action="store_true")
    parser.add_argument(
        "--aggressive-qc",
        action="store_true",
        help="Drop eval rows that fail training multi-gate QC (see scripts/evaluate_asr_batch.py).",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--audio_column", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "simple", "jiwer_default"],
        default="none",
        dest="text_normalize",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cuda_empty_cache", action="store_true")
    parser.add_argument(
        "--no-format-decode",
        action="store_true",
        help="Skip post-decode transcript formatting (spacing after .?!, etc.).",
    )
    parser.add_argument(
        "--discourse-commas",
        action="store_true",
        help="Insert commas before common Swahili discourse markers in decode output.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("evaluate_asr_batch_pipeline")

    bundle_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(bundle_root))
    eb = _load_eval_batch_module()
    eb.load_env_file(bundle_root / ".env")

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

    text_settings = eb.load_eval_text_settings(
        args.model_path,
        args.training_config,
        bool(args.ignore_resolved_training_config),
    )
    if text_settings.get("config_path"):
        log.info("Training settings from %s (stack=%s)", text_settings["config_path"], text_settings.get("stack"))

    qc_bundle = eb.resolve_aggressive_qc_bundle(bool(args.aggressive_qc), text_settings, log)
    if qc_bundle is not None:
        log.info("Aggressive QC on (qc_use_may6_text_norm=%s)", qc_bundle[1])

    if args.backend == "auto":
        decode_backend = "whisper" if text_settings.get("stack") == "whisper" else "ctc"
    else:
        decode_backend = args.backend

    ck_kind = eb.infer_decode_backend_from_checkpoint(args.model_path)
    if ck_kind == "whisper":
        decode_backend = "whisper"
    elif ck_kind == "ctc":
        decode_backend = "ctc"

    wh_lang = args.whisper_language or text_settings.get("whisper_language") or "sw"
    wh_task = args.whisper_task or text_settings.get("whisper_task") or "transcribe"
    if wh_task not in ("transcribe", "translate"):
        raise SystemExit(f"Invalid whisper task {wh_task!r}; use transcribe or translate.")

    proc_path = eb.resolve_processor_path(args.model_path, args.processor_path)
    log.info("Loading processor from %s", proc_path)

    generate_kwargs: Dict[str, Any] = {}
    whisper_decoder_cap = 0

    if decode_backend == "whisper":
        from transformers import WhisperProcessor

        from src.models.whisper_factory import is_peft_adapter_checkpoint, load_whisper_model_for_eval

        processor = WhisperProcessor.from_pretrained(proc_path)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=wh_lang, task=wh_task)
        gen_req = int(text_settings.get("generation_max_length") or 444)
        if is_peft_adapter_checkpoint(args.model_path):
            log.info("Loading LoRA Whisper checkpoint from %s", args.model_path)
        model = load_whisper_model_for_eval(
            args.model_path,
            processor=processor,
            whisper_language=wh_lang,
            whisper_task=wh_task,
            generation_max_length=gen_req,
            text_settings=text_settings,
        )
        whisper_decoder_cap = int(model.generation_config.max_length)
        generate_kwargs = {
            "forced_decoder_ids": forced_decoder_ids,
            "max_length": whisper_decoder_cap,
        }
        log.info("Whisper pipeline: language=%s task=%s max_length=%s", wh_lang, wh_task, whisper_decoder_cap)
    else:
        from transformers import AutoProcessor

        from src.models.ctc_factory import load_ctc_model_for_eval
        from src.models.whisper_factory import is_peft_adapter_checkpoint

        processor = AutoProcessor.from_pretrained(proc_path)
        if getattr(processor, "tokenizer", None) is None and not hasattr(processor, "batch_decode"):
            raise RuntimeError("Processor has no tokenizer; cannot decode CTC outputs.")
        if is_peft_adapter_checkpoint(args.model_path):
            log.info("Loading LoRA CTC checkpoint from %s", args.model_path)
        else:
            log.info("Loading CTC model from %s", args.model_path)
        model = load_ctc_model_for_eval(args.model_path, text_settings=text_settings)
        model_input_name = eb.probe_model_input_name(processor)
        log.info("CTC pipeline: model_input_name=%s", model_input_name)

    model.to(device)
    model.eval()

    if args.fp16 and device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    log.info(
        "Building ASR pipeline (chunk_length_s=%s stride_length_s=%s batch_size=%s dtype=%s device=%s)",
        args.chunk_length_s,
        args.stride_length_s,
        args.batch_size,
        torch_dtype,
        device,
    )
    pipe = build_asr_pipeline(
        model,
        processor,
        device=device,
        torch_dtype=torch_dtype,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length_s,
        stride_length_s=args.stride_length_s,
        return_timestamps=bool(args.return_timestamps),
    )

    max_audio_seconds = eb.resolve_max_audio_seconds_for_eval(args)
    text_mode = args.text_normalize
    jiwer_tr_w: Any = None
    jiwer_tr_c: Any = None
    if text_mode == "jiwer_default":
        jiwer_tr_w, jiwer_tr_c = eb.try_build_jiwer_transforms()

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    revision = args.dataset_revision.strip() if args.dataset_revision else None
    kw = eb.revision_kw(revision) if revision else {}

    all_pred_raw: List[str] = []
    all_ref_raw: List[str] = []
    per_set: Dict[str, Any] = {}
    predictions_out: List[Dict[str, Any]] = []

    for raw in args.test_datasets:
        spec = eb.SplitSpec.parse(raw)
        log.info("Loading %s split=%s", spec.dataset_id, spec.split)
        ds = load_dataset(spec.dataset_id, split=spec.split, **kw)

        if args.audio_column and args.text_column:
            audio_col, text_col = args.audio_column, args.text_column
        else:
            audio_col, text_col = eb.resolve_columns(list(ds.column_names))

        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        from src.data.eval_paths import collect_audio_path_labels

        path_labels = collect_audio_path_labels(ds, audio_col)
        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))

        rows_meta, dropped, dropped_qc = eb.build_eval_meta(
            ds,
            audio_col,
            text_col,
            max_audio_seconds,
            path_labels=path_labels,
            qc_bundle=qc_bundle,
            log=log,
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
            r["reference"] = eb.eval_reference_like_training(r["reference"], text_settings)

        pred_raw, ref_raw_col, decode_times = transcribe_split_with_pipeline(
            ds,
            audio_col,
            pipe,
            rows_meta,
            generate_kwargs=generate_kwargs if decode_backend == "whisper" else None,
            return_timestamps=bool(args.return_timestamps),
        )

        if args.cuda_empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()

        if not args.no_format_decode:
            from src.data.text_format import format_decode_output

            discourse_commas = bool(args.discourse_commas) or bool(
                text_settings.get("enrich_discourse_punctuation", False)
            )
            pred_raw = [
                format_decode_output(p, discourse_commas=discourse_commas) for p in pred_raw
            ]

        qm = eb.compute_split_quality_metrics(
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
            rx = eb.rtfx_from_times(dur, dwall)
            wu, cu = eb.utterance_wer_cer(ref_j, pred_j, "none", wer_m, cer_m, jiwer_tr_w, jiwer_tr_c)
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
                    eb.extra_normalized_fields_for_row(
                        ref_j, pred_j, text_mode, wer_m, cer_m, jiwer_tr_w, jiwer_tr_c
                    )
                )
                rec["rtfx_normalized"] = rx
            predictions_out.append(rec)

    pool_pairs = [(p, r) for p, r in zip(all_pred_raw, all_ref_raw) if str(r).strip()]
    pooled_qm = eb.compute_split_quality_metrics(
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
            "decode_mode": "transformers_pipeline_automatic_speech_recognition",
            "decode_backend": decode_backend,
            "chunk_length_s": args.chunk_length_s,
            "stride_length_s": args.stride_length_s,
            "batch_size": args.batch_size,
            "fp16": bool(args.fp16),
            "return_timestamps": bool(args.return_timestamps),
            "max_audio_seconds": max_audio_seconds,
            "test_datasets": list(args.test_datasets),
            "dataset_revision": args.dataset_revision,
            "training_text_settings": {k: v for k, v in text_settings.items() if k != "training_config_raw"},
            "aggressive_qc": bool(qc_bundle),
            "qc_use_may6_text_norm": qc_bundle[1] if qc_bundle else None,
            "format_decode": not bool(args.no_format_decode),
            "whisper_generate_kwargs": generate_kwargs if decode_backend == "whisper" else None,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions.json").write_text(
        json.dumps(predictions_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    eb._write_predictions_csv(out_dir / "predictions.csv", predictions_out, text_mode)
    log.info("Wrote %s, %s, and %s", out_dir / "metrics.json", out_dir / "predictions.json", out_dir / "predictions.csv")
    log.info("Pooled WER=%s CER=%s (n=%d)", pooled_qm.get("wer"), pooled_qm.get("cer"), len(pool_pairs))


if __name__ == "__main__":
    main()
