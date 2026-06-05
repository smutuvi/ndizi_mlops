#!/usr/bin/env python3
"""
Transcribe one local WAV (16 kHz resampling via ``datasets.Audio``) with a CTC or Whisper
checkpoint from this bundle. Reuses decode logic from ``scripts/evaluate_asr_batch.py``.

Examples::

  python3 scripts/evaluate_asr_single.py \\
    --model_path inprogress/.../facebook-w2v-bert-2.0-12052026-091323 \\
    --audio_path /data/clip.wav

  python3 scripts/evaluate_asr_single.py \\
    --model_path inprogress/.../openai-whisper-small-... \\
    --audio_path /data/clip.wav --backend auto

  python3 scripts/evaluate_asr_single.py --model_path ... --audio_path clip.wav \\
    --chunk_long_audio_seconds 30 --output_json pred.json

  # Match training duration cap (aliases align with train_whisper.py):
  python3 scripts/evaluate_asr_single.py --model_path ... --audio_path clip.wav \\
    --max-input-seconds 30

  # LoRA continual checkpoint (adapter + training_config_resolved.json in run dir):
  python3 scripts/evaluate_asr_single.py --model_path runs/.../msingiai-sauti-asr-... \\
    --audio_path clip.wav --backend auto
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def _load_eval_batch_module():
    """Load ``evaluate_asr_batch`` without running its CLI (``__name__`` is not ``__main__``).

    The module must be registered in ``sys.modules`` before ``exec_module`` so that
    ``@dataclass`` (and similar) can resolve ``sys.modules[cls.__module__]`` during class body
    execution.
    """
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


def _one_row_dataset(audio_path: Path):
    from datasets import Audio, Dataset

    p = audio_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")
    ds = Dataset.from_dict({"audio": [str(p)], "transcription": [""]})
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one WAV with a CTC (w2v-BERT) or Whisper checkpoint.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Saved checkpoint directory or Hub id.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to a .wav (or soundfile-readable) mono/stereo file.")
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Override processor directory (same semantics as evaluate_asr_batch).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "ctc", "whisper"],
        default="auto",
        help='Decode path: "auto" uses training_config_resolved.json next to the checkpoint when present.',
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default=None,
        help="Optional training JSON/YAML (overrides resolved config next to checkpoint).",
    )
    parser.add_argument(
        "--ignore_resolved_training_config",
        action="store_true",
        help="Do not read <model_path>/training_config_resolved.json.",
    )
    parser.add_argument("--whisper_language", type=str, default=None)
    parser.add_argument("--whisper_task", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--fp16", action="store_true", help="Use torch.autocast(float16) on CUDA for forward/generate.")
    parser.add_argument("--cuda_empty_cache", action="store_true")
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=None,
        help="If set, refuse to decode clips longer than this (seconds).",
    )
    parser.add_argument(
        "--max-input-seconds",
        type=float,
        default=None,
        dest="max_input_seconds",
        metavar="SEC",
        help="Alias for --max_audio_seconds (same as train_whisper / evaluate_asr_batch).",
    )
    parser.add_argument(
        "--no-max-input-filter",
        action="store_true",
        dest="no_max_audio_cap",
        help="Do not apply a max-duration refusal (same as omitting --max_audio_seconds).",
    )
    parser.add_argument(
        "--chunk_long_audio_seconds",
        type=float,
        default=None,
        help="If set and clip is longer, decode in non-overlapping chunks (preds joined with spaces).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write {\"text\": ...} to this path.",
    )
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
    log = logging.getLogger("evaluate_asr_single")

    os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

    bundle_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(bundle_root))

    eb = _load_eval_batch_module()
    eb.load_env_file(bundle_root / ".env")

    import torch
    from huggingface_hub import login as hf_login

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if args.backend == "auto":
        decode_backend = "whisper" if text_settings.get("stack") == "whisper" else "ctc"
    else:
        decode_backend = args.backend

    ck_kind = eb.infer_decode_backend_from_checkpoint(args.model_path)
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

    wh_lang = args.whisper_language or text_settings.get("whisper_language") or "sw"
    wh_task = args.whisper_task or text_settings.get("whisper_task") or "transcribe"
    if wh_task not in ("transcribe", "translate"):
        raise SystemExit(f"Invalid whisper task {wh_task!r}; use transcribe or translate.")

    proc_path = eb.resolve_processor_path(args.model_path, args.processor_path)
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
            "Whisper generate cap: max_length=%s (generation_max_length=%s)",
            whisper_decoder_cap,
            gen_req,
        )
    else:
        whisper_decoder_cap = 0
        from transformers import AutoProcessor

        from src.models.ctc_factory import load_ctc_model_for_eval
        from src.models.whisper_factory import is_peft_adapter_checkpoint

        processor = AutoProcessor.from_pretrained(proc_path)
        if getattr(processor, "tokenizer", None) is None and not hasattr(processor, "batch_decode"):
            raise RuntimeError("Processor has no tokenizer; cannot decode CTC outputs.")
        forced_decoder_ids = None
        model_input_name = eb.probe_model_input_name(processor)
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
    log.info("decode_backend=%s model_input_name=%s device=%s", decode_backend, model_input_name, device)

    audio_p = Path(args.audio_path)
    ds = _one_row_dataset(audio_p)
    max_audio = eb.resolve_max_audio_seconds_for_eval(args)
    rows_meta, dropped, _dropped_qc = eb.build_eval_meta(ds, "audio", "transcription", max_audio)
    if dropped:
        raise SystemExit(
            f"Audio longer than max duration cap ({max_audio}); refusing to decode ({dropped} row(s))."
        )
    if not rows_meta:
        raise SystemExit("No audio to decode (empty file or filtering removed the clip).")

    if decode_backend == "whisper":
        pred_raw, _refs, _dt = eb.transcribe_whisper_batches_streaming(
            ds,
            "audio",
            model,
            processor,
            rows_meta,
            device,
            1,
            forced_decoder_ids=forced_decoder_ids,
            decoder_max_length=whisper_decoder_cap,
            chunk_long_audio_seconds=args.chunk_long_audio_seconds,
            fp16=bool(args.fp16),
            cuda_empty_cache=bool(args.cuda_empty_cache),
        )
    else:
        pred_raw, _refs, _dt = eb.transcribe_batches_streaming(
            ds,
            "audio",
            model,
            processor,
            model_input_name,
            rows_meta,
            device,
            1,
            chunk_long_audio_seconds=args.chunk_long_audio_seconds,
            fp16=bool(args.fp16),
            cuda_empty_cache=bool(args.cuda_empty_cache),
        )

    text = pred_raw[0] if pred_raw else ""
    if not args.no_format_decode:
        from src.data.text_format import format_decode_output

        discourse_commas = bool(args.discourse_commas)
        if not discourse_commas and text_settings:
            discourse_commas = bool(text_settings.get("enrich_discourse_punctuation", False))
        text = format_decode_output(text, discourse_commas=discourse_commas)
    print(text, flush=True)

    if args.output_json:
        out_p = Path(args.output_json).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": text,
            "model_path": args.model_path,
            "audio_path": str(audio_p.resolve()),
            "decode_backend": decode_backend,
        }
        out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Wrote %s", out_p)


if __name__ == "__main__":
    main()
