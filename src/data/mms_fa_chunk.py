# src/data/mms_fa_chunk.py — MMS_FA word-aligned chunking for long train utterances (QC prep only).
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import torchaudio
    from torchaudio.pipelines import MMS_FA as _MMS_FA_BUNDLE

    _HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None  # type: ignore[assignment]
    _MMS_FA_BUNDLE = None  # type: ignore[assignment]
    _HAS_TORCHAUDIO = False


def require_torchaudio_mms_fa() -> None:
    if not _HAS_TORCHAUDIO:
        raise SystemExit(
            "qc_chunk_long_with_mms_fa requires torchaudio with MMS_FA. "
            "Install torchaudio in this environment and retry."
        )


def resolve_fa_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def load_mms_fa_context(device: str = "auto") -> Dict[str, Any]:
    """Load torchaudio MMS_FA model/tokenizer/aligner once per training run."""
    require_torchaudio_mms_fa()
    fa_device = resolve_fa_device(device)
    fa_model = _MMS_FA_BUNDLE.get_model().to(fa_device)  # type: ignore[union-attr]
    fa_tokenizer = _MMS_FA_BUNDLE.get_tokenizer()  # type: ignore[union-attr]
    fa_aligner = _MMS_FA_BUNDLE.get_aligner()  # type: ignore[union-attr]
    logger.info("MMS_FA loaded on device=%s", fa_device)
    return {
        "model": fa_model,
        "tokenizer": fa_tokenizer,
        "aligner": fa_aligner,
        "sample_rate": int(_MMS_FA_BUNDLE.sample_rate),  # type: ignore[union-attr]
        "device": fa_device,
        "allowed_chars": set(_MMS_FA_BUNDLE.get_dict().keys()),  # type: ignore[union-attr]
    }


def _forced_align_word_times_mms_fa(
    *,
    waveform_1d: np.ndarray,
    sampling_rate: int,
    transcript_words: List[str],
    fa_model,
    fa_tokenizer,
    fa_aligner,
    fa_sample_rate: int,
    device: str,
) -> List[Tuple[float, float, str]]:
    if not transcript_words:
        return []
    if sampling_rate != fa_sample_rate:
        wf = torch.tensor(waveform_1d, dtype=torch.float32).unsqueeze(0)
        wf = torchaudio.functional.resample(wf, sampling_rate, fa_sample_rate)  # type: ignore[union-attr]
    else:
        wf = torch.tensor(waveform_1d, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        emission, _ = fa_model(wf.to(device))
    token_spans = fa_aligner(emission[0], fa_tokenizer(transcript_words))
    num_frames = int(emission.size(1))
    if num_frames <= 0:
        return []
    ratio = wf.size(1) / num_frames / float(fa_sample_rate)

    out: List[Tuple[float, float, str]] = []
    for spans, word in zip(token_spans, transcript_words):
        if not spans:
            continue
        start_s = float(ratio * spans[0].start)
        end_s = float(ratio * spans[-1].end)
        if end_s <= start_s:
            continue
        out.append((start_s, end_s, word))
    return out


def _segment_by_word_times(
    word_times: List[Tuple[float, float, str]],
    *,
    chunk_seconds: float,
) -> List[Tuple[float, float, int, int]]:
    """
    Pack aligned words into <= chunk_seconds segments.
    Returns (start_s, end_s, word_start_idx, word_end_idx_exclusive).
    """
    if not word_times:
        return []
    segs: List[Tuple[float, float, int, int]] = []
    cur_start = word_times[0][0]
    cur_end = cur_start
    word_start = 0
    cur_count = 0

    for idx, (w_start, w_end, _w) in enumerate(word_times):
        if cur_count == 0:
            cur_start = w_start
            cur_end = w_end
            word_start = idx
            cur_count = 1
            continue

        proposed_end = w_end
        if proposed_end - cur_start <= chunk_seconds:
            cur_end = proposed_end
            cur_count += 1
            continue

        segs.append((cur_start, cur_end, word_start, idx))
        cur_start = w_start
        cur_end = w_end
        word_start = idx
        cur_count = 1

    if cur_count > 0:
        segs.append((cur_start, cur_end, word_start, len(word_times)))
    return segs


def _word_spans_in_text(text: str) -> List[tuple[int, int]]:
    """Character spans of whitespace-delimited tokens (preserves glued punctuation)."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", str(text or ""))]


def _slice_text_by_fa_word_span(
    text: str,
    *,
    word_start: int,
    word_end_exclusive: int,
    n_fa_words: int,
) -> str:
    """Map FA word span to a substring of ``text`` using original token boundaries."""
    from src.data.text_format import format_transcript

    raw = str(text or "").strip()
    if not raw or n_fa_words <= 0 or word_end_exclusive <= word_start:
        return ""

    spans = _word_spans_in_text(raw)
    n_words = len(spans)
    if n_words == 0:
        return ""

    if n_words == n_fa_words:
        i0 = max(0, min(word_start, n_words))
        i1 = max(i0, min(word_end_exclusive, n_words))
    else:
        i0 = int(round(word_start * n_words / n_fa_words))
        i1 = int(round(word_end_exclusive * n_words / n_fa_words))
        i1 = max(i1, i0 + 1) if word_end_exclusive > word_start else i0
        i0 = max(0, min(i0, n_words))
        i1 = max(i0, min(i1, n_words))

    if i0 >= n_words:
        return ""

    start = spans[i0][0]
    end = spans[i1 - 1][1]
    chunk = raw[start:end]
    if i1 < n_words:
        between = raw[end : spans[i1][0]]
        trail = re.match(r"^\s*([.!?;:]+)\s*", between)
        if trail:
            chunk = raw[start : end] + trail.group(1)

    return format_transcript(chunk.strip())


def _fa_sanitize_words_for_mms_fa(text_norm: str, *, allowed_chars: set[str]) -> List[str]:
    t = str(text_norm or "").strip().lower()
    if not t:
        return []
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []

    out: List[str] = []
    for w in t.split():
        w2 = "".join(ch for ch in w if ch in allowed_chars)
        if w2:
            out.append(w2)
    return out


def expand_long_examples_mms_fa(
    dataset: Any,
    *,
    audio_col: str,
    text_col: str,
    chunk_seconds: float,
    fa_ctx: Dict[str, Any],
    split_label: str = "",
    label_col: str = "clean_transcription",
) -> Any:
    """
    Split long rows into word-aligned segments using MMS_FA.

    ``text_col`` is used only for forced alignment (e.g. ``__text_norm`` when May6 QC is on).
    Training labels are written to ``label_col`` (default ``clean_transcription``) by slicing the
    original label text to each segment's FA word span. ``transcription`` is sliced the same way
    when present.
    """
    n_before = len(dataset)
    fa_model = fa_ctx["model"]
    fa_tokenizer = fa_ctx["tokenizer"]
    fa_aligner = fa_ctx["aligner"]
    fa_sr = int(fa_ctx["sample_rate"])
    fa_device = str(fa_ctx["device"])
    fa_allowed_chars = set(fa_ctx["allowed_chars"])

    if label_col not in dataset.column_names:
        raise ValueError(
            f"MMS_FA chunking requires label column {label_col!r}; "
            f"have {dataset.column_names}"
        )

    stats = {"unchanged": 0, "expanded": 0, "fa_fallback": 0, "fa_sanitize_fallback": 0}

    def expand_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        out_audio: List[Dict[str, Any]] = []
        out_fa_text: List[str] = []
        out_label: List[str] = []
        out_transcription: List[str] = []
        out_duration: List[float] = []
        has_transcription = "transcription" in batch
        transcriptions = batch.get("transcription") or [""] * len(batch[text_col])
        labels = batch.get(label_col) or [""] * len(batch[text_col])

        for aud, fa_txt, label_txt, txt_raw in zip(
            batch[audio_col], batch[text_col], labels, transcriptions
        ):
            arr = np.asarray(aud["array"], dtype=np.float32)
            sr = int(aud["sampling_rate"])
            t = str(fa_txt or "").strip()
            label_full = str(label_txt or "").strip()
            if not t or not label_full:
                continue
            dur = len(arr) / sr if sr > 0 else 0.0
            if dur <= float(chunk_seconds):
                out_audio.append({"array": arr, "sampling_rate": sr})
                out_fa_text.append(t)
                out_label.append(label_full)
                out_transcription.append(str(txt_raw or ""))
                out_duration.append(float(dur))
                stats["unchanged"] += 1
                continue

            words = _fa_sanitize_words_for_mms_fa(t, allowed_chars=fa_allowed_chars)
            if not words:
                out_audio.append({"array": arr, "sampling_rate": sr})
                out_fa_text.append(t)
                out_label.append(label_full)
                out_transcription.append(str(txt_raw or ""))
                out_duration.append(float(dur))
                stats["fa_sanitize_fallback"] += 1
                continue

            wt = _forced_align_word_times_mms_fa(
                waveform_1d=arr,
                sampling_rate=sr,
                transcript_words=words,
                fa_model=fa_model,
                fa_tokenizer=fa_tokenizer,
                fa_aligner=fa_aligner,
                fa_sample_rate=fa_sr,
                device=fa_device,
            )
            segs = _segment_by_word_times(wt, chunk_seconds=float(chunk_seconds))
            if not segs:
                out_audio.append({"array": arr, "sampling_rate": sr})
                out_fa_text.append(t)
                out_label.append(label_full)
                out_transcription.append(str(txt_raw or ""))
                out_duration.append(float(dur))
                stats["fa_fallback"] += 1
                continue

            stats["expanded"] += 1
            n_fa = len(words)
            raw_full = str(txt_raw or "")
            for s0, s1, w0, w1 in segs:
                seg_fa = _slice_text_by_fa_word_span(t, word_start=w0, word_end_exclusive=w1, n_fa_words=n_fa)
                seg_label = _slice_text_by_fa_word_span(
                    label_full, word_start=w0, word_end_exclusive=w1, n_fa_words=n_fa
                )
                if not seg_label:
                    continue
                i0 = max(0, int(round(s0 * sr)))
                i1 = min(len(arr), int(round(s1 * sr)))
                if i1 <= i0:
                    continue
                seg_arr = arr[i0:i1]
                seg_dur = len(seg_arr) / sr if sr > 0 else 0.0
                out_audio.append({"array": seg_arr, "sampling_rate": sr})
                out_fa_text.append(seg_fa or seg_label)
                out_label.append(seg_label)
                seg_raw = _slice_text_by_fa_word_span(
                    raw_full, word_start=w0, word_end_exclusive=w1, n_fa_words=n_fa
                )
                out_transcription.append(seg_raw if seg_raw else seg_label)
                out_duration.append(float(seg_dur))

        out: Dict[str, Any] = {
            audio_col: out_audio,
            label_col: out_label,
            "audio_duration": out_duration,
        }
        if text_col != label_col:
            out[text_col] = out_fa_text
        if has_transcription:
            out["transcription"] = out_transcription
        return out

    keep_cols = [audio_col, label_col]
    if text_col != label_col:
        keep_cols.append(text_col)
    if "transcription" in dataset.column_names:
        keep_cols.append("transcription")
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    dataset = dataset.map(
        expand_batch,
        batched=True,
        batch_size=8,
        remove_columns=drop_cols,
        desc=f"mms_fa_chunk {split_label}".strip(),
    )

    n_after = len(dataset)
    label = f"[{split_label}] " if split_label else ""
    logger.info(
        "%sMMS_FA chunk: %d → %d rows | unchanged=%d expanded_src=%d fa_fallback=%d sanitize_fallback=%d",
        label,
        n_before,
        n_after,
        stats["unchanged"],
        stats["expanded"],
        stats["fa_fallback"],
        stats["fa_sanitize_fallback"],
    )
    return dataset
