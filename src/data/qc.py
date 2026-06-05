# src/data/qc.py — multi-gate audio/text quality control (ported from cleaned_ndizi_may_6.py).
#
# Activated via WhisperTrainingConfig / ASRConfig field:
#   apply_data_qc: bool = False
# Optional long-audio chunking (only when apply_data_qc is true):
#   qc_chunk_long_with_mms_fa: bool = False
# Optional May 6 text normalization for QC gates only (training labels unchanged):
#   qc_use_may6_text_norm: bool = False
#
# Thresholds live in QCConfig and can be overridden per-run via config JSON fields
# prefixed with "qc_" (e.g. "qc_min_dur": 1.5), plus training_config_raw on the config object.
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Swahili language-likelihood helpers
# ---------------------------------------------------------------------------

_SWAHILI_FUNCTION_WORDS = {
    "na", "ya", "kwa", "katika", "kama", "hii", "hiyo", "hizo", "hapa", "pale",
    "ni", "sio", "ndio", "kuwa", "yenye", "bila", "sana", "tu", "pia", "lakini",
    "hivyo", "ambayo", "hapo", "huku", "kile", "mimi", "sisi", "yeye", "wao",
    "wangu", "wetu", "wake", "zao", "kwangu", "kwetu", "kwake", "kwenu",
}
_SW_PREFIX_RE = re.compile(r"^(ku|wa|ni|ya|ki|vi|li|si|ha|na)\w+$")


# ---------------------------------------------------------------------------
# QCConfig — all thresholds in one place; JSON-overridable via "qc_*" prefix
# ---------------------------------------------------------------------------

@dataclass
class QCConfig:
    # Duration (seconds)
    min_dur: float = 1.0
    max_dur: float = 30.0
    # Audio signal quality
    min_rms_dbfs: float = -45.0
    clip_thresh: float = 0.999
    max_clipping_rate: float = 0.002
    # Energy-based VAD
    vad_frame_ms: float = 25.0
    vad_hop_ms: float = 10.0
    vad_db_thresh: float = -35.0
    min_speech_ratio: float = 0.25
    # Transcript text
    min_text_chars: int = 3
    max_text_chars: int = 400
    max_weird_char_ratio: float = 0.02
    max_repetition_token_prop: float = 0.35
    max_digit_ratio: float = 0.20
    min_swahili_likeness: float = 0.05
    min_tokens_for_langcheck: int = 6
    # Speaking rate plausibility
    min_words_per_sec: float = 0.5
    max_words_per_sec: float = 4.5
    min_chars_per_sec: float = 4.0
    max_chars_per_sec: float = 25.0
    # When true, standard sentence punctuation does not count toward weird_char_ratio.
    allow_sentence_punctuation: bool = True


def qc_config_from_dict(raw: Dict[str, Any]) -> QCConfig:
    """Build a QCConfig from a config dict, reading keys prefixed with 'qc_'."""
    from dataclasses import fields as dc_fields
    valid = {f.name for f in dc_fields(QCConfig)}
    kwargs = {k[3:]: v for k, v in raw.items() if k.startswith("qc_") and k[3:] in valid}
    return QCConfig(**kwargs)


def qc_config_for_training(config: Any) -> QCConfig:
    """Merge dataclass fields + JSON ``qc_*`` overrides; bump max_dur when MMS_FA chunking is on."""
    from dataclasses import asdict

    merged: Dict[str, Any] = dict(asdict(config))
    raw = getattr(config, "training_config_raw", None) or {}
    for k, v in raw.items():
        if k.startswith("qc_"):
            merged[k] = v
    qc_cfg = qc_config_from_dict(merged)
    if getattr(config, "apply_data_qc", False) and getattr(config, "qc_chunk_long_with_mms_fa", False):
        chunk_s = float(getattr(config, "qc_chunk_seconds", 30.0))
        qc_cfg.max_dur = max(float(qc_cfg.max_dur), chunk_s + 0.25)
    return qc_cfg


def qc_config_from_training_json(raw: Dict[str, Any]) -> QCConfig:
    """
    Build :class:`QCConfig` from a flat training JSON/YAML dict (``qc_*`` keys only),
    including the MMS_FA ``max_dur`` bump when ``apply_data_qc`` and
    ``qc_chunk_long_with_mms_fa`` are true — same rule as :func:`qc_config_for_training`,
    for callers that only have the raw dict (e.g. batch eval).
    """
    qc_cfg = qc_config_from_dict(raw)
    if bool(raw.get("apply_data_qc")) and bool(raw.get("qc_chunk_long_with_mms_fa")):
        chunk_s = float(raw.get("qc_chunk_seconds", 30.0))
        qc_cfg.max_dur = max(float(qc_cfg.max_dur), chunk_s + 0.25)
    return qc_cfg


# ---------------------------------------------------------------------------
# Low-level signal metrics
# ---------------------------------------------------------------------------

def _rms_dbfs(audio: np.ndarray) -> float:
    eps = 1e-12
    rms = float(np.sqrt(np.mean(audio * audio) + eps))
    return 20.0 * math.log10(max(rms, eps))


def _clipping_rate(audio: np.ndarray, clip_thresh: float = 0.999) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= clip_thresh))


def _energy_vad_speech_ratio(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    db_thresh: float = -35.0,
) -> float:
    if len(audio) == 0 or sr <= 0:
        return 0.0
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if frame <= 0 or hop <= 0 or len(audio) < frame:
        return 1.0 if _rms_dbfs(audio) > db_thresh else 0.0
    eps = 1e-12
    n_frames = 1 + (len(audio) - frame) // hop
    speech_frames = sum(
        1
        for i in range(n_frames)
        if 20.0 * math.log10(max(float(np.sqrt(np.mean(audio[i * hop: i * hop + frame] ** 2) + eps)), eps)) > db_thresh
    )
    return float(speech_frames) / float(n_frames)


def _char_stats(text: str, *, allow_sentence_punctuation: bool = True) -> Dict[str, float]:
    if not text:
        return {"alpha_ratio": 0.0, "digit_ratio": 0.0, "weird_ratio": 0.0}
    n = len(text)
    digits = sum(ch.isdigit() for ch in text)
    from src.data.text_format import SENTENCE_PUNCT_CHARS

    extra_ok = set(SENTENCE_PUNCT_CHARS) if allow_sentence_punctuation else set()
    weird = sum(
        1
        for ch in text
        if not (ch.isalnum() or ch.isspace() or ch in ("'", "-", "_") or ch in extra_ok)
    )
    return {"digit_ratio": digits / n, "weird_ratio": weird / n}


def _repetition_score(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) / len(toks)


def _swahili_likeness(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _SWAHILI_FUNCTION_WORDS)
    prefix_hits = sum(1 for t in toks if _SW_PREFIX_RE.match(t))
    return (hits + 0.25 * prefix_hits) / len(toks)


# ---------------------------------------------------------------------------
# Per-example gate (first fail wins)
# ---------------------------------------------------------------------------

def check_example(
    audio_array: np.ndarray,
    sr: int,
    text_norm: str,
    cfg: QCConfig,
    *,
    audio_only: bool = False,
) -> Tuple[bool, str]:
    """Return (keep, reason). reason is 'ok' when kept, or the failing gate name."""
    dur = len(audio_array) / sr if sr > 0 else 0.0

    if dur < cfg.min_dur:
        return False, "dur_low"
    if dur > cfg.max_dur:
        return False, "dur_high"
    if _rms_dbfs(audio_array) < cfg.min_rms_dbfs:
        return False, "rms_low"
    if _clipping_rate(audio_array, cfg.clip_thresh) > cfg.max_clipping_rate:
        return False, "clip_high"
    if _energy_vad_speech_ratio(audio_array, sr, cfg.vad_frame_ms, cfg.vad_hop_ms, cfg.vad_db_thresh) < cfg.min_speech_ratio:
        return False, "vad_low"

    if audio_only:
        return True, "ok"

    if len(text_norm) < cfg.min_text_chars:
        return False, "text_short"
    if len(text_norm) > cfg.max_text_chars:
        return False, "text_long"

    stats = _char_stats(text_norm, allow_sentence_punctuation=cfg.allow_sentence_punctuation)
    if stats["weird_ratio"] > cfg.max_weird_char_ratio:
        return False, "weird_high"
    if _repetition_score(text_norm) > cfg.max_repetition_token_prop:
        return False, "repeat_high"
    if stats["digit_ratio"] > cfg.max_digit_ratio:
        return False, "digit_high"

    toks = text_norm.split()
    if len(toks) >= cfg.min_tokens_for_langcheck and _swahili_likeness(text_norm) < cfg.min_swahili_likeness:
        return False, "swahili_low"

    wps = len(toks) / dur if dur > 0 else 0.0
    cps = len(text_norm) / dur if dur > 0 else 0.0
    if wps < cfg.min_words_per_sec:
        return False, "wps_low"
    if wps > cfg.max_words_per_sec:
        return False, "wps_high"
    if cps < cfg.min_chars_per_sec:
        return False, "cps_low"
    if cps > cfg.max_chars_per_sec:
        return False, "cps_high"

    return True, "ok"


# ---------------------------------------------------------------------------
# Dataset-level filter
# ---------------------------------------------------------------------------

def apply_qc_filter(
    dataset: Any,
    audio_col: str,
    text_col: str,
    cfg: QCConfig,
    split_label: str = "",
    *,
    audio_only: bool = False,
) -> Any:
    """
    Filter a HuggingFace Dataset in-place using the multi-gate QC pipeline.
    Logs a first-fail histogram and the number of rows dropped.

    When ``audio_only`` is true, only duration / level / VAD gates run (for MMS_FA chunks).
    """
    n_before = len(dataset)
    counters: Counter = Counter()
    mode = "audio_only" if audio_only else "full"
    desc = f"qc_filter {split_label} ({mode})".strip()

    def _keep(ex: Dict[str, Any]) -> bool:
        audio = ex[audio_col]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or 16000)
        text = str(ex.get(text_col) or ex.get("clean_transcription") or "").strip()
        keep, reason = check_example(arr, sr, text, cfg, audio_only=audio_only)
        counters[reason] += 1
        return keep

    dataset = dataset.filter(_keep, desc=desc)
    n_after = len(dataset)
    dropped = n_before - n_after

    label = f"[{split_label}] " if split_label else ""
    logger.info(
        "%sQC filter: %d → %d rows (dropped %d / %.1f%%)",
        label, n_before, n_after, dropped, 100.0 * dropped / max(n_before, 1),
    )
    keys = sorted(k for k in counters if k != "ok") + (["ok"] if "ok" in counters else [])
    for k in keys:
        v = counters[k]
        logger.info("  %-14s  %6d  (%.1f%%)", k, v, 100.0 * v / max(n_before, 1))

    return dataset
