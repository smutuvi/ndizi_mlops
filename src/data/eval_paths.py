# src/data/eval_paths.py — stable audio file labels for eval CSV/JSON (Hub parquet audio).
from __future__ import annotations

import os
from typing import Any, List


def _basename_label(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    return os.path.basename(s.replace("\\", "/"))


def extract_audio_path_label(example: dict[str, Any], audio_col: str, row_idx: int = 0) -> str:
    """
    Best-effort audio identifier for eval outputs (usually a file name).

    Hub datasets often store audio as embedded bytes with no ``audio["path"]``.
    Falls back to ``case_id`` / ``id`` (Ndizi hubs) then ``row_<idx>``.
    """
    audio = example.get(audio_col)
    if isinstance(audio, dict):
        for key in ("path", "src", "source", "filename"):
            raw = audio.get(key)
            if raw is not None and str(raw).strip():
                return _basename_label(str(raw))

    for key in (
        "file_name",
        "filename",
        "audio_filename",
        "wav",
        "wav_path",
        "audio_path",
        "path",
        "file",
        "clip",
        "utterance_id",
        "audio_id",
        "case_id",
        "id",
    ):
        if key == audio_col or key not in example:
            continue
        val = example.get(key)
        if val is not None and str(val).strip():
            return _basename_label(str(val))

    return f"row_{row_idx}"


def collect_audio_path_labels(ds, audio_col: str) -> List[str]:
    """Read paths/metadata with ``Audio(decode=False)`` so bytes columns still expose ``path`` when set."""
    from datasets import Audio

    feat = ds.features.get(audio_col)
    decode_on = getattr(feat, "decode", True) if feat is not None else True
    work = ds.cast_column(audio_col, Audio(decode=False)) if decode_on else ds

    labels: List[str] = []
    for i in range(len(work)):
        labels.append(extract_audio_path_label(work[i], audio_col, i))
    return labels
