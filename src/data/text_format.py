# src/data/text_format.py — Swahili transcript formatting for training labels and decode post-process.
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

# Characters treated as normal sentence punctuation (not "weird" in QC).
SENTENCE_PUNCT_CHARS = ".,?!:;"

# Conservative oral → written normalizations (optional; off by default).
_ORAL_WORD_FIXES: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bkwasababu\b", re.IGNORECASE), "kwa sababu"),
    (re.compile(r"\bkunaa\b", re.IGNORECASE), "kuna"),
    (re.compile(r"\bpiya\b", re.IGNORECASE), "pia"),
    (re.compile(r"\bsanaa\b", re.IGNORECASE), "sana"),
]

_GLUE_AFTER_SENTENCE_END = re.compile(
    r"([.!?])([A-Za-z\u00c0-\u024f])"
)
_GLUE_AFTER_COMMA = re.compile(
    r",([A-Za-z\u00c0-\u024f])"
)
_MULTI_SPACE = re.compile(r"\s+")
_SPACED_PUNCT_RUN = re.compile(r"(\s+[.!?]){2,}")

# Oral Swahili discourse markers: insert a comma when missing (training + optional decode).
_DISCOURSE_COMMA_BEFORE = re.compile(
    r"(?<![.,!?;:\s])\s+("
    r"lakini|vile vile|kwasababu|kwa sababu|kwa mfano|kwa mfanoo|na pia|ningependa|"
    r"je|sawa|asante|labda|namaanisha|unamaanisha|nimeona|ninafikiria"
    r")\b",
    re.IGNORECASE,
)


def enrich_discourse_punctuation(text: str) -> str:
    """
    Insert commas before common Swahili clause boundaries when the source has none.

    Does not remove existing punctuation. Safe to run on labels and on decode output.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    s = _DISCOURSE_COMMA_BEFORE.sub(r", \1", s)
    return _MULTI_SPACE.sub(" ", s).strip()


def format_transcript(
    text: str,
    *,
    normalize_oral: bool = False,
    discourse_commas: bool = False,
) -> str:
    """
    Normalize spacing and light punctuation layout for training / references.

    - NFC unicode
    - Space after sentence-ending . ! ? when glued to the next word (``hiyo.Aina``)
    - Space after comma when glued to a following letter
    - Collapse repeated whitespace; trim stray spaced punctuation runs
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text).strip())
    if not s:
        return ""

    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = _GLUE_AFTER_SENTENCE_END.sub(r"\1 \2", s)
    s = _GLUE_AFTER_COMMA.sub(r", \1", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    s = _SPACED_PUNCT_RUN.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()

    if normalize_oral:
        for pat, repl in _ORAL_WORD_FIXES:
            s = pat.sub(repl, s)
        s = _MULTI_SPACE.sub(" ", s).strip()

    if discourse_commas:
        s = enrich_discourse_punctuation(s)

    return s


def format_decode_output(text: str, *, discourse_commas: bool = False) -> str:
    """Post-decode cleanup (chunk joins, CTC spacing)."""
    return format_transcript(text, normalize_oral=False, discourse_commas=discourse_commas)


def ensure_chunk_label_terminal_period(label: str) -> str:
    """Append ``.`` when a MMS-FA chunk label has no sentence-ending punctuation."""
    s = str(label or "").strip()
    if not s or s[-1] in SENTENCE_PUNCT_CHARS:
        return s
    return s + "."


def join_chunk_predictions(parts: List[str]) -> str:
    """
    Join non-overlapping decode segments.

    Inserts a period between chunks when the previous segment does not end with sentence
    punctuation, then runs :func:`format_decode_output`.
    """
    cleaned = [str(p).strip() for p in parts if str(p).strip()]
    if not cleaned:
        return ""
    out = cleaned[0]
    for nxt in cleaned[1:]:
        prev = out.rstrip()
        if prev and prev[-1] not in SENTENCE_PUNCT_CHARS:
            out = f"{prev}. {nxt.lstrip()}"
        else:
            out = f"{prev} {nxt.lstrip()}"
    return format_decode_output(out)


def format_transcription_batch(
    batch: Dict[str, Any],
    *,
    text_col: str = "transcription",
    normalize_oral: bool = False,
    discourse_commas: bool = False,
) -> Dict[str, Any]:
    batch[text_col] = [
        format_transcript(
            t,
            normalize_oral=normalize_oral,
            discourse_commas=discourse_commas,
        )
        for t in batch[text_col]
    ]
    return batch


def punctuation_recall(reference: str, hypothesis: str) -> float:
    """
    Fraction of reference sentence-punctuation marks (.,?!:;) present in the hypothesis.
    Returns 1.0 when the reference has no such marks.
    """
    ref = str(reference or "")
    hyp = str(hypothesis or "")
    marks = [c for c in ref if c in SENTENCE_PUNCT_CHARS]
    if not marks:
        return 1.0
    hit = sum(1 for c in marks if c in hyp)
    return float(hit) / float(len(marks))


def mean_punctuation_recall(references: List[str], hypotheses: List[str]) -> float:
    pairs = [(r, h) for r, h in zip(references, hypotheses) if str(r).strip()]
    if not pairs:
        return 1.0
    return sum(punctuation_recall(r, h) for r, h in pairs) / len(pairs)


def combined_asr_score(
    wer: float,
    cer: float,
    punct_recall: float,
    *,
    wer_weight: float = 0.4,
    cer_weight: float = 0.4,
    punct_weight: float = 0.2,
) -> float:
    """Higher is better (0–100 scale), for optional checkpoint selection."""
    w = wer_weight + cer_weight + punct_weight
    if w <= 0:
        w = 1.0
    loss = (
        wer_weight * wer + cer_weight * cer + punct_weight * (1.0 - punct_recall)
    ) / w
    return max(0.0, (1.0 - loss) * 100.0)
