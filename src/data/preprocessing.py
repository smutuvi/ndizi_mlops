# src/data/preprocessing.py
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Union

from transformers import Wav2Vec2BertProcessor, Wav2Vec2Processor

ASRProcessor = Union[Wav2Vec2Processor, Wav2Vec2BertProcessor]

_ACCENT_REPLACEMENTS = {
    "á": "a",
    "à": "a",
    "â": "a",
    "ä": "a",
    "ã": "a",
    "å": "a",
    "ā": "a",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ë": "e",
    "ē": "e",
    "í": "i",
    "ì": "i",
    "î": "i",
    "ï": "i",
    "ī": "i",
    "ó": "o",
    "ò": "o",
    "ô": "o",
    "ö": "o",
    "õ": "o",
    "ō": "o",
    "ú": "u",
    "ù": "u",
    "û": "u",
    "ü": "u",
    "ū": "u",
    "ç": "c",
    "ñ": "n",
    "ÿ": "y",
}


DEFAULT_CTC_CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz0123456789 .,?!-':/%()"


def clean_text_batch(
    batch: Dict[str, Any],
    allowed_chars: str = DEFAULT_CTC_CHARACTER_SET,
    apply_accent_replacements: bool = True,
    *,
    lowercase: bool = True,
) -> Dict[str, Any]:
    allowed_char_set = set(allowed_chars)
    batch["clean_transcription"] = [
        _clean_text_with_char_set(
            str(t or ""),
            allowed_char_set,
            apply_accent_replacements,
            lowercase=lowercase,
        )
        for t in batch["transcription"]
    ]
    return batch


def _clean_text_with_char_set(
    text: str,
    character_set: set[str],
    apply_accent_replacements: bool = True,
    *,
    lowercase: bool = True,
) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("'", "'").replace("ʼ", "'")
    if apply_accent_replacements:
        for src, tgt in _ACCENT_REPLACEMENTS.items():
            text = text.replace(src, tgt)
    text = "".join(c for c in text if c.lower() in character_set)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def hub_ctc_identity_clean_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Strip only; preserve case and punctuation (Whisper / formatted labels)."""
    batch["clean_transcription"] = [str(t or "").strip() for t in batch["transcription"]]
    return batch


def hub_ctc_label_charset(pretrained_model_path: str) -> str:
    """
    Build a ``character_set`` string from a Hub CTC tokenizer (e.g. badrex Swahili).

    Hub ASR models only support their pretrained alphabet; punctuation in formatted
    transcripts must be stripped before label encoding.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(pretrained_model_path)
    chars: list[str] = []
    for token, _idx in sorted(tok.get_vocab().items(), key=lambda item: item[1]):
        if len(token) != 1:
            continue
        if token == "|" or (token.startswith("<") and token.endswith(">")):
            continue
        chars.append(token)
    if " " not in chars:
        chars.append(" ")
    return "".join(chars)


# May 6 (cleaned_ndizi_may_6.py) text normalization for QC gates only — not training labels.
_NUM_MAP_MAY6 = {
    "0": "sifuri",
    "1": "moja",
    "2": "mbili",
    "3": "tatu",
    "4": "nne",
    "5": "tano",
    "6": "sita",
    "7": "saba",
    "8": "nane",
    "9": "tisa",
}


def normalize_numbers_simple_may6(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tok = match.group(0)
        return " ".join(_NUM_MAP_MAY6.get(ch, ch) for ch in tok)

    return re.sub(r"\b\d+\b", repl, text)


def normalize_text_may6(
    text: str,
    *,
    normalize_numbers: bool = True,
    drop_punct: bool = True,
    unicode_nfc: bool = False,
) -> str:
    """Match ``normalize_text`` in ``cleaned_ndizi_may_6.py`` (QC / audit text, not Whisper labels)."""
    if text is None:
        return ""
    t = str(text).strip()
    if unicode_nfc:
        t = unicodedata.normalize("NFC", t)
    t = t.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.lower()
    t = re.sub(r"\[(noise|laughter|music|cough|silence)\]", " ", t, flags=re.IGNORECASE)
    if normalize_numbers:
        t = normalize_numbers_simple_may6(t)
    if drop_punct:
        t = re.sub(r"[^\w\s'-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def add_may6_text_norm_batch(
    batch: Dict[str, Any],
    *,
    text_col: str = "transcription",
    normalize_numbers: bool = True,
    drop_punct: bool = True,
    unicode_nfc: bool = False,
) -> Dict[str, Any]:
    """Add ``__text_norm`` for QC (same role as ``add_text_norm`` in cleaned_ndizi_may_6.py)."""
    batch["__text_norm"] = [
        normalize_text_may6(
            t,
            normalize_numbers=normalize_numbers,
            drop_punct=drop_punct,
            unicode_nfc=unicode_nfc,
        )
        for t in batch[text_col]
    ]
    return batch


def wer_normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def prepare_dataset_batch(batch: Dict[str, List[Any]], processor: ASRProcessor) -> Dict[str, List[Any]]:
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    sampling_rate = batch["audio"][0]["sampling_rate"]
    features = processor(audio_arrays, sampling_rate=sampling_rate)
    if isinstance(processor, Wav2Vec2BertProcessor):
        key = "input_features"
        batch[key] = features.input_features
    else:
        key = "input_values"
        batch[key] = features.input_values
    batch["length"] = [len(f) for f in batch[key]]
    labels = processor(text=batch["clean_transcription"]).input_ids
    batch["labels"] = labels
    return batch
