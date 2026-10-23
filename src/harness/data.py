"""Dataset loading and text cleaning."""
from __future__ import annotations

import re
import unicodedata

from datasets import Dataset, load_dataset

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")
_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # tashkeel, dagger alef, tatweel
_AR_ALEF = re.compile(r"[\u0622\u0623\u0625\u0671]")         # alef variants -> bare alef

AR_STEPS_DEFAULT = ("diacritics", "alef")
_AR_STEPS = {"diacritics", "alef", "ya", "ta_marbuta"}

# Undo SST-2-style tokenization: "does n't" -> "doesn't", "movie ." -> "movie."
_DETOK = [
    (re.compile(r"\s+([.,!?;:])"), r"\1"),
    (re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r"-lrb-\s*"), "("),
    (re.compile(r"\s*-rrb-"), ")"),
]


def clean_text(text: str, detokenize: bool = False, language: str = "en",
               ar_steps: tuple[str, ...] | None = None) -> str:
    """Normalize raw text before it goes into a prompt.

    Steps: Unicode NFKC normalization, remove zero-width characters,
    strip HTML tags and URLs, collapse runs of whitespace, trim the ends.
    If detokenize=True, also rejoin pre-tokenized punctuation and
    contractions (useful for SST-2).
    If language == "ar", also apply Arabic normalization (ar_steps, default
    diacritics + alef). Other languages are left untouched.

    Does not change case or remove words, so the meaning is preserved.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    if detokenize:
        for pattern, repl in _DETOK:
            text = pattern.sub(repl, text)
    if language == "ar":
        text = normalize_arabic(text, ar_steps or AR_STEPS_DEFAULT)
    return _WS.sub(" ", text).strip()


def build_prompt(template: str, text: str, **clean_kwargs) -> str:
    """Clean the raw text, then fill it into the prompt template."""
    return template.format(text=clean_text(text, **clean_kwargs))


def load_sample(dataset_id: str, split: str, n_samples: int, seed: int) -> Dataset:
    """Load a split and take a fixed, reproducible random slice."""
    ds = load_dataset(dataset_id, split=split)
    return ds.shuffle(seed=seed).select(range(n_samples))


def normalize_arabic(text: str, steps: tuple[str, ...] = AR_STEPS_DEFAULT) -> str:
    """Apply optional Arabic normalization steps.

    diacritics: remove tashkeel, dagger alef and tatweel (elongation)
    alef:       map alef variants (with hamza/madda, wasla) to bare alef
    ya:         alef maqsura -> ya        (off by default, can change meaning)
    ta_marbuta: ta marbuta -> ha          (off by default, can change meaning)
    """
    unknown = set(steps) - _AR_STEPS
    if unknown:
        raise ValueError(f"unknown Arabic normalization steps: {sorted(unknown)}")
    if "diacritics" in steps:
        text = _AR_DIACRITICS.sub("", text)
    if "alef" in steps:
        text = _AR_ALEF.sub("\u0627", text)
    if "ya" in steps:
        text = text.replace("\u0649", "\u064A")
    if "ta_marbuta" in steps:
        text = text.replace("\u0629", "\u0647")
    return text