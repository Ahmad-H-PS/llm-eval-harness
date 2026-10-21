"""Dataset loading and text cleaning."""
from __future__ import annotations

import re
import unicodedata

from datasets import Dataset, load_dataset

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")

# Undo SST-2-style tokenization: "does n't" -> "doesn't", "movie ." -> "movie."
_DETOK = [
    (re.compile(r"\s+([.,!?;:])"), r"\1"),
    (re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r"-lrb-\s*"), "("),
    (re.compile(r"\s*-rrb-"), ")"),
]


def clean_text(text: str, detokenize: bool = False) -> str:
    """Normalize raw text before it goes into a prompt.

    Steps: Unicode NFKC normalization, remove zero-width characters,
    strip HTML tags and URLs, collapse runs of whitespace, trim the ends.
    If detokenize=True, also rejoin pre-tokenized punctuation and
    contractions (useful for SST-2).

    Does not change case or remove words, so the meaning is preserved.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    if detokenize:
        for pattern, repl in _DETOK:
            text = pattern.sub(repl, text)
    return _WS.sub(" ", text).strip()


def build_prompt(template: str, text: str, detokenize: bool = False) -> str:
    """Clean the raw text, then fill it into the prompt template."""
    return template.format(text=clean_text(text, detokenize=detokenize))


def load_sample(dataset_id: str, split: str, n_samples: int, seed: int) -> Dataset:
    """Load a split and take a fixed, reproducible random slice."""
    ds = load_dataset(dataset_id, split=split)
    return ds.shuffle(seed=seed).select(range(n_samples))