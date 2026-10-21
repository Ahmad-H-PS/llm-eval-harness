"""Model loading, inference and output parsing."""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import clean_text

POS = {"positive"}
NEG = {"negative"}
NEGATION = {"not", "no", "isn", "never"}


def parse_label(raw: str) -> Optional[int]:
    """Return 1 (positive), 0 (negative), or None (unparseable).

    Looks in the first 6 words; a negated label counts as unparseable.
    """
    words = re.findall(r"[a-z]+", raw.lower())[:6]
    for i, w in enumerate(words):
        if w in POS or w in NEG:
            if i > 0 and words[i - 1] in NEGATION:
                return None
            return 1 if w in POS else 0
    return None


def load_model(hf_id: str, device: str | None = None):
    """Load tokenizer and model; resets the CUDA peak-memory counter first."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype).to(device)
    model.eval()
    return tok, model


def unload_model(model) -> None:
    """Free GPU memory so the next model fits on a 4 GB card."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_classification(
    tok,
    model,
    ds: Dataset,
    template: str,
    text_key: str = "text",
    label_key: str = "label",
    detokenize: bool = False,
    max_new_tokens: int = 16,
    parser: Callable[[str], Optional[int]] = parse_label,
) -> tuple[list[dict], dict]:
    """Run the model over ds. Returns (rows, stats).

    Each row: id, text (cleaned), prompt, gold, raw output, parsed pred.
    stats: latency_s_per_ex and peak_vram_gb (None on CPU).
    """
    rows = []
    start = time.time()
    for i, ex in enumerate(tqdm(ds, total=len(ds))):
        text = clean_text(ex[text_key], detokenize=detokenize)
        prompt = template.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        raw = tok.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        rows.append(
            {"id": i, "text": text, "prompt": prompt, "gold": ex[label_key],
             "raw": raw, "pred": parser(raw)}
        )
    elapsed = time.time() - start

    stats = {
        "latency_s_per_ex": round(elapsed / len(rows), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
    }
    return rows, stats