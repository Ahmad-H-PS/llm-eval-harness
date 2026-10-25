"""Model loading, inference and output parsing."""
from __future__ import annotations

import gc
import re
import time
from collections.abc import Callable

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import clean_text, normalize_arabic

POS = {"positive"}
NEG = {"negative"}
NEGATION = {"not", "no", "isn", "never"}
AR_POS = {"ايجابي", "ايجابية", "ايجابيه", "positive"}
AR_NEG = {"سلبي", "سلبية", "سلبيه", "negative"}
AR_NEGATION = {"غير", "ليس", "ليست", "لا"}


def parse_label(raw: str) -> int | None:
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

def parse_label_ar(raw: str) -> int | None:
    """Arabic version of parse_label: normalize, look in the first 6 words,
    negated labels are unparseable. English labels are also accepted."""
    words = re.findall(r"\w+", normalize_arabic(raw).lower())[:6]
    for i, w in enumerate(words):
        if w in AR_POS or w in AR_NEG:
            if i > 0 and words[i - 1] in AR_NEGATION:
                return None
            return 1 if w in AR_POS else 0
    return None


PARSERS = {"en": parse_label, "ar": parse_label_ar}


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


def free_gpu() -> None:
    """Call after `del model, tok` in the caller so the next model fits in 4 GB."""
    gc.collect()
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
    language: str = "en",
    parser: Callable[[str], int | None] | None = None,
) -> tuple[list[dict], dict]:
    """Run the model over ds. Returns (rows, stats).

    Each row: id, text (cleaned), prompt, gold, raw output, parsed pred.
    stats: latency_s_per_ex and peak_vram_gb (None on CPU).
    """
    parser = parser or PARSERS.get(language, parse_label)
    rows = []
    start = time.time()
    for i, ex in enumerate(tqdm(ds, total=len(ds))):
        text = clean_text(ex[text_key], detokenize=detokenize, language=language)
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

def run_qa(
    tok,
    model,
    ds: Dataset,
    template: str,
    max_new_tokens: int = 32,
    context_key: str = "context",
    question_key: str = "question",
    answers_key: str = "answers",
) -> tuple[list[dict], dict]:
    """Extractive-style QA. Pred = first line of the output. gold = list of answer strings."""
    rows = []
    start = time.time()
    for i, ex in enumerate(tqdm(ds, total=len(ds))):
        question = clean_text(ex[question_key])
        prompt = template.format(context=clean_text(ex[context_key]), question=question)
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
            {"id": i, "text": question, "prompt": prompt, "gold": ex[answers_key]["text"],
             "raw": raw, "pred": raw.strip().split("\n")[0].strip()}
        )
    elapsed = time.time() - start
    stats = {
        "latency_s_per_ex": round(elapsed / len(rows), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
    }
    return rows, stats