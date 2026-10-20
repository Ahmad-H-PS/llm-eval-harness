import json
import re
import sys
import time

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

OUT_DIR = Path(__file__).parent  # the folder this script lives in (scratch/)

sys.stdout.reconfigure(encoding="utf-8")

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-3-1b-it",
]
N = 20
SEED = 42
MAX_NEW_TOKENS = 16

PROMPT = (
    "Classify the sentiment of the following text. "
    "Answer with exactly one word: positive or negative. "
    "Do not add any explanation.\n\nText: {text}\n\nAnswer:"
)

POS = {"positive"}
NEG = {"negative"}
NEGATION = {"not", "no", "isn", "never"}


def parse_label(raw: str):
    """Return 1 (positive), 0 (negative), or None (unparseable).
    Looks for a label in the first 6 words; negated labels are unparseable."""
    words = re.findall(r"[a-z]+", raw.lower())[:6]
    for i, w in enumerate(words):
        if w in POS or w in NEG:
            if i > 0 and words[i - 1] in NEGATION:
                return None
            return 1 if w in POS else 0
    return None


# --- data (SST-2 test labels are hidden, so use the validation split) ---
ds = load_dataset("stanfordnlp/sst2", split="validation")
print("columns:", ds.features)  # confirm which label id means positive
ds = ds.shuffle(seed=SEED).select(range(N))

for hf_id in MODELS:
    print(f"\n=== {hf_id} ===")
    model = None
    try:
        torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16).to("cuda")

        rows = []
        start = time.time()
        for ex in ds:
            messages = [{"role": "user", "content": PROMPT.format(text=ex["sentence"])}]
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to("cuda")
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            raw = tok.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            rows.append({"text": ex["sentence"], "gold": ex["label"], "raw": raw, "pred": parse_label(raw)})
        elapsed = time.time() - start

        fname = OUT_DIR / f"sst2_test_{hf_id.split('/')[-1]}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        parsed = [r for r in rows if r["pred"] is not None]
        correct = sum(r["pred"] == r["gold"] for r in parsed)
        # unparseable counts as wrong for F1, so errors aren't hidden
        y_true = [r["gold"] for r in rows]
        y_pred = [r["pred"] if r["pred"] is not None else 1 - r["gold"] for r in rows]
        print(f"unparseable: {len(rows) - len(parsed)}/{len(rows)}")
        print(f"accuracy on parsed: {correct}/{len(parsed)}")
        print(f"macro-F1 (unparseable = wrong): {f1_score(y_true, y_pred, average='macro'):.2f}")
        print("pred counts:", {k: sum(r['pred'] == k for r in rows) for k in (0, 1, None)})
        print("gold counts:", {k: sum(r['gold'] == k for r in rows) for k in (0, 1)})
        print(f"time/example: {elapsed/N:.2f}s | VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    except Exception as e:
        print("FAILED:", repr(e))
    finally:
        del model
        torch.cuda.empty_cache()