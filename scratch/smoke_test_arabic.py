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

HF_ID = "hammh0a/Hala-700M"
N = 20
SEED = 42
MAX_NEW_TOKENS = 24

PROMPT_AR = (
    "صنّف مشاعر النص التالي. أجب بكلمة واحدة فقط: «إيجابي» أو «سلبي». "
    "لا تضف أي شرح.\n\nالنص: {text}\n\nالإجابة:"
)

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
POS = {"ايجابي", "ايجابية", "ايجابيه", "positive"}
NEG = {"سلبي", "سلبية", "سلبيه", "negative"}
NEGATION = {"غير", "ليس", "ليست", "لا"}


def norm_ar(s: str) -> str:
    s = _DIACRITICS.sub("", s)
    return re.sub("[أإآ]", "ا", s)


def parse_label(raw: str):
    """Return 1 (positive), 0 (negative), or None (unparseable).
    Looks for a label in the first 6 words; negated labels are unparseable."""
    words = re.findall(r"\w+", norm_ar(raw).lower())[:6]
    for i, w in enumerate(words):
        if w in POS or w in NEG:
            if i > 0 and words[i - 1] in NEGATION:
                return None
            return 1 if w in POS else 0
    return None


# --- data ---
ds = load_dataset("komari6/ajgt_twitter_ar", split="train")
print("columns:", ds.features)  # confirm which label id means positive
ds = ds.shuffle(seed=SEED).select(range(N))

# --- model ---
torch.cuda.reset_peak_memory_stats()
tok = AutoTokenizer.from_pretrained(HF_ID)
model = AutoModelForCausalLM.from_pretrained(HF_ID, dtype=torch.float16).to("cuda")

# --- run ---
rows = []
start = time.time()
for ex in ds:
    messages = [{"role": "user", "content": PROMPT_AR.format(text=ex["text"])}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    raw = tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    rows.append({"text": ex["text"], "gold": ex["label"], "raw": raw, "pred": parse_label(raw)})
elapsed = time.time() - start

# --- save (view in VS Code; the terminal won't render Arabic RTL correctly) ---
fname = OUT_DIR / f"ajgt_test_{HF_ID.split('/')[-1]}.jsonl"
with open(fname, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- score ---
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