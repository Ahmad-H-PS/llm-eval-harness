"""Tokenization comparison: how many tokens each model's tokenizer needs per text."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from .config import load_config
from .data import clean_text, load_sample


def token_stats(tok, texts: list[str]) -> dict:
    """Token-count statistics for a list of texts (special tokens excluded)."""
    n_tok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    n_words = [len(t.split()) for t in texts]
    n_chars = [len(t) for t in texts]
    return {
        "n": len(texts),
        "tokens_per_text": sum(n_tok) / len(texts),
        "tokens_per_word": sum(n_tok) / sum(n_words),
        "chars_per_token": sum(n_chars) / sum(n_tok),
    }


def compare(cfg: dict) -> pd.DataFrame:
    """One row per (model, task) for every classification task in the config."""
    tasks = [t for t in cfg["tasks"] if t["type"] == "classification"]
    texts = {}
    for t in tasks:
        ds = load_sample(t["dataset"], t["split"], t["n_samples"], cfg["seed"])
        texts[t["name"]] = [
            clean_text(ex[t["text_key"]], detokenize=t["detokenize"], language=t["language"])
            for ex in ds
        ]

    rows = []
    for m in cfg["models"]:
        tok = AutoTokenizer.from_pretrained(m["hf_id"])
        for t in tasks:
            rows.append({
                "model": m["name"], "task": t["name"], "language": t["language"],
                **token_stats(tok, texts[t["name"]]),
            })
    return pd.DataFrame(rows).round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tokenizers on English vs Arabic text.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/tokenization.csv")
    args = parser.parse_args()

    df = compare(load_config(args.config))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\ntokens_per_word by model and task:")
    print(df.pivot(index="model", columns="task", values="tokens_per_word").to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()