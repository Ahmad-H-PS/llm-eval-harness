"""Run every model on every task and collect one results row each."""
from __future__ import annotations

import pandas as pd
import torch

from .data import load_sample
from .metrics import score_predictions
from .models import free_gpu, load_model, run_classification
from .predictions import save_predictions


def run_all(cfg: dict, pred_dir: str = "results/predictions") -> pd.DataFrame:
    """Loop models x tasks from the config. Returns one row per (model, task)."""
    # Load all samples first so a bad dataset id fails before any model loads.
    samples = {
        t["name"]: load_sample(t["dataset"], t["split"], t["n_samples"], cfg["seed"])
        for t in cfg["tasks"]
    }

    results = []
    for m in cfg["models"]:
        print(f"\n=== {m['name']} ({m['hf_id']}) ===")
        tok, model = load_model(m["hf_id"])
        try:
            for task in cfg["tasks"]:
                print(f"task: {task['name']}")
                if torch.cuda.is_available():
                    # resets the peak to the current allocation (the weights),
                    # so the number is per model + task, not cumulative
                    torch.cuda.reset_peak_memory_stats()
                rows, stats = run_classification(
                    tok, model, samples[task["name"]], task["prompt"],
                    text_key=task["text_key"], label_key=task["label_key"],
                    detokenize=task["detokenize"],
                    max_new_tokens=task["max_new_tokens"],
                )
                save_predictions(pred_dir, m, task, rows, stats)
                results.append({
                    "model": m["name"],
                    "hf_id": m["hf_id"],
                    "task": task["name"],
                    "language": task["language"],
                    **score_predictions(rows),
                    **stats,
                })
        finally:
            del model, tok
            free_gpu()  # so the next model fits in 4 GB
    return pd.DataFrame(results)