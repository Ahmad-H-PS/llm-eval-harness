"""Save raw model outputs and re-score them without re-running any model."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import score_task


def save_predictions(pred_dir: str | Path, model: dict, task: dict,
                     rows: list[dict], stats: dict) -> Path:
    """Write one file per (model, task): rows plus run stats."""
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{model['name']}__{task['name']}.json"
    payload = {
        "model": model["name"],
        "hf_id": model["hf_id"],
        "task": task["name"],
        "type": task["type"],
        "language": task["language"],
        "stats": stats,
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def rescore(pred_dir: str | Path) -> pd.DataFrame:
    """Rebuild the results table from saved predictions only."""
    results = []
    for path in sorted(Path(pred_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        task_type = p.get("type", "classification")  # older files have no type
        results.append({
            "model": p["model"],
            "hf_id": p["hf_id"],
            "task": p["task"],
            "type": task_type,
            "language": p["language"],
            **score_task(task_type, p["rows"]),
            **p["stats"],
        })
    if not results:
        raise FileNotFoundError(f"no prediction files in {pred_dir}")
    return pd.DataFrame(results)