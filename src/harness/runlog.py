"""Write a traceable record of each run: config copy, versions, seed, date, results."""
from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download

_LIBS = ["torch", "transformers", "datasets", "scikit-learn", "pandas",
         "numpy", "PyYAML", "huggingface_hub"]


def _lib_versions() -> dict:
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = None
    return out


def _cached_revision(hf_id: str) -> str | None:
    """Commit hash of the locally cached copy of a model (offline lookup)."""
    try:
        return Path(snapshot_download(hf_id, local_files_only=True)).name
    except Exception:
        return None


def write_run_log(cfg: dict, config_path: str | Path, df: pd.DataFrame,
                  started_at: datetime, log_dir: str | Path = "results/run_logs") -> Path:
    """Save one JSON log plus a copy of the config for this run. Returns the log path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")

    config_copy = log_dir / f"{run_id}_config.yaml"
    shutil.copy2(config_path, config_copy)

    log = {
        "run_id": run_id,
        "started_utc": started_at.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "seed": cfg["seed"],
        "config_copy": config_copy.name,
        "models": [
            {"name": m["name"], "hf_id": m["hf_id"],
             "cached_revision": _cached_revision(m["hf_id"])}
            for m in cfg["models"]
        ],
        "tasks": [
            {k: t[k] for k in ("name", "dataset", "split", "n_samples", "language")}
            for t in cfg["tasks"]
        ],
        "libraries": _lib_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # to_json handles NaN (e.g. VRAM on CPU) safely
        "results": json.loads(df.to_json(orient="records")),
    }
    path = log_dir / f"{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    return path