"""Load and validate the YAML config."""
from __future__ import annotations

from pathlib import Path

import yaml

_MODEL_KEYS = {"name", "hf_id"}
_TASK_KEYS = {"name", "dataset", "split", "n_samples", "language", "prompt", "metric"}
# Optional task settings; defaults match run_classification.
_TASK_DEFAULTS = {
    "type": "classification",
    "text_key": "text",
    "label_key": "label",
    "detokenize": False,
    "max_new_tokens": 16,
}
_PLACEHOLDERS = {"classification": ["text"], "qa": ["context", "question"]}


def _check_entries(entries: list, required: set[str], kind: str) -> None:
    if not entries:
        raise ValueError(f"config needs at least one {kind}")
    names = [e.get("name") for e in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate {kind} names: {names}")
    for e in entries:
        missing = required - e.keys()
        if missing:
            raise ValueError(f"{kind} '{e.get('name')}' is missing keys: {sorted(missing)}")


def load_config(path: str | Path) -> dict:
    """Read the YAML file, validate it, and fill in task defaults."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "seed" not in cfg:
        raise ValueError("config is missing 'seed'")
    _check_entries(cfg.get("models", []), _MODEL_KEYS, "model")
    _check_entries(cfg.get("tasks", []), _TASK_KEYS, "task")

    for task in cfg["tasks"]:
        for key, default in _TASK_DEFAULTS.items():
            task.setdefault(key, default)
        if task["type"] not in _PLACEHOLDERS:
            raise ValueError(f"task '{task['name']}': unknown type '{task['type']}'")
        for ph in _PLACEHOLDERS[task["type"]]:
            if "{" + ph + "}" not in task["prompt"]:
                raise ValueError(f"task '{task['name']}': prompt must contain {{{ph}}}")
    return cfg