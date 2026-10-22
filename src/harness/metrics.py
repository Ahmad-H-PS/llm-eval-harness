"""Scoring. Unparseable outputs count as wrong, never dropped."""
from __future__ import annotations

import re
import string
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


def score_predictions(rows: list[dict]) -> dict:
    """Accuracy, macro-F1 and unparseable rate for binary classification rows.

    An unparseable prediction (None) is scored as wrong by flipping the gold
    label (valid for binary tasks only).
    """
    y_true = [r["gold"] for r in rows]
    y_pred = [r["pred"] if r["pred"] is not None else 1 - r["gold"] for r in rows]
    n_unparse = sum(r["pred"] is None for r in rows)
    return {
        "n": len(rows),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "unparseable_rate": n_unparse / len(rows),
    }

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold answer."""
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    n_same = sum((Counter(p) & Counter(g)).values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(p), n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def score_qa(rows: list[dict]) -> dict:
    """Exact match and token F1 (best over gold answers). Empty outputs score 0, never dropped."""
    n = len(rows)
    return {
        "n": n,
        "exact_match": round(sum(exact_match(r["pred"], r["gold"]) for r in rows) / n, 3),
        "f1": round(sum(max(token_f1(r["pred"], g) for g in r["gold"]) for r in rows) / n, 3),
        "empty_rate": sum(not r["pred"].strip() for r in rows) / n,
    }


def score_task(task_type: str, rows: list[dict]) -> dict:
    """Dispatch on task type and add a common `score` column for the chart."""
    if task_type == "qa":
        s = score_qa(rows)
        s["score"] = s["f1"]
    else:
        s = score_predictions(rows)
        s["score"] = s["macro_f1"]
    return s