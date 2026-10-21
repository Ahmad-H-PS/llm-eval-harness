"""Scoring. Unparseable outputs count as wrong, never dropped."""
from __future__ import annotations

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