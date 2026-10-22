"""Charts for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no GUI window
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(df: pd.DataFrame, out_path: str | Path, metric: str = "macro_f1") -> None:
    """Grouped bar chart: one group per task, one bar per model."""
    pivot = df.pivot(index="task", columns="model", values=metric)
    ax = pivot.plot.bar(figsize=(7, 4.5), rot=0, width=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model and task (single run, small samples)")
    ax.legend(title="model", loc="upper left", bbox_to_anchor=(1.01, 1))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    plt.close(ax.figure)