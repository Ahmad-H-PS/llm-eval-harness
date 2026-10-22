"""Re-score saved predictions: python -m harness.rescore"""
from __future__ import annotations

import argparse
from pathlib import Path

from .plots import plot_results
from .predictions import rescore


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild results from saved predictions.")
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--out", default="results/results.csv")
    parser.add_argument("--chart", default="results/results_chart.png")
    args = parser.parse_args()

    df = rescore(args.pred_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plot_results(df, args.chart)
    print(df.to_string(index=False))
    print(f"\nRescored {len(df)} rows from {args.pred_dir} -> {out}")


if __name__ == "__main__":
    main()