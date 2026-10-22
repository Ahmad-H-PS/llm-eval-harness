"""Command-line entry point: python -m harness.cli --config configs/default.yaml"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import run_all
from .plots import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM evaluation harness.")
    parser.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    parser.add_argument("--out", default="results/results.csv", help="where to write the results CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = run_all(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plot_results(df, args.chart)

    print("\n" + df.to_string(index=False))
    print(f"\nSaved {len(df)} rows to {out} and chart to {args.chart}")


if __name__ == "__main__":
    main()