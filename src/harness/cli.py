"""Command-line entry point: python -m harness.cli --config configs/default.yaml"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .plots import plot_results
from .runlog import write_run_log
from .runner import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM evaluation harness.")
    parser.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    parser.add_argument("--out", default="results/results.csv", help="where to write the results CSV")
    parser.add_argument("--chart", default="results/results_chart.png", help="where to write the chart")
    parser.add_argument("--pred-dir", default="results/predictions", help="where to save raw predictions")
    parser.add_argument("--log-dir", default="results/run_logs", help="where to write the run log")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    cfg = load_config(args.config)
    df = run_all(cfg, pred_dir=args.pred_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plot_results(df, args.chart)
    log_path = write_run_log(cfg, args.config, df, started, log_dir=args.log_dir)

    print("\n" + df.to_string(index=False))
    print(f"\nSaved {len(df)} rows to {out}, chart to {args.chart}, run log to {log_path}")


if __name__ == "__main__":
    main()