# -*- coding: utf-8 -*-
"""Run the daily index-construction backtest on the synthetic data set.

    python scripts/make_synthetic_data.py      # once
    python examples/run_daily_backtest.py

Four issuance slots roll two-year weekly-grid snowballs one week apart; every day each
live leg's delta is re-read from its PDE curve at the day's close, the aggregate exposure
is held in front-month IC futures with a five-day roll, and the NAV accumulates the
futures mark-to-market net of fees. The first run solves every PDE curve and takes a few
minutes; the curves are cached under data/pde_cache/ so later runs take seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest import run_backtest  # noqa: E402

START, END = "2022-01-03", "2023-12-29"
OUT = Path(__file__).resolve().parent / "output"

if __name__ == "__main__":
    tic = time.perf_counter()
    r = run_backtest(start_date=START, end_date=END, delta_cap=2.0)
    print(f"backtest {START} -> {END} done in {time.perf_counter() - tic:.0f}s\n")

    print(r.summary().to_string())

    cols = ["slot", "issue_date", "end_date", "s0", "coupon", "scenario_label",
            "life_years", "truncated"]
    legs = r.legs[cols].copy()
    legs["issue_date"] = legs["issue_date"].dt.date
    legs["end_date"] = legs["end_date"].dt.date
    print("\nlegs:")
    print(legs.to_string(float_format=lambda x: f"{x:.4f}"))

    OUT.mkdir(exist_ok=True)
    r.daily[["nav", "coupon_line", "n_live"]].to_csv(OUT / "daily_backtest.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        r.daily["nav"].plot(ax=ax, label="strategy NAV")
        r.daily["coupon_line"].plot(ax=ax, label="accrued coupon line")
        ax.set_title(f"Synthetic data, {START} to {END}: 4-slot snowball ladder, delta-hedged in IC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "daily_backtest.png", dpi=120)
        print(f"\nwrote {OUT / 'daily_backtest.csv'} and daily_backtest.png")
    except ImportError:
        print(f"\nwrote {OUT / 'daily_backtest.csv'} (matplotlib not installed, no plot)")
