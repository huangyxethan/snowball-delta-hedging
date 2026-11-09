# -*- coding: utf-8 -*-
"""Run a few execution strategies on top of the daily backtest, on synthetic minute bars.

    python scripts/make_synthetic_data.py      # once
    python examples/run_execution_demo.py

The daily index assumes "signal at the close, fill at the settlement price", which is
not tradable. The execution layer replays each day on minute bars: every strategy reads
the same PDE curves, decides its own orders, and is marked to market by a shared
accounting engine. Its NAV difference to the index decomposes exactly into

    te_qty    overnight position gap x settlement move
    te_price  traded lots x (settlement price - fill price)
    te_fee    index fees - strategy fees

Two acceptance checks are printed first: the degenerate replica (close signal, settle
fill) must reproduce the index NAV to machine precision, and the three terms must sum
to the NAV difference day by day.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest import run_backtest                                   # noqa: E402
from batch import batch_summary                                     # noqa: E402
from execution import ExecutionConfig, run_execution, validate_replica  # noqa: E402
from strategies import (MarketOnSignal, TwapLastHour, TwapTwoStage,  # noqa: E402
                        TwapPowerWeight)

INDEX_START, INDEX_END = "2022-01-03", "2023-12-29"   # daily backtest window
EXEC_START, EXEC_END = "2023-02-06", "2023-06-30"     # inside the synthetic minute window

STRATEGIES = [
    MarketOnSignal(name="market_on_signal", signal_time=135500, trade_time=140000, delta_cap=2.0),
    TwapLastHour(name="twap_1h", signal_time=135500, delta_cap=2.0),
    TwapTwoStage(name="twap_2stage", signal_time=135500, final_time=145500, recalc=True, delta_cap=2.0),
    TwapPowerWeight(name="twap_power_p1", signal_time=135500, final_time=145500, power=1.0, delta_cap=2.0),
]

if __name__ == "__main__":
    bt = run_backtest(start_date=INDEX_START, end_date=INDEX_END, delta_cap=2.0)

    dev = validate_replica(bt, start=EXEC_START, end=EXEC_END)
    print(f"replica check: max |NAV(settle replica) - NAV(index)| = {dev:.2e}  (expect ~1e-12)")

    tic = time.perf_counter()
    res = run_execution(bt, STRATEGIES, ExecutionConfig(fill_mode="bar_vwap"),
                        start=EXEC_START, end=EXEC_END)
    print(f"execution {EXEC_START} -> {EXEC_END}, {len(STRATEGIES)} strategies, "
          f"{time.perf_counter() - tic:.0f}s\n")

    # decomposition identity: cumulative (te_qty + te_price + te_fee) == nav - nav_index (up to the start offset)
    worst = 0.0
    for name, d in res.daily.items():
        lhs = (d["te_qty"] + d["te_price"] + d["te_fee"]).cumsum()
        rhs = (d["nav"] - res.nav_index)
        rhs = rhs - rhs.iloc[0] + lhs.iloc[0]
        worst = max(worst, float((lhs - rhs).abs().max()))
    print(f"decomposition check: max |sum(te_*) - (nav - nav_index)| = {worst:.2e}\n")

    pd_opts = {"float_format": lambda x: f"{x:,.4f}"}
    print(batch_summary(res).to_string(**pd_opts))

    OUT = Path(__file__).resolve().parent / "output"
    OUT.mkdir(exist_ok=True)
    res.navs.to_csv(OUT / "execution_navs.csv")
    print(f"\nwrote {OUT / 'execution_navs.csv'}")
