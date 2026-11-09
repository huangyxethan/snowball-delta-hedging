# -*- coding: utf-8 -*-
"""Price one snowball with both engines and cross-check them. Needs no data files.

    python examples/price_snowball.py

What it shows:
  * the issue-date coupon that makes PV = 1, solved without root finding by exploiting
    the payoff's linearity in the coupon (two Monte Carlo evaluations) and, independently,
    by the Crank-Nicolson PDE engine;
  * value and delta at issue from both engines;
  * a small delta profile across spot levels from the PDE curves, including the jump
    around the knock-in barrier.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_params import MarketSnapshot            # noqa: E402
from snowball_contract import make_snowball_weekly  # noqa: E402
from pricer import solve_coupon, value, delta       # noqa: E402
from pde_engine import pde_solve_coupon, pde_value, pde_delta  # noqa: E402

# A weekday calendar is enough for a pricing demo (observation dates only need a grid).
calendar_days = pd.bdate_range("2024-01-01", "2027-12-31")
issue = calendar_days[calendar_days.weekday == 0][0]          # first Monday in the calendar
snap = MarketSnapshot(date=issue, s=6000.0, r=0.02, repo=-0.08, sigma=0.22)

# Two-year weekly-grid snowball: knock-in 70 %, knock-out 103 %, first KO observation in
# week 12, then every 4 weeks on Fridays.
terms = make_snowball_weekly(calendar_days, issue, s0=snap.s,
                             ki_pct=0.70, ko_pct=1.03, tenor_weeks=104,
                             first_ko_week=12, ko_interval_weeks=4)
print(f"issue {issue.date()}  S0={snap.s:.0f}  r={snap.r:.2%}  repo={snap.repo:+.2%}  "
      f"sigma={snap.sigma:.0%}  KO obs dates={len(terms.ko_obs_dates)}  "
      f"maturity={terms.maturity_date.date()}")

tic = time.perf_counter()
c_mc = solve_coupon(terms, snap, method="mc")
t_mc = time.perf_counter() - tic
tic = time.perf_counter()
c_pde = pde_solve_coupon(terms, snap)
t_pde = time.perf_counter() - tic
print(f"\ncoupon  MC  {c_mc:.4%}  ({t_mc:.1f}s)      PDE {c_pde:.4%}  ({t_pde:.1f}s)      "
      f"diff {abs(c_mc - c_pde) * 1e4:.1f} bp")

priced = terms.with_coupon(c_pde)
v_mc, v_pde = value(priced, snap), pde_value(priced, snap)
d_mc, d_pde = delta(priced, snap), pde_delta(priced, snap)
print(f"value   MC  {v_mc:.6f}            PDE {v_pde:.6f}   (both should be ~1 at the PDE coupon)")
print(f"delta   MC  {d_mc * snap.s:+.4f}            PDE {d_pde * snap.s:+.4f}   "
      f"(dV/dS x S: futures exposure as a fraction of notional)")

print("\nPDE delta profile at issue (dV/dS x S, exposure as a fraction of notional):")
for s_mult in (0.68, 0.70, 0.72, 0.80, 0.90, 1.00, 1.03, 1.10):
    s = snap.s * s_mult
    d = pde_delta(priced, replace(snap, s=s))
    print(f"  S = {s_mult:4.0%} of S0   exposure = {d * s:+.3f}")
