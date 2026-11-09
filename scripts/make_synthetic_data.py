# -*- coding: utf-8 -*-
"""Generate a self-consistent synthetic data set in the exact schema the loaders expect,
so the whole pipeline (pricing, daily hedging backtest, minute-level execution backtest)
runs without any licensed market data.

Files written, relative to the repository root:

    data/raw/000905.SH.xlsx         index daily OHLC         columns: time, open, high, low, close
    data/raw/IC_daily_price.xlsx    IC futures daily, exactly 4 contracts per day
                                    columns: day, code, open, high, low, close, change,
                                             volume, amount, presettle, settle, oi
    data/raw/shibor.xlsx            1Y Shibor in percent     columns: time, value
    data/raw/csi500_index_1min.csv  index 1-minute closes    columns: date, time, close
    data/raw/IC_quote_1min.csv      IC 1-minute bars         columns: code, date, time, close, volume, value

Daily futures cover [--start, --end] on a Monday-to-Friday calendar (no holidays); the
index and Shibor start earlier (--index-start) because the volatility estimator needs a
250-day warm-up before the first futures date. Keep --end at least ten trading days after
the Monday that follows a third Friday and before the next front-month roll: the chain
builder treats contracts alive at the data end as expiring on their third Friday, which
creates one artificial roll five days before the end, and that roll must not collide with
a genuine one when a five-day roll window is used.
Minute bars cover [--minute-start, --minute-end]; on those days the daily futures
OHLC, volume and settlement price are derived from the minute bars (settle = volume-
weighted average of the last hour), so the daily and intraday layers agree exactly.

Nothing here resembles real market history: the index is a geometric Brownian motion,
the futures basis is an AR(1) discount around -8 % a year, volumes are a smooth U-shape
with noise. Contract listing follows the CFFEX rule (current month, next month, and the
two following quarter months; last trading day = third Friday of the contract month).

Usage:
    python scripts/make_synthetic_data.py                      # defaults below
    python scripts/make_synthetic_data.py --seed 11 --minute-start 2023-01-03 --minute-end 2023-06-30
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MULT = 200.0                      # IC contract multiplier (CNY per index point)
S0 = 6000.0                       # index level at the start of the sample
VOL = 0.22                        # annualised index volatility
DRIFT = 0.04                      # annualised index drift
REPO_MEAN = -0.08                 # annualised basis (negative = futures discount)
SLOT_VOLUME = (80_000, 12_000, 9_000, 3_000)    # daily lots by slot: M0, M1, Q1, Q2
SLOT_OI = (120_000, 30_000, 25_000, 8_000)

# 240 one-minute bars labelled by their END time: 09:31-11:30 and 13:01-15:00
BAR_TIMES = ([h * 10000 + m * 100 for h in (9, 10, 11) for m in range(60)
              if (h, m) >= (9, 31) and (h, m) <= (11, 30)]
             + [h * 10000 + m * 100 for h in (13, 14, 15) for m in range(60)
                if (h, m) >= (13, 1) and (h, m) <= (15, 0)])
assert len(BAR_TIMES) == 240, len(BAR_TIMES)


def third_friday(year: int, month: int) -> pd.Timestamp:
    fridays = [d for d in calendar.Calendar().itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return pd.Timestamp(fridays[2])


def add_months(year: int, month: int, k: int) -> tuple[int, int]:
    m0 = month - 1 + k
    return year + m0 // 12, m0 % 12 + 1


def listed_contracts(d: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    """CFFEX listing on date d: (code, expiry) for M0, M1 and the next two quarter months."""
    y, m = d.year, d.month
    if d > third_friday(y, m):
        y, m = add_months(y, m, 1)
    months = [(y, m), add_months(y, m, 1)]
    yy, mm = months[-1]
    while len(months) < 4:
        yy, mm = add_months(yy, mm, 1)
        if mm in (3, 6, 9, 12):
            months.append((yy, mm))
    return [(f"IC{yy % 100:02d}{mm:02d}.CFE", third_friday(yy, mm)) for yy, mm in months]


def ar1(rng, n, mean, phi, sd, x0=None):
    x = np.empty(n)
    x[0] = mean if x0 is None else x0
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + rng.normal(0.0, sd)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index-start", default="2013-01-07",
                    help="first index / Shibor date (two years before --start for vol warm-up)")
    ap.add_argument("--start", default="2015-01-05", help="first futures date")
    ap.add_argument("--end", default="2025-12-10", help="last date (see note in the module docstring)")
    ap.add_argument("--minute-start", default="2023-01-03")
    ap.add_argument("--minute-end", default="2023-06-30")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    idx_days = pd.bdate_range(args.index_start, args.end)      # index + Shibor calendar
    days = idx_days[idx_days >= pd.Timestamp(args.start)]       # futures calendar
    n = len(idx_days)
    RAW.mkdir(parents=True, exist_ok=True)

    # ---- index: GBM daily closes with a simple OHLC envelope -------------------------
    rets = rng.normal((DRIFT - 0.5 * VOL ** 2) / 252, VOL / np.sqrt(252), n)
    close = S0 * np.exp(np.cumsum(rets))
    prev_close = np.concatenate([[S0], close[:-1]])
    open_ = prev_close * (1 + rng.normal(0, 0.003, n))
    wiggle = np.abs(rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    index = pd.DataFrame({"time": idx_days, "open": open_, "high": high, "low": low, "close": close})

    # ---- rates and basis -------------------------------------------------------------
    shibor_pct = np.clip(ar1(rng, n, 2.3, 0.995, 0.01), 0.5, 5.0)
    repo = np.clip(ar1(rng, n, REPO_MEAN, 0.98, 0.006), -0.25, 0.05)

    # ---- futures daily: 4 listed contracts per day ------------------------------------
    rows = []
    for d in days:
        i = idx_days.get_loc(d)
        for slot, (code, expiry) in enumerate(listed_contracts(d)):
            tau = max((expiry - d).days, 1) / 365.0
            f_close = close[i] * np.exp(repo[i] * tau + rng.normal(0, 0.0008))
            f_open = f_close * open_[i] / close[i] * (1 + rng.normal(0, 0.0005))
            w = abs(rng.normal(0, 0.006))
            vol_lots = SLOT_VOLUME[slot] * np.exp(rng.normal(0, 0.3))
            rows.append({
                "day": d, "code": code, "slot": slot, "expiry": expiry,
                "open": f_open,
                "high": max(f_open, f_close) * (1 + w),
                "low": min(f_open, f_close) * (1 - w),
                "close": f_close,
                "settle": f_close * (1 + rng.normal(0, 0.0003)),
                "volume": float(round(vol_lots)),
                "oi": float(round(SLOT_OI[slot] * (1 + rng.normal(0, 0.05)))),
            })
    ic = pd.DataFrame(rows)

    # ---- minute bars on the intraday window ------------------------------------------
    mdays = days[(days >= pd.Timestamp(args.minute_start)) & (days <= pd.Timestamp(args.minute_end))]
    t = np.arange(1, 241) / 240.0
    u_shape = 1.0 + 1.5 * ((t - 0.5) * 2) ** 2          # more volume at open and close
    sd_bar = VOL / np.sqrt(252 * 240)
    idx_rows, fut_rows = [], []
    ic = ic.set_index(["day", "code"])
    for d in mdays:
        i = idx_days.get_loc(d)
        o, c = open_[i], close[i]
        z = np.cumsum(rng.normal(0, sd_bar, 240))
        bridge = z - t * z[-1]                           # Brownian bridge, ends at 0
        path = o * np.exp(np.log(c / o) * t + bridge)    # path[-1] == c exactly
        index.loc[i, ["open", "high", "low"]] = [o, max(path.max(), o), min(path.min(), o)]
        date_int = int(d.strftime("%Y%m%d"))
        idx_rows.append(pd.DataFrame({"date": date_int, "time": BAR_TIMES, "close": path}))
        for code, _ in listed_contracts(d):
            row = ic.loc[(d, code)]
            basis = row["close"] / c
            fut = path * basis * (1 + rng.normal(0, 1.5e-4, 240))
            vol = row["volume"] * u_shape / u_shape.sum() * np.exp(rng.normal(0, 0.25, 240))
            vol = np.maximum(np.round(vol), 1.0)
            value = vol * fut * MULT
            fut_rows.append(pd.DataFrame({"code": code, "date": date_int, "time": BAR_TIMES,
                                          "close": fut, "volume": vol, "value": value}))
            last_hour = slice(180, 240)                  # the 60 bars ending 14:01-15:00
            settle = float((fut[last_hour] * vol[last_hour]).sum() / vol[last_hour].sum())
            ic.loc[(d, code), ["open", "high", "low", "close", "settle", "volume"]] = \
                [fut[0], fut.max(), fut.min(), fut[-1], settle, float(vol.sum())]
    ic = ic.reset_index()

    # ---- presettle / change / amount, then write --------------------------------------
    ic = ic.sort_values(["code", "day"])
    ic["presettle"] = ic.groupby("code")["settle"].shift(1).fillna(ic["settle"])
    ic["change"] = ic["close"] - ic["presettle"]
    ic["amount"] = ic["volume"] * ic["close"] * MULT
    ic = ic.sort_values(["day", "code"])
    cols = ["day", "code", "open", "high", "low", "close", "change",
            "volume", "amount", "presettle", "settle", "oi"]

    index.to_excel(RAW / "000905.SH.xlsx", index=False)
    ic[cols].to_excel(RAW / "IC_daily_price.xlsx", index=False)
    pd.DataFrame({"time": idx_days, "value": shibor_pct}).to_excel(RAW / "shibor.xlsx", index=False)
    if idx_rows:
        pd.concat(idx_rows).to_csv(RAW / "csi500_index_1min.csv", index=False)
        pd.concat(fut_rows).to_csv(RAW / "IC_quote_1min.csv", index=False)

    print(f"index calendar : {idx_days[0].date()} -> {idx_days[-1].date()}  ({n} days)")
    print(f"futures        : {days[0].date()} -> {days[-1].date()}  ({len(days)} days, "
          f"{ic['code'].nunique()} contracts, {len(ic)} rows)")
    print(f"minute window  : {mdays[0].date() if len(mdays) else '-'} -> "
          f"{mdays[-1].date() if len(mdays) else '-'}  ({len(mdays)} days x 240 bars x 4 contracts)")
    print(f"written to     : {RAW}")


if __name__ == "__main__":
    main()
