# -*- coding: utf-8 -*-
"""最后一小时分钟成交量分布（执行策略的成交量基底权重）。

结算价 = 最后一小时 (14:00, 15:00] 六十根 bar 的成交量加权均价。切片权重
若取市场自身的分钟成交量份额，成交均价按构造逼近结算价；本模块提供其
**无前视**估计：截至前一交易日的滚动 window 日平均份额。

口径
----
- 逐日以最后一小时成交量最大的合约为"主力"，取其 60 根 bar 的成交量
  占该小时总量的份额（每日行和为 1）；
- 滚动 window 日均值后 **shift(1)**：日期 d 的行只用 d 之前的数据；
- 分钟成交量的日内季节性与合约无关，换月窗口内两个合约共用同一行。

档案
----
    data/processed/ic_vol_profile_w<window>.parquet
        index=交易日, columns=60 个 bar 标记 (140100..150000), 值=份额
    首次调用计算并落盘（读 IC 分钟 CSV 约 20s），此后直接读档；
    删除该文件即重算。

用法
----
    from vol_profile import load_vol_profile
    prof = load_vol_profile(window=20)      # DataFrame
    prof.loc[date].to_numpy()               # 当日基底权重（和为 1）
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .intraday_data import IC_1MIN_FILE, LAST_HOUR_START, SESSION_END
except ImportError:
    from intraday_data import IC_1MIN_FILE, LAST_HOUR_START, SESSION_END

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_WINDOW = 20

_cache: dict = {}


def _daily_shares() -> pd.DataFrame:
    """逐日主力合约最后一小时的分钟成交量份额（index=日, columns=60 bar）。"""
    fut = pd.read_csv(IC_1MIN_FILE, usecols=["code", "date", "time", "volume"])
    fut = fut[(fut["time"] >= LAST_HOUR_START) & (fut["time"] <= SESSION_END)]
    fut["date"] = pd.to_datetime(fut["date"], format="%Y%m%d")
    tot = fut.groupby(["date", "code"])["volume"].sum()
    main = tot.groupby(level=0).idxmax()               # (date, code) of max
    fut = fut.set_index(["date", "code"]).loc[main.values].reset_index()
    wide = fut.pivot(index="date", columns="time", values="volume")
    wide = wide.sort_index(axis=1)
    if wide.shape[1] != 60:
        raise ValueError(f"最后一小时应有 60 根 bar，实际 {wide.shape[1]}")
    wide = wide.fillna(0.0)
    tot = wide.sum(axis=1)
    wide = wide[tot > 0]
    return wide.div(wide.sum(axis=1), axis=0)


def build_vol_profile(window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """计算滚动 window 日均值份额并落盘（无前视：shift(1)）。"""
    shares = _daily_shares()
    prof = shares.rolling(window, min_periods=window).mean().shift(1)
    prof = prof.dropna(how="all")
    prof = prof.div(prof.sum(axis=1), axis=0)          # 数值上重归一
    prof.columns = [int(c) for c in prof.columns]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"ic_vol_profile_w{window}.parquet"
    out = prof.copy()
    out.columns = [str(c) for c in out.columns]        # parquet 列名须为字符串
    out.to_parquet(path)
    return prof


def load_vol_profile(window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """读档（无则计算）；进程内缓存。"""
    if window in _cache:
        return _cache[window]
    path = PROCESSED_DIR / f"ic_vol_profile_w{window}.parquet"
    if path.exists():
        prof = pd.read_parquet(path)
        prof.columns = [int(c) for c in prof.columns]
    else:
        prof = build_vol_profile(window)
    _cache[window] = prof
    return prof


if __name__ == "__main__":
    p = build_vol_profile()
    print(f"{p.index[0].date()} ~ {p.index[-1].date()}，{len(p)} 天")
    tail = p.iloc[:, -5:].sum(axis=1)
    print(f"尾窗(14:56-15:00)份额：均值 {tail.mean()*100:.1f}%，"
          f"范围 {tail.min()*100:.1f}% ~ {tail.max()*100:.1f}%")
    print(p.iloc[-1].round(4).to_string())
