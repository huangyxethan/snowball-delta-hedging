"""分钟级行情数据层：对冲执行回测（execution.py）的数据基础。

职责
----
1. 加载并校验两个分钟 CSV（中证500指数、IC 全合约）；
2. 提供统一的查询接口：任意交易日的指数分钟路径、任意合约的分钟价 /
   逐 bar VWAP；
3. 已知缺陷的补丁：指数分钟数据 2019-01-16 ~ 2019-02-01 共 13 天缺
   14:33-15:00 段，用当月合约分钟价按比率反推补齐（比率锚定当日最后一根
   共同可得的 bar），并打 patched 标记供统计时剔除做敏感性对照；
4. 自建结算价（最后一小时 14:01-15:00 六十根 bar 的成交量加权均价）——
   与官方结算价的对账见 __main__ 自检，其偏差分布是"TWAP 可复制结算价"
   这一执行层前提的实证依据。

时间约定
--------
bar 以**结束时刻**标记（HHMMSS 整数）：93100 表示 9:30-9:31 这一分钟，
150000 为最后一根。每个交易日恰好 240 根（9:31-11:30, 13:01-15:00）。
"在 t 时刻观察价格"= 取标记为 t 的 bar 的 close（该分钟末的价格）。

用法
----
    from intraday_data import load_intraday
    itd = load_intraday()          # 首次约 20 秒，进程内缓存
    itd.index_price(date, 135500)  # 13:55 的指数价
    itd.fut_price(date, "IC2406", 135500)
    itd.bar_fill(date, "IC2406", 140100, mode="bar_vwap")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INDEX_1MIN_FILE = DATA_DIR / "csi500_index_1min.csv"
IC_1MIN_FILE = DATA_DIR / "IC_quote_1min.csv"

IC_MULTIPLIER = 200.0
N_BARS = 240
# 最后一小时 = (14:00, 15:00] 的 60 根 bar（140000 那根覆盖 13:59-14:00，不算）
LAST_HOUR_START = 140100
SESSION_END = 150000
# 执行回测可支持的最早日期 = 指数分钟数据起点（2016-01-04）。
# 2015 年虽有期货分钟数据，但当年为旧交易时段（9:15 开盘、bar 结构不同）
# 且无指数分钟，不支持。
DEFAULT_START = "2016-01-04"


@dataclass(frozen=True)
class IntradayData:
    """分钟数据的统一查询接口（全部预透视为宽表，查询零解析成本）。"""

    times: np.ndarray            # 240 个 bar 结束时刻（HHMMSS int，升序）
    index_px: pd.DataFrame       # index=交易日, columns=times；已补丁，无 NaN
    patched: pd.Series           # 逐日 bool：该日指数是否含补丁 bar
    fut_close: pd.DataFrame      # index=(date, code), columns=times
    fut_vwap: pd.DataFrame       # 同上；bar VWAP=value/(volume×乘数)，无量 bar 回退 close
    _tpos: dict = field(default_factory=dict, repr=False)   # time -> 列号

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _col(self, time: int) -> int:
        try:
            return self._tpos[int(time)]
        except KeyError:
            raise KeyError(f"时刻 {time} 不是有效的 bar 标记（须为 240 个"
                           f"结束时刻之一，如 135500）") from None

    def has_day(self, date) -> bool:
        return pd.Timestamp(date) in self.index_px.index

    def codes_on(self, date) -> list:
        """该日有分钟数据的合约列表。"""
        d = pd.Timestamp(date)
        return list(self.fut_close.loc[d].index)

    def index_path(self, date) -> np.ndarray:
        """全日 240 根指数 bar close（已补丁）。"""
        return self.index_px.loc[pd.Timestamp(date)].to_numpy()

    def index_price(self, date, time: int) -> float:
        return float(self.index_px.at[pd.Timestamp(date), int(time)])

    def fut_price(self, date, code: str, time: int) -> float:
        """合约在 t 时刻的 bar close。"""
        return float(self.fut_close.at[(pd.Timestamp(date), code), int(time)])

    def bar_fill(self, date, code: str, time: int,
                 mode: str = "bar_vwap") -> float:
        """在 t 时刻 bar 内成交的价格假设。mode: bar_vwap | bar_close。"""
        if mode == "bar_close":
            return self.fut_price(date, code, time)
        if mode != "bar_vwap":
            raise ValueError(f"未知成交价模式 {mode!r}")
        return float(self.fut_vwap.at[(pd.Timestamp(date), code), int(time)])

def _pivot(df: pd.DataFrame, value_col: str, by_code: bool) -> pd.DataFrame:
    keys = ["date", "code", "time"] if by_code else ["date", "time"]
    wide = df.pivot_table(index=keys[:-1], columns="time",
                          values=value_col, aggfunc="first")
    return wide


def _patch_index(idx_px: pd.DataFrame, fut_close: pd.DataFrame) -> pd.Series:
    """指数缺失 bar 用当月合约按比率反推补齐（原地修改），返回补丁标记。"""
    patched = pd.Series(False, index=idx_px.index)
    missing_days = idx_px.index[idx_px.isna().any(axis=1)]
    for d in missing_days:
        row = idx_px.loc[d]
        # 当月合约 = 代码最小者（ICyymm 字典序即到期序）
        codes = fut_close.loc[d].index
        front = min(codes)
        frow = fut_close.loc[(d, front)]
        ok = row.notna()
        if not ok.any():
            raise ValueError(f"{d.date()} 指数分钟数据整日缺失，无法补丁")
        t0 = row[ok].index[-1]                 # 最后一根共同可得的 bar
        ratio = row[t0] / frow[t0]
        fill = frow[~ok.to_numpy()] * ratio
        idx_px.loc[d, ~ok.to_numpy()] = fill.to_numpy()
        patched[d] = True
    return patched


@lru_cache(maxsize=2)
def load_intraday(start: str = DEFAULT_START) -> IntradayData:
    """加载 + 校验 + 补丁。进程内缓存（两个 CSV 共约 430MB，只读一次）。"""
    t0 = pd.Timestamp(start)

    # ---- 指数 --------------------------------------------------------------
    idx = pd.read_csv(INDEX_1MIN_FILE, usecols=["date", "time", "close"])
    idx["date"] = pd.to_datetime(idx["date"], format="%Y%m%d")
    idx = idx[idx["date"] >= t0]
    idx_px = _pivot(idx, "close", by_code=False)

    # ---- 期货 --------------------------------------------------------------
    fut = pd.read_csv(IC_1MIN_FILE,
                      usecols=["code", "date", "time", "close",
                               "volume", "value"])
    fut["date"] = pd.to_datetime(fut["date"], format="%Y%m%d")
    fut = fut[fut["date"] >= t0]
    fut["code"] = fut["code"].str.replace(".CFE", "", regex=False)
    fut_close = _pivot(fut, "close", by_code=True)

    # 逐 bar VWAP：value / (volume × 乘数)；无量 bar 回退该 bar close
    with np.errstate(invalid="ignore", divide="ignore"):
        fut["vwap"] = fut["value"] / (fut["volume"] * IC_MULTIPLIER)
    fut.loc[fut["volume"] <= 0, "vwap"] = np.nan
    fut_vwap = _pivot(fut, "vwap", by_code=True)
    fut_vwap = fut_vwap.fillna(fut_close)

    # ---- 校验 --------------------------------------------------------------
    times = idx_px.columns.to_numpy()
    if len(times) != N_BARS:
        raise ValueError(f"指数分钟数据的 bar 标记数 {len(times)} != {N_BARS}")
    if not np.array_equal(times, fut_close.columns.to_numpy()):
        raise ValueError("指数与期货的 bar 时刻标记不一致")
    if fut_close.isna().any().any():
        bad = fut_close.index[fut_close.isna().any(axis=1)][:3]
        raise ValueError(f"期货分钟数据存在缺 bar，首例: {list(bad)}")
    # 期货 vs 指数的交易日对齐（期货日历是回测日历的基准）
    fut_days = fut_close.index.get_level_values(0).unique()
    miss = fut_days.difference(idx_px.index)
    if len(miss):
        raise ValueError(f"有 {len(miss)} 个期货交易日缺指数分钟数据，"
                         f"首个: {miss[0].date()}")

    # ---- 补丁 --------------------------------------------------------------
    patched = _patch_index(idx_px, fut_close)
    if idx_px.isna().any().any():
        raise ValueError("补丁后指数分钟数据仍有 NaN")

    return IntradayData(times=times, index_px=idx_px, patched=patched,
                        fut_close=fut_close, fut_vwap=fut_vwap,
                        _tpos={int(t): i for i, t in enumerate(times)})


# ---------------------------------------------------------------------------
# 自建结算价对账（诊断用，独立于 IntradayData 的查询主路径）
# ---------------------------------------------------------------------------

def settle_check(start: str = "2018-01-01") -> pd.DataFrame:
    """自建结算价（最后一小时成交量加权）vs 官方结算价的逐合约日对账表。

    Returns
    -------
    DataFrame[date, code, settle_official, settle_synth, diff_bp]
    """
    try:
        from .data_loader import load_ic
    except ImportError:
        from data_loader import load_ic

    fut = pd.read_csv(IC_1MIN_FILE,
                      usecols=["code", "date", "time", "volume", "value"])
    fut["date"] = pd.to_datetime(fut["date"], format="%Y%m%d")
    fut = fut[(fut["date"] >= pd.Timestamp(start))
              & (fut["time"] >= LAST_HOUR_START)]
    fut["code"] = fut["code"].str.replace(".CFE", "", regex=False)
    g = fut.groupby(["date", "code"])[["volume", "value"]].sum()
    g = g[g["volume"] > 0]
    synth = g["value"] / (g["volume"] * IC_MULTIPLIER)

    ic = load_ic()
    ic = ic[ic["date"] >= pd.Timestamp(start)]
    official = ic.set_index(["date", "code"])["settle"]

    out = pd.DataFrame({"settle_official": official,
                        "settle_synth": synth}).dropna()
    out["diff_bp"] = (out["settle_synth"] / out["settle_official"] - 1) * 1e4
    return out.reset_index()


if __name__ == "__main__":
    import time as _time

    t = _time.time()
    itd = load_intraday()
    print(f"加载 {_time.time() - t:.1f}s")
    print(f"指数: {len(itd.index_px)} 天  "
          f"{itd.index_px.index[0].date()} -> {itd.index_px.index[-1].date()}")
    print(f"期货: {len(itd.fut_close)} 合约日, "
          f"{itd.fut_close.index.get_level_values(1).nunique()} 只合约")
    print(f"补丁日: {int(itd.patched.sum())} 天 "
          f"{[d.date().isoformat() for d in itd.patched[itd.patched].index[:3]]}...")

    # 抽查：13:55 查询与补丁日的尾段单调性
    d = itd.index_px.index[-1]
    print(f"\n抽查 {d.date()}: 13:55 指数 {itd.index_price(d, 135500):.2f}, "
          f"15:00 {itd.index_price(d, 150000):.2f}")
    pd_day = itd.patched[itd.patched].index[0]
    print(f"补丁日 {pd_day.date()}: 14:32 {itd.index_price(pd_day, 143200):.2f} "
          f"-> 15:00(补) {itd.index_price(pd_day, 150000):.2f}")

    # 结算价对账
    t = _time.time()
    chk = settle_check()
    print(f"\n结算价对账 {_time.time() - t:.1f}s, {len(chk)} 合约日")
    absbp = chk["diff_bp"].abs()
    print(f"自建 vs 官方 |diff| bp: mean={absbp.mean():.3f} "
          f"p50={absbp.quantile(.5):.3f} p95={absbp.quantile(.95):.3f} "
          f"p99={absbp.quantile(.99):.3f} max={absbp.max():.3f}")
    worst = chk.nlargest(3, "diff_bp", keep="all")[
        ["date", "code", "settle_official", "settle_synth", "diff_bp"]]
    print("最大偏差样本:")
    print(worst.to_string(index=False))
