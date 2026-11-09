# -*- coding: utf-8 -*-
"""
market_params.py —— 第1层：定价所需市场参数的估计

雪球定价（第2层）需要四个输入：现价 S、无风险利率 r、repo（= r − q，
风险中性漂移）、波动率 sigma。其中 S 和 r 直接来自 data_loader，
本模块负责估计剩下两个"看不见的"参数，各输出一条日频序列：

repo 序列（贴水的年化度量，本策略最重要的市场参数）
    raw_t = ln(F_t / S_t) / tau_t x (252 / 年均交易日数)
        F   : 当季槽位合约的结算价（该槽位剩余期限稳定在约2~5个月，
              分母 tau 天然安全，无需再剔除临近交割时段）
        tau : 该合约剩余期限（自然日/365）
        末项: 双时钟换算——引擎按 1/252 交易日时钟施加漂移，repo 须为
              "每 252 交易日"口径（详见 estimate_repo 内注释）
    处理: 先换算、再截断（±30%，防脏数据）、再20日均值平滑（贴水日间
          噪音大，直接用 raw 会让 delta 每天乱抖）。
    符号: IC 长期贴水 => F < S => repo 为负（即 q > r）。

vol 序列
    指数对数收益的250日滚动标准差，年化 sqrt(252)。
    指数历史比 IC 早约两年，正好覆盖预热，序列在整个回测区间无缺失。

使用约定：
    - 票息在虚拟雪球发行日用当日参数锁定，存续期内不变；
    - delta 每日用最新参数重算。
    两者都通过 snapshot(date) 取数，区别只是调用的日期。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .data_loader import MarketData
    from .futures_chain import FuturesChain
except ImportError:  # 直接运行本文件做冒烟测试
    from data_loader import MarketData
    from futures_chain import FuturesChain

# 默认参数（可在 build_market_params 调用处覆盖）
REPO_WINDOW = 20     # repo 平滑窗口（交易日）
REPO_CLIP = 0.30     # repo 原始值截断带（年化 ±30%，防脏数据）
VOL_WINDOW = 250     # 历史波动率窗口（交易日，约1年）
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class MarketSnapshot:
    """某一天定价所需的全部市场参数。"""
    date: pd.Timestamp
    s: float        # 指数收盘价
    r: float        # 无风险利率（Shibor 1年）
    repo: float     # r - q，风险中性漂移（贴水为负值）
    sigma: float    # 年化波动率

    @property
    def q(self) -> float:
        """隐含分红/贴水率 q = r − repo。"""
        return self.r - self.repo


@dataclass(frozen=True)
class MarketParams:
    """两条参数序列 + 取数接口。由 build_market_params() 构造。"""
    repo: pd.Series   # 平滑后的年化 repo，index=共同交易日（前约20日为NaN）
    vol: pd.Series    # 年化历史波动率，index=共同交易日（无NaN）
    raw_repo: pd.Series  # 截断后、平滑前的 repo（诊断用，不参与定价）
    _md: MarketData

    def snapshot(self, date) -> MarketSnapshot:
        """取某日定价参数包。预热未完成的日期直接报错，防止悄悄用NaN定价。"""
        d = pd.Timestamp(date)
        return self._build(d, d)

    def snapshot_prev(self, date) -> MarketSnapshot:
        """T-1 参数快照：现价与日期取 date 当日，sigma/repo/r 取前一交易日。

        参数估计全部涉及当日收盘/结算信息（vol 窗含当日收盘、repo 由当日
        期货结算价反解），当日收盘前不可得。用前一交易日已确认的参数为
        当日定价，是日频回测与盘中执行层统一的无前视口径。
        """
        d = pd.Timestamp(date)
        days = self._md.trading_days
        pos = days.get_loc(d)
        if pos == 0:
            raise ValueError(f"{d.date()} 为日历首日，无前一交易日参数")
        return self._build(d, days[pos - 1])

    def _build(self, d: pd.Timestamp, dp: pd.Timestamp) -> MarketSnapshot:
        """快照组装：现价取 d 日，参数取 dp 日（dp=d 即当日口径）。"""
        repo, sigma = self.repo.get(dp, np.nan), self.vol.get(dp, np.nan)
        if np.isnan(repo) or np.isnan(sigma):
            raise ValueError(
                f"{dp.date()} 的参数不可用（repo={repo}, vol={sigma}）。"
                f"repo 首个可用日: {self.repo.first_valid_index().date()}"
            )
        return MarketSnapshot(date=d, s=self._md.spot(d), r=self._md.rf(dp),
                              repo=float(repo), sigma=float(sigma))


# ---------------------------------------------------------------------------
# 估计函数
# ---------------------------------------------------------------------------

def estimate_repo(md: MarketData, chain: FuturesChain,
                  window: int = REPO_WINDOW,
                  clip: float = REPO_CLIP) -> tuple[pd.Series, pd.Series]:
    """当季合约反推年化 repo 序列。

    Returns
    -------
    (smoothed, raw) : 平滑后的序列（定价用）与截断后的原始序列（诊断用）。
    """
    rows = []
    for d in md.trading_days:
        _, f, tau = chain.quarterly_for_repo(d)
        rows.append(np.log(f / md.spot(d)) / tau)
    raw = pd.Series(rows, index=md.trading_days, name="repo_raw")

    # 双时钟换算：ln(F/S)/tau_natural 是"每自然年"的漂移，而两个引擎都按
    # 1/252 的**交易日**时钟施加（与 vol 同钟）。一个自然年只有约 243 个
    # 交易日，直接混用会让实际施加的年漂移缩水 ~243/252 ≈ 3.6%，且 MC/PDE
    # 共用口径、互验不可见。按本日历的经验交易日密度换算成"每 252 交易日"
    # 口径，使引擎的累积漂移与市场隐含远期一致（残余为逐日节假日分布的
    # 噪音，量级远小于 repo 本身的估计噪音，且被 window 日平滑吸收）。
    days = md.trading_days
    td_per_year = len(days) / ((days[-1] - days[0]).days / 365.0)
    raw = raw * (TRADING_DAYS_PER_YEAR / td_per_year)

    raw = raw.clip(-clip, clip)                      # 1) 截断：防脏数据
    smoothed = raw.rolling(window, min_periods=window).mean()  # 2) 平滑
    smoothed.name = "repo"
    return smoothed, raw


def estimate_vol(md: MarketData, window: int = VOL_WINDOW) -> pd.Series:
    """指数对数收益的滚动历史波动率（年化）。

    在指数全部历史上计算（2013年起），再对齐到共同交易日——
    这样共同日历的第一天就已有完整的250日预热，序列无缺失。
    """
    log_ret = np.log(md.index["close"]).diff()
    vol_full = log_ret.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol = vol_full.reindex(md.trading_days)
    vol.name = "vol"
    if vol.isna().any():
        raise ValueError("波动率序列在共同日历上存在缺失，指数预热数据不足")
    return vol


def build_market_params(md: MarketData, chain: FuturesChain,
                        repo_window: int = REPO_WINDOW,
                        repo_clip: float = REPO_CLIP,
                        vol_window: int = VOL_WINDOW) -> MarketParams:
    """一站式构建。notebook 用法::

        params = build_market_params(md, chain)
        snap = params.snapshot("2021-06-15")   # snap.s / .r / .repo / .sigma / .q
    """
    repo, raw = estimate_repo(md, chain, repo_window, repo_clip)
    vol = estimate_vol(md, vol_window)
    return MarketParams(repo=repo, vol=vol, raw_repo=raw, _md=md)


if __name__ == "__main__":
    try:
        from .data_loader import load_market_data
    except ImportError:
        from data_loader import load_market_data

    md = load_market_data()
    chain = FuturesChain(md)
    params = build_market_params(md, chain)

    print(f"repo 首个可用日: {params.repo.first_valid_index().date()}")
    print(f"截断触发次数: {(params.raw_repo.abs() >= REPO_CLIP).sum()} / {len(params.raw_repo)}\n")

    print("repo 分年度统计（%，负=贴水）:")
    by_year = params.repo.groupby(params.repo.index.year).agg(["mean", "min", "max"]) * 100
    print(by_year.round(2).to_string())

    print("\nvol 分年度均值（%）:")
    print((params.vol.groupby(params.vol.index.year).mean() * 100).round(1).to_string())

    snap = params.snapshot("2021-06-15")
    print(f"\nsnapshot(2021-06-15): S={snap.s:.1f}, r={snap.r:.3%}, "
          f"repo={snap.repo:.3%}, q={snap.q:.3%}, sigma={snap.sigma:.3%}")
