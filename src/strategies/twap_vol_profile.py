# -*- coding: utf-8 -*-
"""成交量分布基底的幂权重两段式 TWAP（切片格子按市场分钟成交量份额加权）。"""

import numpy as np

try:
    from ..execution import DayContext, SIGNAL_TIME, FINAL_SIGNAL_TIME
    from ..vol_profile import load_vol_profile, DEFAULT_WINDOW
except ImportError:
    from execution import DayContext, SIGNAL_TIME, FINAL_SIGNAL_TIME
    from vol_profile import load_vol_profile, DEFAULT_WINDOW

from .twap_power_weight import TwapPowerWeight


class TwapVolProfile(TwapPowerWeight):
    """TwapPowerWeight 的成交量基底变体：

        w_i ∝ v̄_i · (i/N)^power

    v̄_i 为最后一小时第 i 分钟成交量份额的滚动 window 日均值（截至前一
    交易日，无前视；见 vol_profile.py）。结算价按当日实际成交量加权，
    v̄ 是其事前可知的最佳近似：power=0 时成交均价近似复制结算价，是
    无前视条件下相对结算价的中性调度；power>0 在其上后置押尾盘延续。
    成交量基底自带约 0.25 个 power 的后置（尾窗份额 14.5% vs 等权
    8.3%），同等后置程度下 power 取值应低于等权基底。

    两段结构、14:55 重算、尾窗重归一化与父类相同。
    """

    def __init__(self, name: str = "twap_volprof",
                 signal_time: int = SIGNAL_TIME,
                 final_time: int = FINAL_SIGNAL_TIME,
                 power: float = 0.0, window: int = DEFAULT_WINDOW,
                 delta_cap: float = 2.0):
        super().__init__(name=name, signal_time=signal_time,
                         final_time=final_time, power=power,
                         delta_cap=delta_cap)
        self.window = window

    def _base_weights(self, ctx: DayContext, n: int) -> np.ndarray:
        prof = load_vol_profile(self.window)          # 模块级进程内缓存
        try:
            row = prof.loc[ctx.date].to_numpy(dtype=float)
        except KeyError:
            raise KeyError(f"成交量分布档案缺 {ctx.date.date()}（窗口 "
                           f"{self.window} 日；请检查 data/processed）")
        if row.size != n or not np.isfinite(row).all():
            raise ValueError(f"{ctx.date.date()} 成交量分布行异常")
        return row
