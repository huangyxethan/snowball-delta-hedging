# -*- coding: utf-8 -*-
"""幂权重两段式 TWAP（执行权重时间分布参数化的一族策略）。"""

import numpy as np

try:
    from ..execution import (ExecutionStrategy, DayContext, _diff_orders,
                             SIGNAL_TIME, FINAL_SIGNAL_TIME, HOUR_WINDOW)
except ImportError:
    from execution import (ExecutionStrategy, DayContext, _diff_orders,
                           SIGNAL_TIME, FINAL_SIGNAL_TIME, HOUR_WINDOW)


class TwapPowerWeight(ExecutionStrategy):
    """等速两段式 TWAP 的权重变体：切片量随时间按幂律分布，
    幂指数 power 为参数（一族策略，扫"前置—均匀—后置"谱系）。

    记最后一小时 N=60 根 bar（按时间序 i=1..N），第 i 根的权重

        w_i ∝ (i/N)^power，归一化后 sum(w) = 1。

    power=0 等权，逐 bar 退化为 TwapHourPace（等速）；power>0 后置
    （尾盘权重大），power<0 前置。两段结构与等速版相同：

    13:55 算目标 A -> (14:00, final_time] 前 m 根 bar 按 w_i 成交
    缺口 (A − 持仓) 的对应份额（共完成 sum(w_1..w_m)）；
    final_time 重算目标 B -> 剩余 N−m 根 bar 按 w_i 重归一化
    分摊 (B − 实际持仓)。

    动机（execution_report 第 6 节方向 3）：尾盘修正单方向与当日尾盘
    走势反向、成交于窗口末端而结算价为整窗 vwap，权重后置预期放大
    该正贡献；代价是前段建仓不足，数量项方差上升。

    基底权重由 _base_weights 给出（本类为等权 1；子类可换成市场成交量
    分布，见 TwapVolProfile），幂倾斜乘在基底之上。

    mid_time（可选）为 final_time 之前的中间重算时刻（单个 int 或递增
    元组）：每次重算目标，差额按剩余 bar 的权重重归一化执行至下一重算
    时刻，final_time 再重算收尾。None = 只有 final_time 一次重算。

    delta_cap 为执行侧敞口上限（保证金约束，默认 2.0；None = 不截断）。
    """

    def __init__(self, name: str = "twap_power",
                 signal_time: int = SIGNAL_TIME,
                 final_time: int = FINAL_SIGNAL_TIME,
                 power: float = 1.0, delta_cap: float = 2.0,
                 mid_time: int = None):
        if not HOUR_WINDOW[0] <= final_time < HOUR_WINDOW[1]:
            raise ValueError("重算时刻须在最后一小时内且早于收盘"
                             "（bar 按结束时刻标记）")
        mids = ([] if mid_time is None else
                [mid_time] if isinstance(mid_time, int) else list(mid_time))
        if any(not HOUR_WINDOW[0] <= m < final_time for m in mids) \
                or mids != sorted(set(mids)):
            raise ValueError("中间重算时刻须在最后一小时内、早于 final_time "
                             "且严格递增")
        self.name = name
        self.signal_time = signal_time
        self.final_time = final_time
        self.power = power
        self.delta_cap = delta_cap
        self.mid_time = mid_time

    def _base_weights(self, ctx: DayContext, n: int) -> np.ndarray:
        """60 根 bar 的基底权重（未归一）；等权。"""
        return np.ones(n)

    def _target(self, ctx: DayContext, time: int) -> dict:
        """t 时刻的目标持仓（子类钩子：事件清仓等目标改写在此注入）。"""
        return ctx.target_lots(time, delta_cap=self.delta_cap)

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        a = self._target(ctx, self.signal_time)
        hour = ctx.bars_between(*HOUR_WINDOW)
        n = len(hour)
        w = self._base_weights(ctx, n) * (np.arange(1, n + 1) / n) ** self.power
        w /= w.sum()
        # 重算时刻序列；每次重算的差额按剩余 bar 的权重重归一化，
        # 执行至下一重算时刻（首段的"剩余"即整窗，权重和为 1）
        mids = ([] if self.mid_time is None else
                [self.mid_time] if isinstance(self.mid_time, int)
                else list(self.mid_time))
        recalcs = mids + [self.final_time]
        bounds = [HOUR_WINDOW[0] - 1] + recalcs + [HOUR_WINDOW[1]]
        orders, cur, tgt, info = [], dict(pos), a, {"lots_a": sum(a.values())}
        for k, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
            if k > 0:
                tgt = self._target(ctx, recalcs[k - 1])
                info[f"lots_mid{k}" if k < len(recalcs) else "lots_b"] = \
                    sum(tgt.values())
            seg = (hour > t0) & (hour <= t1)
            wseg = w[seg] / w[hour > t0].sum()
            for c, q in _diff_orders(tgt, cur).items():
                orders += [(int(t), c, q * wi)
                           for t, wi in zip(hour[seg], wseg)]
                cur[c] = cur.get(c, 0.0) + q * wseg.sum()
        info["arrival_px"] = ctx.fut_price(ctx.code, self.signal_time)
        return orders, info
