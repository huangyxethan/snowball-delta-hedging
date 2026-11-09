# -*- coding: utf-8 -*-
"""计价窗前执行的两段式 TWAP（对照策略：成交落在结算价计价窗之外）。"""

try:
    from ..execution import (ExecutionStrategy, DayContext, _diff_orders,
                             FINAL_SIGNAL_TIME, FINAL_WINDOW)
except ImportError:
    from execution import (ExecutionStrategy, DayContext, _diff_orders,
                           FINAL_SIGNAL_TIME, FINAL_WINDOW)


class TwapPreWindow(ExecutionStrategy):
    """两段式 TWAP 整体前移一小时至计价窗之前，再保留 14:55 尾盘修正。

    11:30 算目标 A -> (13:00, 13:55] 55 根 bar TWAP 执行 (A − 持仓)；
    13:55 重算目标 B -> (13:55, 14:00] 5 根 bar 调整 (B − A)；
    14:55 重算目标 C -> (14:55, 15:00] 5 根 bar 调整 (C − B)。

    设计意图：首两段与基准两段式几何相同、仅整体早一小时，全部成交落在
    结算价计价窗 (14:00, 15:00] 之前；末段与基准完全一致，故日终持仓、
    数量项与基准逐日相同，净值差的变化全部由"计价窗外成交"的价格项
    承担——用于检验"信号提前 + 日中多次修正"的思路。

    delta_cap 为执行侧敞口上限（保证金约束，默认 2.0；None = 不截断）。
    """

    def __init__(self, name: str = "twap_pre_window",
                 signal_time: int = 113000, mid_time: int = 135500,
                 final_time: int = FINAL_SIGNAL_TIME, delta_cap: float = 2.0):
        if not signal_time < mid_time < final_time:
            raise ValueError("须满足 signal_time < mid_time < final_time"
                             "（bar 按结束时刻标记）")
        self.name = name
        self.signal_time = signal_time
        self.mid_time = mid_time
        self.final_time = final_time
        self.delta_cap = delta_cap

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        a = ctx.target_lots(self.signal_time, delta_cap=self.delta_cap)
        orders = []
        bars = ctx.bars_between(130100, self.mid_time)        # (13:00, 13:55]
        for c, q in _diff_orders(a, pos).items():
            per = q / len(bars)
            orders += [(int(t), c, per) for t in bars]
        b = ctx.target_lots(self.mid_time, delta_cap=self.delta_cap)
        mbars = ctx.bars_between(self.mid_time + 100, 140000)  # (13:55, 14:00]
        for c, q in _diff_orders(b, a).items():
            per = q / len(mbars)
            orders += [(int(t), c, per) for t in mbars]
        cc = ctx.target_lots(self.final_time, delta_cap=self.delta_cap)
        fbars = ctx.bars_between(*FINAL_WINDOW)                # (14:55, 15:00]
        for c, q in _diff_orders(cc, b).items():
            per = q / len(fbars)
            orders += [(int(t), c, per) for t in fbars]
        return orders, {"lots_a": sum(a.values()), "lots_b": sum(b.values()),
                        "lots_c": sum(cc.values()),
                        "arrival_px": ctx.fut_price(ctx.code,
                                                    self.signal_time)}
