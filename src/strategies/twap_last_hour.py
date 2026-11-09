# -*- coding: utf-8 -*-
"""一小时 TWAP（执行窗与结算价计算窗重合的对照策略）。"""

try:
    from ..execution import (ExecutionStrategy, DayContext, _diff_orders,
                             SIGNAL_TIME, HOUR_WINDOW)
except ImportError:
    from execution import (ExecutionStrategy, DayContext, _diff_orders,
                           SIGNAL_TIME, HOUR_WINDOW)


class TwapLastHour(ExecutionStrategy):
    """一小时 TWAP：t0 算目标，(14:00, 15:00] 六十根 bar 等量执行。

    执行窗与结算价的计算窗（最后一小时成交量加权）完全重合。
    """

    def __init__(self, name: str = "twap_1h", signal_time: int = SIGNAL_TIME,
                 delta_cap: float = 2.0):
        if signal_time >= HOUR_WINDOW[0]:
            raise ValueError("信号时刻须早于执行窗起点（bar 按结束时刻标记）")
        self.name = name
        self.signal_time = signal_time
        self.delta_cap = delta_cap

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        tgt = ctx.target_lots(self.signal_time, delta_cap=self.delta_cap)
        orders = []
        bars = ctx.bars_between(*HOUR_WINDOW)
        for c, q in _diff_orders(tgt, pos).items():
            per = q / len(bars)
            orders += [(int(t), c, per) for t in bars]
        return orders, {"lots_tgt": sum(tgt.values()),
                        "arrival_px": ctx.fut_price(ctx.code,
                                                    self.signal_time)}
