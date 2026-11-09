# -*- coding: utf-8 -*-
"""等速两段式 TWAP（重算时刻参数化的一族策略）。"""

try:
    from ..execution import (ExecutionStrategy, DayContext, _diff_orders,
                             SIGNAL_TIME, FINAL_SIGNAL_TIME, HOUR_WINDOW)
except ImportError:
    from execution import (ExecutionStrategy, DayContext, _diff_orders,
                           SIGNAL_TIME, FINAL_SIGNAL_TIME, HOUR_WINDOW)


class TwapHourPace(ExecutionStrategy):
    """两段式 TWAP 的等速变体：首段按一小时 TWAP 的步速执行，
    重算时刻 final_time 为参数（一族策略）。

    记最后一小时共 N=60 根 bar，重算时刻前有 m 根（final_time=14:55 时
    m=55、14:50 时 m=50）：

    13:55 算目标 A -> (14:00, final_time] m 根 bar 每根成交 (A − 持仓)/N
    （即一小时 TWAP 的单根量，m 根共完成缺口的 m/N）；
    final_time 重算目标 B -> (final_time, 15:00] N−m 根 bar
    调整 (B − 实际持仓)。

    与 TwapTwoStage 的差异是首段步速：后者在重算前完成全部缺口，收尾窗
    只成交修正量 (B − A)；本策略把 (N−m)/N 的缺口留给收尾窗，与修正量
    合并成交。B = A 时逐根成交量与一小时 TWAP 完全一致。

    delta_cap 为执行侧敞口上限（保证金约束，默认 2.0；None = 不截断）。
    """

    def __init__(self, name: str = "twap_hour_pace",
                 signal_time: int = SIGNAL_TIME,
                 final_time: int = FINAL_SIGNAL_TIME,
                 delta_cap: float = 2.0):
        if not HOUR_WINDOW[0] <= final_time < HOUR_WINDOW[1]:
            raise ValueError("重算时刻须在最后一小时内且早于收盘"
                             "（bar 按结束时刻标记）")
        self.name = name
        self.signal_time = signal_time
        self.final_time = final_time
        self.delta_cap = delta_cap

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        a = ctx.target_lots(self.signal_time, delta_cap=self.delta_cap)
        orders = []
        hour = ctx.bars_between(*HOUR_WINDOW)
        bars = hour[hour <= self.final_time]      # 首段：(14:00, final_time]
        fbars = hour[hour > self.final_time]      # 收尾：(final_time, 15:00]
        n_total = len(hour)
        mid = dict(pos)                           # 首段结束时的持仓
        for c, q in _diff_orders(a, pos).items():
            per = q / n_total
            orders += [(int(t), c, per) for t in bars]
            mid[c] = mid.get(c, 0.0) + per * len(bars)
        b = ctx.target_lots(self.final_time, delta_cap=self.delta_cap)
        for c, q in _diff_orders(b, mid).items():
            per = q / len(fbars)
            orders += [(int(t), c, per) for t in fbars]
        return orders, {"lots_a": sum(a.values()), "lots_b": sum(b.values()),
                        "arrival_px": ctx.fut_price(ctx.code,
                                                    self.signal_time)}
