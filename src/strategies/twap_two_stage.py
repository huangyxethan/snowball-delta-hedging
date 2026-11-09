# -*- coding: utf-8 -*-
"""两段式 TWAP（基准执行策略）。"""

try:
    from ..execution import (ExecutionStrategy, DayContext, _diff_orders,
                             SIGNAL_TIME, FINAL_SIGNAL_TIME, EXEC_WINDOW,
                             FINAL_WINDOW)
except ImportError:
    from execution import (ExecutionStrategy, DayContext, _diff_orders,
                           SIGNAL_TIME, FINAL_SIGNAL_TIME, EXEC_WINDOW,
                           FINAL_WINDOW)


class TwapTwoStage(ExecutionStrategy):
    """基准执行策略（潘老师方案）。

    13:55 算目标 A -> (14:00, 14:55] 55 根 bar TWAP 执行 (A − 持仓)；
    14:55 重算目标 B -> (14:55, 15:00] 5 根 bar 调整 (B − A)。
    recalc=False 为"14:00 定死计划"的下界档（曾老师的可接受基础），
    日终停在 A。

    delta_cap 为执行侧敞口上限（保证金约束，默认 2.0 即旧指数基准的
    取值；None = 不截断）。指数端默认不截断，截断的代价由 TE 分解量化。
    """

    def __init__(self, name: str = "twap_2stage", recalc: bool = True,
                 signal_time: int = SIGNAL_TIME,
                 final_time: int = FINAL_SIGNAL_TIME,
                 delta_cap: float = 2.0):
        self.name = name
        self.recalc = recalc
        self.signal_time = signal_time
        self.final_time = final_time
        self.delta_cap = delta_cap

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        a = ctx.target_lots(self.signal_time, delta_cap=self.delta_cap)
        orders = []
        bars = ctx.bars_between(*EXEC_WINDOW)
        for c, q in _diff_orders(a, pos).items():
            per = q / len(bars)
            orders += [(int(t), c, per) for t in bars]
        info = {"lots_a": sum(a.values())}
        if self.recalc:
            b = ctx.target_lots(self.final_time, delta_cap=self.delta_cap)
            fbars = ctx.bars_between(*FINAL_WINDOW)
            for c, q in _diff_orders(b, a).items():
                per = q / len(fbars)
                orders += [(int(t), c, per) for t in fbars]
            info["lots_b"] = sum(b.values())
        # 到达价（滑点的辅助尺子）：信号时刻的目标合约价
        info["arrival_px"] = ctx.fut_price(ctx.code, self.signal_time)
        return orders, info
