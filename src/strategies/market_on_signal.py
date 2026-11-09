# -*- coding: utf-8 -*-
"""集中成交（时点集中度对照策略）。"""

try:
    from ..execution import (ExecutionStrategy, DayContext, _diff_orders,
                             SIGNAL_TIME)
except ImportError:
    from execution import (ExecutionStrategy, DayContext, _diff_orders,
                           SIGNAL_TIME)


class MarketOnSignal(ExecutionStrategy):
    """朴素对照策略：t0 算信号，t1 一根 bar 内全部成交。

    默认与 TwapTwoStage 同用 13:55 信号、14:00 那根 bar 成交——
    信号对齐后，两者的差异纯粹来自执行方式（一把梭 vs TWAP 摊开），
    构成"时点集中度"两端的干净对照。
    """

    def __init__(self, name: str = "market_on_signal",
                 signal_time: int = SIGNAL_TIME, trade_time: int = 140000,
                 delta_cap: float = 2.0):
        if trade_time <= signal_time:
            raise ValueError("成交时刻须晚于信号时刻（bar 按结束时刻标记）")
        self.name = name
        self.signal_time = signal_time
        self.trade_time = trade_time
        self.delta_cap = delta_cap

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        tgt = ctx.target_lots(self.signal_time, delta_cap=self.delta_cap)
        orders = [(self.trade_time, c, q)
                  for c, q in _diff_orders(tgt, pos).items()]
        return orders, {"lots_tgt": sum(tgt.values()),
                        "arrival_px": ctx.fut_price(ctx.code,
                                                    self.signal_time)}
