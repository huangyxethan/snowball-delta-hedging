# -*- coding: utf-8 -*-
"""执行诊断导出：任意执行策略相对指数的数量项诊断 Excel（通用工具）。

export_qty_diagnostics(bt, res, strategy) 生成同一 Excel 的三张 sheet：

逐日总表    全区间每交易日一行（净值与三项分解 / 持仓与缺口 / 价格 /
            市场位置 / 事件）。核心恒等式：数量项[t] = 前日缺口 x
            当日结算价变动 x 乘数 / 名义，"复核数量项"列逐行验证
            （换月窗口两合约混合处有小残差）。事件标在**发生日**：
            0 常规 / 1 敲入切换 / 2 敲出确认 / 3 到期截断 / 4 发行；
            突变日靠"前日事件 / 前日缺口"两列实现行内自解释。
事件窗口    每个事件取 前一日 / 事件日 / 次日 三行（数量项在事件次日
            实现），块首列为 事件日 / 事件类型 / 相对日。
突变日      |数量项| 超过阈值（默认 0.5pp）的交易日，各取缺口形成日
            （前一日）与突变日本身两行。

仅支持单链条（n_slots=1）的回测结果——多槽位下"当日存续腿"不唯一，
逐行语义需另行设计。
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .hedger import IC_MULTIPLIER
    from .intraday_data import load_intraday
except ImportError:
    from hedger import IC_MULTIPLIER
    from intraday_data import load_intraday

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def export_qty_diagnostics(bt, res, strategy: str, out=None,
                           jump_threshold_pp: float = 0.5,
                           signal_time: int = 135500) -> pd.DataFrame:
    """导出 strategy（res.daily 的键）的数量项诊断 Excel，返回逐日总表。

    Parameters
    ----------
    bt / res : run_backtest / run_execution 的结果（须同一次配对）
    strategy : res.daily 中的策略名
    out : 输出路径；缺省为 output/<策略名>_数量项诊断总表.xlsx
    jump_threshold_pp : 突变日阈值（|数量项| 百分点）
    signal_time : 信号时刻（"信号→收盘"列的起点，bar 结束时刻标记）
    """
    if bt.cfg.n_slots != 1:
        raise ValueError("诊断表按单链条设计（n_slots=1），当前配置不适用")
    if strategy not in res.daily:
        raise KeyError(f"策略 {strategy!r} 不在结果中: {list(res.daily)}")
    d = res.daily[strategy]
    itd = load_intraday()
    days = d.index
    sig_lbl = f"{signal_time // 10000:02d}:{signal_time // 100 % 100:02d}→收盘%"

    tbl = pd.DataFrame(index=days)
    tbl.index.name = "日期"
    # -- 净值与差异 --
    tbl["指数净值"] = bt.daily.loc[days, "nav"]
    tbl["策略净值"] = d["nav"]
    tbl["累计净值差(pp)"] = (tbl["策略净值"] - tbl["指数净值"]) * 100
    tbl["数量项(pp)"] = d["te_qty"] * 100
    tbl["价格项(pp)"] = d["te_price"] * 100
    tbl["费差项(pp)"] = d["te_fee"] * 100
    tbl["累计数量项(pp)"] = tbl["数量项(pp)"].cumsum()
    # -- 持仓与缺口 --
    tbl["指数delta"] = bt.daily.loc[days, "delta"]
    tbl["指数目标(手)"] = d["lots_index"]
    tbl["策略持仓(手)"] = d["lots_eod"]
    tbl["缺口(手)"] = d["lots_eod"] - d["lots_index"]
    tbl["前日缺口(手)"] = tbl["缺口(手)"].shift(1)
    tbl["当日成交(手)"] = d["traded_lots"]
    # -- 价格 --
    tbl["前日结算价"] = d["settle_px"].shift(1)
    tbl["当日结算价"] = d["settle_px"]
    tbl["当日结算涨跌%"] = d["settle_px"].pct_change() * 100
    tbl["复核数量项(pp)"] = (tbl["前日缺口(手)"]
                             * (tbl["当日结算价"] - tbl["前日结算价"])
                             * IC_MULTIPLIER / bt.cfg.notional * 100)
    # -- 市场位置 --
    tbl["指数收盘价"] = bt.daily.loc[days, "spot"]
    states = bt.leg_states.set_index("date")
    leg_of_day = states["leg"].reindex(days)
    ko_lv = {i: t.ko_level for i, t in enumerate(bt.terms_list)}
    ki_lv = {i: t.ki_level for i, t in enumerate(bt.terms_list)}
    tbl["腿号"] = leg_of_day
    tbl["收盘/敲出线%"] = tbl["指数收盘价"] / leg_of_day.map(ko_lv) * 100
    tbl["收盘/敲入线%"] = tbl["指数收盘价"] / leg_of_day.map(ki_lv) * 100
    tbl[sig_lbl] = [(bt.daily.at[dt, "spot"]
                     / itd.index_price(dt, signal_time) - 1) * 100
                    for dt in days]
    # -- 事件 --
    ev = pd.Series(0, index=days, dtype=int)
    ev.loc[ev.index.intersection(pd.DatetimeIndex(bt.legs["issue_date"]))] = 4
    ev.loc[ev.index.intersection(
        states.index[states["ki_close"] & ~states["ki_prev"]])] = 1
    for dt, row in states[states["terminal"]].iterrows():
        sc = int(bt.legs.at[int(row["leg"]), "scenario"])
        ev.loc[dt] = 2 if sc in (1, 5) else 3
    label = {0: "常规", 1: "敲入切换", 2: "敲出确认", 3: "到期/截断", 4: "发行"}
    tbl["事件"] = ev
    tbl["事件_文字"] = ev.map(label)
    tbl["前日事件"] = ev.shift(1)
    tbl["换月"] = d["is_roll"]
    tbl["补丁日"] = d["patched"]

    # -- 事件窗口表 --
    win_cols = ["事件_文字", "腿号", "指数delta", "指数目标(手)", "策略持仓(手)",
                "缺口(手)", "当日成交(手)", "指数收盘价", "当日结算价",
                "当日结算涨跌%", "收盘/敲出线%", "收盘/敲入线%",
                sig_lbl, "数量项(pp)"]
    ev_blocks = []
    for p0 in np.flatnonzero(tbl["事件"].to_numpy() > 0):
        lo, hi = max(p0 - 1, 0), min(p0 + 1, len(tbl) - 1)
        seg = tbl.iloc[lo:hi + 1][win_cols].copy()
        seg.insert(0, "相对日", np.arange(lo, hi + 1) - p0)
        seg.insert(0, "事件类型", tbl["事件_文字"].iat[p0])
        seg.insert(0, "事件日", tbl.index[p0])
        ev_blocks.append(seg)
    ev_win = (pd.concat(ev_blocks) if ev_blocks else
              pd.DataFrame(columns=["事件日", "事件类型", "相对日", *win_cols]))

    # -- 突变日速览表 --
    jmp_cols = ["数量项(pp)", "累计数量项(pp)", "缺口(手)", "指数目标(手)",
                "策略持仓(手)", "当日结算涨跌%", "事件_文字", "收盘/敲出线%",
                "收盘/敲入线%", sig_lbl]
    jumps = []
    for p0 in np.flatnonzero(
            tbl["数量项(pp)"].abs().to_numpy() > jump_threshold_pp):
        lo = max(p0 - 1, 0)
        seg = tbl.iloc[lo:p0 + 1][jmp_cols].copy()
        seg.insert(0, "相对日", np.arange(lo, p0 + 1) - p0)
        seg.insert(0, "突变日", tbl.index[p0])
        jumps.append(seg)
    jump_tbl = (pd.concat(jumps) if jumps else
                pd.DataFrame(columns=["突变日", "相对日", *jmp_cols]))

    if out is None:
        out = OUTPUT_DIR / f"{strategy}_数量项诊断总表.xlsx"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        tbl.to_excel(xw, sheet_name="逐日总表", freeze_panes=(1, 1))
        ev_win.to_excel(xw, sheet_name="事件窗口", freeze_panes=(1, 4))
        jump_tbl.to_excel(xw, sheet_name="突变日", freeze_panes=(1, 3))
    print(f"逐日总表 {tbl.shape[0]} 行 / 事件窗口 {len(ev_win)} 行"
          f"（{len(ev_blocks)} 个事件） / 突变日 {len(jumps)} 个 -> {out}")
    return tbl


if __name__ == "__main__":
    from backtest import run_backtest
    from execution import run_execution, TwapLastHour

    print("小型冒烟：单链条 2 腿 + 一小时TWAP ...")
    bt = run_backtest(n_slots=1, weekly_terms=False, monday_issue=False,
                      max_legs=2, start_date="2018-01-02")
    res = run_execution(bt, [TwapLastHour(name="一小时TWAP")])
    tbl = export_qty_diagnostics(bt, res, "一小时TWAP",
                                 out=OUTPUT_DIR / "_smoke_诊断.xlsx")
    assert (tbl["数量项(pp)"] - tbl["复核数量项(pp)"]).abs().median() < 1e-6
    (OUTPUT_DIR / "_smoke_诊断.xlsx").unlink()
    print("通过（复核恒等式中位偏差 < 1e-6，临时文件已清理）")
