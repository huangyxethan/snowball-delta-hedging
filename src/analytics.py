# -*- coding: utf-8 -*-
"""
analytics.py —— 回测结果的对照基准与统计量

把 notebook 中重复的分析代码收敛到此处，保持 notebook 只做调用与展示。

净值口径（三条曲线统一，随 cfg.notional_mode 切换）
--------------------------------------------------
    notional_mode="leg"（每腿重置，默认）:
        策略  : 名义在各腿发行日按前一日收盘净值重置、腿内不变，故净值腿内
                算术、腿间连乘
        CSI500: 现货多头走同一套腿边界（accumulate_nav），即同一份资金注入表
        年化  : NAV_T^(1/年数) - 1（几何）

    notional_mode="daily"（名义逐日随净值缩放）:
        策略  : NAV_T = prod_t (1 + delta_{t-1} r_t)，由名义基数的缩放自动产生
        CSI500: 买入持有的几何净值 prod_t (1 + S_t/S_{t-1} - 1)
        年化  : 几何

    notional_mode="fixed"（固定名义、不复利）:
        策略  : NAV_T = 1 + sum_t (当日损益 - 当日手续费) / 名义本金，即每日
                收益的**算术和**
        CSI500: 固定名义的现货多头 1 + sum_t (S_t/S_{t-1} - 1)，无手续费
        年化  : 总收益 / 年数（线性）

    日收益：fixed 取净值一阶差分，leg / daily 取净值相对变化。
    三种模式下 IC多头 均为 delta 恒为 1 的期货多头，走同一套 hedger 记账
    （含换月、取整、手续费，费率、名义口径与腿边界均同策略）。

    ⚠ 切换口径会**同时**改变累计方式与年化定义。实测本项目区间上总收益升高
    而"年化收益"数字下降，纯粹是年化从线性换成几何所致，不是策略变差——
    比较时须同口径对同口径。
"""

import numpy as np
import pandas as pd

try:
    from .backtest import _market_data, _chain, _annualize, _coverage_ann
    from .hedger import run_hedge, constant_delta
except ImportError:
    from backtest import _market_data, _chain, _annualize, _coverage_ann
    from hedger import run_hedge, constant_delta


# ---------------------------------------------------------------------------
# 回撤
# ---------------------------------------------------------------------------

def drawdown_detail(nav: pd.Series) -> dict:
    """最大回撤及其三个时点。

    Returns
    -------
    dict: max_dd（负值）、peak_date（前高日）、trough_date（谷底日）、
          recovery_date（首次回到前高之日，未修复为 NaT）、
          dd_days（前高到谷底的自然日）、recovery_days（谷底到修复的自然日）
    """
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough = dd.idxmin()
    max_dd = float(dd.min())
    peak_date = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    rec = after[after >= nav[peak_date]]
    recovery = rec.index[0] if len(rec) else pd.NaT
    return {
        "max_dd": max_dd,
        "peak_date": peak_date,
        "trough_date": trough,
        "recovery_date": recovery,
        "dd_days": (trough - peak_date).days,
        "recovery_days": ((recovery - trough).days if pd.notna(recovery) else np.nan),
    }


# ---------------------------------------------------------------------------
# 净值统计
# ---------------------------------------------------------------------------

def nav_metrics(nav: pd.Series, rf: float = 0.0,
                geometric: bool = False, start=None) -> pd.Series:
    """单条净值的统计量。

    Parameters
    ----------
    geometric : bool
        False（算术口径，notional_mode="fixed"）：日收益 = 净值一阶差分，
        年化 = 总收益 / 年数。
        True（复利口径，notional_mode="leg" 或 "daily"）：日收益 = 净值相对
        变化，年化 = NAV_T^(1/年数) - 1。
        回撤在两种口径下都是 nav/cummax - 1，无需区分。

        注："leg" 下净值在腿内是冻结基数上的算术累计，净值的相对变化并不严格
        等于该基数上的收益率。此处仍取相对变化——它就是投资者从一条公开净值
        曲线上能算出的日收益，与夏普/波动的通行定义一致；需要严格的基数收益
        请用 HedgeResult.pnl_base 归一。
    start : 可选的起算日。给出时净值截取 [start, 末日] 并在起点重新归一
        （几何口径除以起点值、算术口径平移回 1），统计只覆盖该段。
        ladder 回测建议传 BacktestResult.rampup_end（建仓完成日），
        以剔除建仓期的仓位不足对年化指标的稀释。
    """
    if start is not None:
        nav = nav.loc[pd.Timestamp(start):]
        if len(nav) == 0:
            raise ValueError(f"start={start} 晚于净值序列末端")
        nav = nav / nav.iloc[0] if geometric else nav - nav.iloc[0] + 1.0
    r = (nav.pct_change() if geometric else nav.diff()).dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.0
    total = float(nav.iloc[-1] - 1.0)
    if years <= 0:
        ann_ret = np.nan
    elif geometric:
        ann_ret = float(nav.iloc[-1]) ** (1.0 / years) - 1.0
    else:
        ann_ret = total / years
    ann_vol = float(r.std() * np.sqrt(252))
    dd = drawdown_detail(nav)
    return pd.Series({
        "总收益": total,
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普": (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan,
        "卡玛": ann_ret / abs(dd["max_dd"]) if dd["max_dd"] else np.nan,
        "最大回撤": dd["max_dd"],
        "回撤起始日": dd["peak_date"].date(),
        "回撤谷底日": dd["trough_date"].date(),
        "回撤修复日": (dd["recovery_date"].date()
                       if pd.notna(dd["recovery_date"]) else "未修复"),
        "回撤持续(天)": dd["dd_days"],
        "修复耗时(天)": dd["recovery_days"],
        "日胜率": float((r > 0).mean()),
    })


# ---------------------------------------------------------------------------
# 对照基准
# ---------------------------------------------------------------------------

def accumulate_nav(returns: pd.Series, notional_mode: str,
                   reset_dates=None) -> pd.Series:
    """把"相对当前名义基数的每日收益率"按给定口径累计成净值。

    与 hedger 的三种 notional_mode 一一对应，供不走 hedger 的曲线（如指数
    现货多头）构造同口径的对照净值：

        fixed: 1 + cumsum(r)                    算术和
        leg  : 窗口内 base×(1+cumsum)、窗口间连乘   腿内算术、腿间连乘
        daily: cumprod(1 + r)                   逐日连乘

    reset_dates 仅 "leg" 需要（首日总是隐含重置点）。
    """
    r = returns.fillna(0.0)
    if notional_mode == "fixed":
        return 1.0 + r.cumsum()
    if notional_mode == "daily":
        return (1.0 + r).cumprod()
    if notional_mode != "leg":
        raise ValueError(f"未知 notional_mode: {notional_mode!r}")
    if reset_dates is None:
        raise ValueError('notional_mode="leg" 须提供 reset_dates')

    resets = pd.DatetimeIndex(reset_dates)
    nav = np.empty(len(r))
    base = 1.0
    acc = 0.0
    in_reset = r.index.isin(resets)
    for i, x in enumerate(r.to_numpy(dtype=float)):
        if in_reset[i] and i > 0:
            base = nav[i - 1]       # 前一交易日收盘净值，与 hedger 的口径一致
            acc = 0.0
        acc += x
        nav[i] = base * (1.0 + acc)
    return pd.Series(nav, index=r.index, name=returns.name)

def build_benchmarks(res) -> pd.DataFrame:
    """构造与策略同区间、同口径的两条对照净值。

    Parameters
    ----------
    res : BacktestResult

    Returns
    -------
    pd.DataFrame，index 同 res.daily，列：策略 / CSI500 / IC多头
    """
    md = _market_data()
    chain = _chain(res.cfg.roll_buffer_days)
    idx = res.daily.index

    # 名义基数的重置点：leg 口径下三条曲线用**同一套**腿边界，即同一份"资金
    # 注入时间表"，否则曲线不可比。
    resets = (list(res.legs["issue_date"]) if res.cfg.notional_mode == "leg"
              else None)

    # CSI500：现货多头。口径须与策略一致，否则同图不可比、年化超额失真
    spot = md.index["close"].reindex(idx)
    csi = accumulate_nav(spot.pct_change().fillna(0.0),
                         res.cfg.notional_mode, resets)

    # IC 多头：delta 恒为 1，走同一套记账（含换月/展期、取整、手续费、
    # 名义口径——均同策略）
    ic = run_hedge(constant_delta(idx, 1.0),
                   chain.build_schedule(res.cfg.hedge_contract,
                                        roll_window_days=res.cfg.roll_window_days),
                   spot=md.index["close"], notional=res.cfg.notional,
                   round_lots=res.cfg.round_lots, close_at_end=False,
                   fee_rate=res.cfg.fee_rate,
                   notional_mode=res.cfg.notional_mode, reset_dates=resets)

    return pd.DataFrame({
        "策略": res.daily["nav"],
        "CSI500": csi,
        f"IC多头({res.cfg.hedge_contract})": ic.daily["nav"],
    })


def compare_navs(navs: pd.DataFrame, rf: float = None,
                 benchmark: str = "CSI500",
                 geometric: bool = False) -> pd.DataFrame:
    """多条净值的统计量对比表，附相对基准的年化超额。

    Parameters
    ----------
    benchmark : 用作超额基准的列名，默认 "CSI500"（指数买入持有）。
        该列自身的年化超额记为 NaN。若列名不存在则退回第一列。
    geometric : 见 nav_metrics。须与 build_benchmarks 所用口径一致，
        通常传 res.cfg.compounding。
    """
    if rf is None:
        md = _market_data()
        rf = float(md.shibor.reindex(navs.index).mean())
    tbl = pd.DataFrame({c: nav_metrics(navs[c], rf, geometric)
                        for c in navs.columns}).T
    bm = benchmark if benchmark in tbl.index else tbl.index[0]
    tbl.insert(2, "年化超额", tbl["年化收益"] - tbl.loc[bm, "年化收益"])
    tbl.loc[bm, "年化超额"] = np.nan
    tbl.attrs["rf"] = rf
    tbl.attrs["benchmark"] = bm
    tbl.attrs["geometric"] = geometric
    return tbl


# ---------------------------------------------------------------------------
# 策略专属统计
# ---------------------------------------------------------------------------


def strategy_metrics(res) -> pd.Series:
    """损益归因与仓位/换手统计（策略特有，基准无对应项）。

    归因项按**产生该日损益的持仓所依托的名义基数**逐日归一（见
    HedgeResult.pnl_base）。fixed 口径下该基数恒为名义本金，三项之和精确等于
    对冲端总收益；daily 口径下三项之和为算术归因和，与几何总收益之差单列为
    "复利交叉项"，两者相加仍精确等于对冲端总收益。
    """
    d, lg = res.daily, res.legs
    years = (d.index[-1] - d.index[0]).days / 365.0
    base = res.pnl_base
    spot_pnl = float((d["pnl_spot"] / base).sum())
    basis_pnl = float((d["pnl_basis"] / base).sum())
    fee_total = float((d["fee"] / base).sum())
    total = res.hedge_return                   # 净值口径总收益（daily 下为几何）
    arith = spot_pnl + basis_pnl - fee_total   # 算术归因和（fixed 下即等于 total）
    cross = total - arith
    # 年均换手率：单边成交名义 / 当日名义基数 / 年
    turn = ((d["traded_lots"] * d["settle"] * 200) / d["notional_t"]).sum() / years
    return pd.Series({
        "对冲端总收益": total,
        "  方向性贡献": spot_pnl,
        "  贴水(基差)贡献": basis_pnl,
        "  手续费拖累": -fee_total,
        "  复利交叉项": cross,
        "手续费年化": fee_total / years,
        "贴水贡献年化": basis_pnl / years,
        # 分母取算术归因和而非几何总收益，使占比与分子同口径（fixed 下两者相同）
        "贴水占比": basis_pnl / arith if arith else np.nan,
        "票息计提总额": res.coupon_accrued,
        "覆盖率(对冲/票息)": (res.hedge_return / res.coupon_accrued
                              if res.coupon_accrued else np.nan),
        "  对冲端年化": _annualize(total, years, res.cfg.compounding),
        "  票息年化": _annualize(res.coupon_accrued, years, res.cfg.compounding),
        "  覆盖率(年化)": _coverage_ann(res, years),
        "平均delta": float(d["delta"].mean()),
        "最大delta": float(d["delta"].max()),
        "delta>100%占比": float((d["delta"] > 1.0).mean()),
        "已敲入天数占比": float(d["knocked_in"].mean()),
        "雪球只数": len(lg),
        "敲出只数": int(lg["scenario"].isin([1, 5]).sum()),
        "敲入亏损只数": int((lg["scenario"] == 4).sum()),
        "平均票息": float(lg["coupon"].mean()),
        "平均存续年数": float(lg["life_years"].mean()),
        "换月次数": int(d["is_roll"].sum()),
        "累计成交手数": float(d["traded_lots"].sum()),
        "年均换手率(倍)": turn,
    })


def leg_table(res) -> pd.DataFrame:
    """分腿明细，日期已转 date、便于直接展示（ladder 下含 slot 列）。"""
    lg = res.legs.copy()
    for c in ("issue_date", "end_date", "ki_date"):
        lg[c] = pd.to_datetime(lg[c]).dt.date
    cols = ["issue_date", "end_date", "s0", "coupon", "scenario_label",
            "life_years", "ki_date", "end_spot_ratio",
            "leg_hedge_pnl", "leg_fee", "leg_coupon_accrued", "leg_gap"]
    if "slot" in lg.columns and res.cfg.n_slots > 1:
        cols = ["slot"] + cols
    return lg[cols]


# ---------------------------------------------------------------------------
# 槽位诊断（ladder 专用）
# ---------------------------------------------------------------------------

def slot_navs(res) -> pd.DataFrame:
    """各槽位的**子账户诊断净值** + 组合净值（"组合"列）。

    用同一份逐槽位 delta 序列（res.daily 的 delta_<slot> 列）换一种记账：
    每个槽位当作一个独立的小账户（等分资本、各自复利、各自计费），与策略级
    净值（单一账户、新腿名义锚定组合净值、净额计费）对照。

    这是诊断工具，不是正式净值：
      * fixed 口径、免手续费下，子账户均值与策略级净值严格恒等（线性）；
      * leg 口径下两者之差 = 槽位间资本再平衡效应 + 净额计费节省，
        是有业务含义的量，不是误差；
      * 四条子账户曲线的离散度，度量"错开一至数周入场"的结局差异
        ——月内尺度的入场时点敏感性；
      * rampup_mode="full" 的结果下，本诊断仍按等分资本重放（与策略级的
        时变权重路径不同口径），建仓期段仅供参考。
    """
    if res.cfg.n_slots == 1:
        return pd.DataFrame({"组合": res.daily["nav"]})
    chain = _chain(res.cfg.roll_buffer_days)
    sched = chain.build_schedule(res.cfg.hedge_contract,
                                 roll_window_days=res.cfg.roll_window_days)
    out = {}
    for k in range(res.cfg.n_slots):
        nm = f"slot{k}"
        resets = (list(res.legs.loc[res.legs["slot"] == k, "issue_date"])
                  if res.cfg.notional_mode == "leg" else None)
        h = run_hedge(res.daily[f"delta_{nm}"], sched,
                      notional=res.cfg.notional,
                      round_lots=res.cfg.round_lots, close_at_end=True,
                      fee_rate=res.cfg.fee_rate,
                      notional_mode=res.cfg.notional_mode,
                      reset_dates=resets)
        out[nm] = h.daily["nav"]
    out["组合"] = res.daily["nav"]
    return pd.DataFrame(out)


def slot_dispersion(res) -> pd.Series:
    """槽位离散度小结：四条子账户净值的终点分布与策略级的对照。"""
    navs = slot_navs(res)
    slot_cols = [c for c in navs.columns if c != "组合"]
    ends = navs[slot_cols].iloc[-1]
    return pd.Series({
        "槽位数": len(slot_cols),
        "子账户终值均值": float(ends.mean()),
        "子账户终值最高": float(ends.max()),
        "子账户终值最低": float(ends.min()),
        "子账户终值极差": float(ends.max() - ends.min()),
        "子账户终值标准差": float(ends.std(ddof=0)),
        "策略级终值": float(navs["组合"].iloc[-1]),
        "策略级-子账户均值": float(navs["组合"].iloc[-1] - ends.mean()),
    })


if __name__ == "__main__":
    try:
        from .backtest import run_backtest, BacktestConfig
    except ImportError:
        from backtest import run_backtest, BacktestConfig

    from dataclasses import replace as _replace

    # n_slots=1：单链条即可覆盖三种口径的对照逻辑（ladder 演示见 backtest.py）
    base_cfg = BacktestConfig(start_date="2019-01-04", max_legs=3, n_slots=1)
    for mode in ("fixed", "leg", "daily"):
        r = run_backtest(_replace(base_cfg, notional_mode=mode))
        navs = build_benchmarks(r)
        tbl = compare_navs(navs, geometric=r.cfg.compounding)
        print("=" * 78)
        print(f"notional_mode = {mode}   "
              f"（年化口径：{'几何' if r.cfg.compounding else '线性'}）")
        print("=" * 78)
        print(f"无风险利率（区间均值）= {tbl.attrs['rf']:.3%}   "
              f"超额基准 = {tbl.attrs['benchmark']}\n")
        print(tbl[["总收益", "年化收益", "年化超额", "年化波动", "夏普",
                   "最大回撤", "回撤起始日", "回撤谷底日", "回撤修复日"]].to_string())
        sm = strategy_metrics(r)
        print("\n策略统计:")
        print(sm.to_string())
        recon = (sm["  方向性贡献"] + sm["  贴水(基差)贡献"]
                 + sm["  手续费拖累"] + sm["  复利交叉项"])
        print(f"\n归因加总校验: {recon:+.8%} vs 对冲端总收益 "
              f"{sm['对冲端总收益']:+.8%}  差 {abs(recon-sm['对冲端总收益'])*1e4:.2e}bp\n")
