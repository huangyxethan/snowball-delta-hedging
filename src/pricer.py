# -*- coding: utf-8 -*-
"""
pricer.py —— 第2层：统一定价门面

组装 mc_engine（随机）与 payoff（判定），对外提供三个动作：

    solve_coupon(terms, snap)          发行日反解票息（使 PV = 1）
    value(terms, snap, value_date, …)  任意估值日的现值（需已填票息）
    delta(terms, snap, value_date, …)  任意估值日的 delta（需已填票息）

method="mc" 现已实现；method="pde" 是预留接口，pde_engine 完成后接入，
调用方式完全不变——两引擎互验就是把 method 换个字符串。

三个关键机制（都依赖前两层的铺垫）：

1. 反解票息不用迭代求根。
   payoff 层保住了 PV(C) = A + C·B 的线性，故用同一批路径求 PV(0) 与
   PV(1) 两点，即可精确解出 C* = (1 − PV(0)) / (PV(1) − PV(0))。
   分子分母同源，MC噪声在比值中大幅相消。

2. delta 用公共随机数的中心差分。
   mc_engine 返回相对路径（初值1），对 S(1±h) 的两次估值复用同一
   路径数组，只重跑 payoff 判定。随机噪声同源相消，差分才不会被
   MC噪声淹没；顺带省掉一半模拟时间。

3. 存续期估值只模拟剩余寿命。
   terms.grid_from(value_date) 给出剩余网格，路径从当前现价出发；
   已敲入状态经 knocked_in 传给 payoff。票息计提仍按发行日起算的
   合同口径，贴现只贴剩余时间（payoff.valuation_tau 处理）。

单位约定：PV 以名义本金=1 计；delta = dPV/dS（单位：1/指数点）。
直观的"delta敞口比例"（雪球语境常说的 delta 100%）= delta × **当日现价 snap.s**，
即美元delta / 名义本金。归一化必须用当日现价，不能用期初价 s0——hedger 随后
按当日结算价把比例换回手数，两步的价格口径不一致会引入 s0/settle_t 的因子。
"""

import numpy as np

try:
    from .snowball_contract import SnowballTerms
    from .market_params import MarketSnapshot
    from .mc_engine import simulate_relative_paths
    from .payoff import PayoffResult, evaluate_paths
except ImportError:
    from snowball_contract import SnowballTerms
    from market_params import MarketSnapshot
    from mc_engine import simulate_relative_paths
    from payoff import PayoffResult, evaluate_paths

# MC 默认参数（内存约 n_paths × 网格长 × 8字节：10万条 × 一年期约200MB）
N_PATHS = 100_000
SEED = 2024
DELTA_BUMP = 0.01   # delta 中心差分的相对扰动（现价的 ±1%）


def _check_method(method: str, pde_kw: dict = None) -> None:
    if method not in ("mc", "pde"):
        raise ValueError(f"未知定价方法 {method!r}，可选: 'mc', 'pde'")
    if method == "mc" and pde_kw:
        raise TypeError(f"method='mc' 不接受 PDE 参数: {sorted(pde_kw)}")


def _pde():
    """惰性导入 pde_engine（避免 import pricer 时强制引入 scipy）。"""
    try:
        from . import pde_engine
    except ImportError:
        import pde_engine
    return pde_engine


def _require_coupon(terms: SnowballTerms) -> float:
    if terms.coupon is None:
        raise ValueError("条款尚未填票息：先 solve_coupon 再 with_coupon")
    return terms.coupon


def mc_payoff_result(terms: SnowballTerms, snap: MarketSnapshot,
                     value_date=None, knocked_in: bool = False,
                     n_paths: int = N_PATHS, seed: int = SEED) -> PayoffResult:
    """跑一次 模拟+判定，返回逐路径线性组件（诊断与组装的公共底座）。"""
    grid = terms.grid_from(value_date)
    rel = simulate_relative_paths(snap.repo, snap.sigma,
                                  len(grid) - 1, n_paths, seed)
    return evaluate_paths(terms, snap.s * rel, value_date, knocked_in)


# ---------------------------------------------------------------------------
# 三个对外动作
# ---------------------------------------------------------------------------

def solve_coupon(terms: SnowballTerms, snap: MarketSnapshot, method: str = "mc",
                 n_paths: int = N_PATHS, seed: int = SEED, **pde_kw) -> float:
    """发行日反解票息：找 C* 使 PV(C*) = 1（凭证口径）。

    要求 snap 是发行日的快照（snap.s 即 s0）。
    `**pde_kw`（n_space / halfwidth / theta / rannacher / ki_node）仅在
    method="pde" 时透传给 pde_engine；method="mc" 时必须为空。
    """
    _check_method(method, pde_kw)
    if abs(snap.s / terms.s0 - 1) > 1e-9:
        raise ValueError("solve_coupon 需要发行日快照（snap.s 应等于 terms.s0）")
    if method == "pde":
        return _pde().pde_solve_coupon(terms, snap, **pde_kw)
    res = mc_payoff_result(terms, snap, None, False, n_paths, seed)
    p0, p1 = res.pv(0.0, snap.r), res.pv(1.0, snap.r)
    return (1.0 - p0) / (p1 - p0)


def value(terms: SnowballTerms, snap: MarketSnapshot,
          value_date=None, knocked_in: bool = False, method: str = "mc",
          n_paths: int = N_PATHS, seed: int = SEED, **pde_kw) -> float:
    """估值日现值（名义本金=1）。terms 须已填票息。"""
    _check_method(method, pde_kw)
    c = _require_coupon(terms)
    if method == "pde":
        return _pde().pde_value(terms, snap, value_date, knocked_in, **pde_kw)
    res = mc_payoff_result(terms, snap, value_date, knocked_in, n_paths, seed)
    return res.pv(c, snap.r)


def delta(terms: SnowballTerms, snap: MarketSnapshot,
          value_date=None, knocked_in: bool = False, method: str = "mc",
          n_paths: int = N_PATHS, seed: int = SEED,
          bump: float = DELTA_BUMP, **pde_kw) -> float:
    """估值日 delta = dPV/dS（1/指数点）。公共随机数中心差分。

    敞口比例（"delta 100%"口径）= 返回值 × snap.s（**当日现价**，非期初价）。
    """
    _check_method(method, pde_kw)
    c = _require_coupon(terms)
    if method == "pde":
        return _pde().pde_delta(terms, snap, value_date, knocked_in, **pde_kw)
    grid = terms.grid_from(value_date)
    rel = simulate_relative_paths(snap.repo, snap.sigma,
                                  len(grid) - 1, n_paths, seed)
    s = snap.s
    pv_up = evaluate_paths(terms, (s * (1 + bump)) * rel,
                           value_date, knocked_in).pv(c, snap.r)
    pv_dn = evaluate_paths(terms, (s * (1 - bump)) * rel,
                           value_date, knocked_in).pv(c, snap.r)
    return (pv_up - pv_dn) / (2 * bump * s)


if __name__ == "__main__":
    import time

    try:
        from .data_loader import load_market_data
        from .futures_chain import FuturesChain
        from .market_params import build_market_params
        from .snowball_contract import make_snowball
    except ImportError:
        from data_loader import load_market_data
        from futures_chain import FuturesChain
        from market_params import build_market_params
        from snowball_contract import make_snowball

    md = load_market_data()
    params = build_market_params(md, FuturesChain(md))
    d0 = "2021-06-15"
    snap0 = params.snapshot(d0)

    # ---- 1) 基准一年期 75/103 雪球：反解票息 --------------------------------
    t1y = make_snowball(md.trading_days, d0, s0=snap0.s)   # 默认 75/103, 12M, 锁3
    tic = time.perf_counter()
    c1 = solve_coupon(t1y, snap0)
    print(f"[一年期 75/103] 票息 C* = {c1:.4%}  （耗时 {time.perf_counter()-tic:.2f}s）")

    t1y = t1y.with_coupon(c1)
    pv = value(t1y, snap0)
    print(f"  自洽检验 PV(C*) = {pv:.6f}（应≈1）")

    c1b = solve_coupon(t1y, snap0, seed=SEED + 1)
    print(f"  换种子重解: {c1b:.4%}（与上差 {abs(c1b-c1)*1e4:.1f}bp，MC噪声量级）")

    dlt = delta(t1y, snap0)
    print(f"  发行日 delta = {dlt:.6f} /点  => 敞口比例 {dlt*snap0.s:.1%}\n")

    # ---- 2) 两年期 70/103（回测基准条款） ------------------------------------
    t2y = make_snowball(md.trading_days, d0, s0=snap0.s,
                        ki_pct=0.70, ko_pct=1.03, tenor_months=24, lockup_months=2)
    c2 = solve_coupon(t2y, snap0)
    t2y = t2y.with_coupon(c2)
    print(f"[两年期 70/103] 票息 C* = {c2:.4%}  "
          f"(发行日delta敞口 {delta(t2y, snap0)*snap0.s:.1%})\n")

    # ---- 3) 存续期估值：临近首个敲出观察日、现价远超敲出线 -------------------
    dv = "2021-09-14"       # 首个敲出观察日(09-15)前一天
    snapv = params.snapshot(dv)
    pv_v = value(t2y, snapv, value_date=dv)
    print(f"[存续期] {dv}（首个观察日前一天，现价 {snapv.s:.0f} vs 敲出线 {t2y.ko_level:.0f}）:")
    print(f"  PV = {pv_v:.4f}（明日几乎必敲出 => ≈ 1+C*tau 的贴现值 "
          f"{np.exp(-snapv.r*(t2y.ko_obs_taus[0]-t2y.tau(dv)))*(1+c2*t2y.ko_obs_taus[0]):.4f}）")
    print(f"  delta敞口 = {delta(t2y, snapv, value_date=dv)*snapv.s:.1%}（结局已定 => 对现价不敏感）\n")

    # ---- 4) 敲入状态的影响：同一天、同一现价，仅状态不同 ---------------------
    dv2 = "2022-04-26"      # 2022年低点附近，现价距敲入线不远
    snapv2 = params.snapshot(dv2)
    pv_no = value(t2y, snapv2, value_date=dv2, knocked_in=False)
    pv_ki = value(t2y, snapv2, value_date=dv2, knocked_in=True)
    print(f"[敲入状态] {dv2}（现价 {snapv2.s:.0f}，S/S0 = {snapv2.s/t2y.s0:.1%}）:")
    print(f"  未敲入 PV = {pv_no:.4f}   已敲入 PV = {pv_ki:.4f}   "
          f"差 = {pv_no-pv_ki:.4f}（敲入吃掉的价值）")
