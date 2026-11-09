# -*- coding: utf-8 -*-
"""
payoff.py —— 第2层：雪球 payoff 判定（MC 引擎的确定性部分）

全项目最容易写错的文件。全部规则以下述五种情形为准：

    情形1  敲出（此前从未敲入）：某敲出观察日收盘 >= 敲出线，立即终止，
           兑付 1 + C×tau_敲出
    情形2  未敲入未敲出，持有到期：兑付 1 + C×T
    情形3  敲入过、未敲出、S_T >= S0：兑付 1（保本无票息）
    情形4  敲入过、未敲出、S_T <  S0：兑付 1 + (S_T/S0 − 1)（承担跌幅）
    情形5  敲入后又敲出：同情形1（敲出优先，敲入记录被覆盖）

判定惯例（与 snowball_contract.py 头部备忘一致）：
    敲出 = 观察日收盘 >= 敲出线；敲入 = 观察日收盘 < 敲入线（严格小于）。
    均为"观察日收盘判定"，非盘中触及。

对票息的线性结构（反解票息的关键）：
    每条路径的兑付都可写成  fixed + C × accrual：
        情形1/5: fixed=1,            accrual=tau_敲出
        情形2:   fixed=1,            accrual=T
        情形3/4: fixed=1+min(S_T/S0−1, 0), accrual=0
    因此 PV(C) = E[e^{-r·tau_pay}·fixed] + C·E[e^{-r·tau_pay}·accrual]
             = A + C·B  —— 对 C 严格线性，pricer 两次求值即可精确反解。
    本模块把 (fixed, accrual, tau_pay) 三个数组原样交给 pricer，
    不提前把票息代入，就是为了保住这个线性结构。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .snowball_contract import SnowballTerms
except ImportError:
    from snowball_contract import SnowballTerms

SCENARIO_LABELS = {
    1: "1 敲出(未敲入)",
    2: "2 未敲入未敲出",
    3: "3 敲入未敲出,S_T>=S0",
    4: "4 敲入未敲出,S_T<S0",
    5: "5 敲入后敲出",
}


@dataclass(frozen=True)
class PayoffResult:
    """逐路径判定结果（对票息 C 保持线性，未贴现、未代入票息）。

    时间基准：pay_tau / accrual 一律从"发行日"起算（票息计提的合同口径）；
    valuation_tau 是估值日相对发行日的时点，贴现只贴 (pay_tau − valuation_tau)
    这段剩余时间。发行日估值时 valuation_tau = 0。
    """
    pay_tau: np.ndarray    # 兑付时点（年，自发行日起算）
    fixed: np.ndarray      # 兑付中与票息无关的部分
    accrual: np.ndarray    # 票息计提年数（兑付 = fixed + C × accrual）
    scenario: np.ndarray   # 情形编号 1..5
    valuation_tau: float = 0.0   # 估值日时点（自发行日起算）

    def pv(self, coupon: float, r: float) -> float:
        """给定票息与贴现率、贴现到估值日的现值（凭证口径，名义本金=1）。"""
        disc = np.exp(-r * (self.pay_tau - self.valuation_tau))
        return float(np.mean(disc * (self.fixed + coupon * self.accrual)))

    def pv_stderr(self, coupon: float, r: float) -> float:
        """PV 的蒙特卡洛标准误。"""
        disc = np.exp(-r * (self.pay_tau - self.valuation_tau))
        x = disc * (self.fixed + coupon * self.accrual)
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    def scenario_probs(self) -> pd.Series:
        """五种情形的发生概率（诊断/展示用）。"""
        n = len(self.scenario)
        counts = pd.Series(self.scenario).value_counts().sort_index()
        return pd.Series({SCENARIO_LABELS[k]: v / n for k, v in counts.items()})


def evaluate_paths(terms: SnowballTerms, paths: np.ndarray,
                   value_date=None, knocked_in: bool = False) -> PayoffResult:
    """对一批价格路径逐条判定五种情形。

    Parameters
    ----------
    terms : 条款。敲入/敲出线、观察日、tau 全部由此提供。
    paths : shape (n_paths, len(terms.grid_from(value_date))) 的价格路径矩阵，
            列与网格一一对应（列0=估值日，值=当前现价；发行日估值时即 s0）。
            既可以是 MC 模拟路径，也可以是真实历史路径（回测判定复用本函数）。
    value_date : 估值日（None=发行日）。存续期中估值时传入当日，
                 只判定该日之后的观察日；估值日当天的收盘状态由调用方
                 负责（已敲出的产品不该再来估值，已敲入用 knocked_in 传入）。
    knocked_in : 估值日之前是否已发生敲入。True 时所有路径视同已敲入
                 （只可能落入情形3/4/5）。发行日估值恒为 False。
    """
    grid = pd.DatetimeIndex(terms.grid_from(value_date))
    n_paths, n_cols = paths.shape
    if n_cols != len(grid):
        raise ValueError(f"路径列数 {n_cols} 与网格长度 {len(grid)} 不符")

    # 剩余的敲出观察日及其在网格中的列号（一次定位，全程复用）
    future_ko = [(d, t) for d, t in zip(terms.ko_obs_dates, terms.ko_obs_taus)
                 if d > grid[0]]
    ko_dates = pd.DatetimeIndex([d for d, _ in future_ko])
    ko_taus = np.asarray([t for _, t in future_ko])
    ko_pos = grid.get_indexer(ko_dates)
    if (ko_pos < 0).any():
        raise ValueError("存在不在网格上的敲出观察日")

    # ---- 敲出判定：首个 收盘>=敲出线 的观察日 --------------------------------
    ko_hit = paths[:, ko_pos] >= terms.ko_level          # (n_paths, n_obs)
    has_ko = ko_hit.any(axis=1)
    first_ko = ko_hit.argmax(axis=1)                     # 首个True的列号(无则0,配合has_ko用)
    ko_tau = ko_taus[first_ko]                           # 敲出时点
    ko_grid_pos = ko_pos[first_ko]                       # 敲出日在网格中的位置

    # ---- 敲入判定：任一交易日 收盘<敲入线（列0是估值日，不观察） --------------
    ki_hit = paths[:, 1:] < terms.ki_level
    if knocked_in:
        has_ki = np.ones(n_paths, dtype=bool)            # 历史上已敲入
        ki_before_ko = has_ko.copy()                     # 敲出的必为"先敲入后敲出"
    else:
        has_ki = ki_hit.any(axis=1)
        first_ki_grid = 1 + ki_hit.argmax(axis=1)        # 敲入日在网格中的位置
        # 敲出路径上，敲入是否发生在敲出之前（仅影响情形1/5的区分，不影响兑付）
        ki_before_ko = has_ki & (first_ki_grid < ko_grid_pos)

    # ---- 组装三个线性组件 ---------------------------------------------------
    s_t = paths[:, -1]
    t_mat = terms.maturity_tau

    pay_tau = np.full(n_paths, t_mat)
    fixed = np.ones(n_paths)
    accrual = np.full(n_paths, t_mat)
    scenario = np.full(n_paths, 2, dtype=np.int8)

    # 敲出（情形1/5）：兑付提前到敲出日，票息按实际存续计提，敲入记录被覆盖
    pay_tau[has_ko] = ko_tau[has_ko]
    accrual[has_ko] = ko_tau[has_ko]
    scenario[has_ko] = np.where(ki_before_ko[has_ko], 5, 1)

    # 敲入且未敲出（情形3/4）：无票息，承担跌幅（保本于 S_T>=S0）
    ki_no_ko = has_ki & ~has_ko
    fixed[ki_no_ko] = 1.0 + np.minimum(s_t[ki_no_ko] / terms.s0 - 1.0, 0.0)
    accrual[ki_no_ko] = 0.0
    scenario[ki_no_ko] = np.where(s_t[ki_no_ko] >= terms.s0, 3, 4)

    return PayoffResult(pay_tau=pay_tau, fixed=fixed,
                        accrual=accrual, scenario=scenario,
                        valuation_tau=terms.tau(grid[0]))


if __name__ == "__main__":
    try:
        from .data_loader import load_market_data
        from .snowball_contract import make_snowball
        from .mc_engine import simulate_relative_paths
    except ImportError:
        from data_loader import load_market_data
        from snowball_contract import make_snowball
        from mc_engine import simulate_relative_paths

    md = load_market_data()

    # ---- 测试1：真实历史路径判定（与notebook里肉眼判定对照） ----------------
    terms = make_snowball(md.trading_days, "2021-06-15", s0=md.spot("2021-06-15"),
                          ki_pct=0.70, ko_pct=1.03, tenor_months=24, lockup_months=2)
    real_path = md.index["close"].reindex(pd.DatetimeIndex(terms.sim_grid)).to_numpy()[None, :]
    res = evaluate_paths(terms, real_path)
    k = int(np.argmax(real_path[0, pd.DatetimeIndex(terms.sim_grid).get_indexer(
        pd.DatetimeIndex(terms.ko_obs_dates))] >= terms.ko_level))
    print("[真实路径] 2021-06-15发行 两年期 70/103:")
    print(f"  情形 = {SCENARIO_LABELS[int(res.scenario[0])]}")
    print(f"  兑付时点 tau = {res.pay_tau[0]:.4f}（对应 {terms.ko_obs_dates[k].date()}）")
    print(f"  若票息15%: 兑付金额 = {res.fixed[0] + 0.15 * res.accrual[0]:.4f}\n")

    # ---- 测试2：退化为债券（敲入线极低、敲出线极高 => 必然情形2） ------------
    bond = make_snowball(md.trading_days, "2021-06-15", s0=md.spot("2021-06-15"),
                         ki_pct=1e-6, ko_pct=1e6, tenor_months=12, lockup_months=3)
    paths = md.spot("2021-06-15") * simulate_relative_paths(
        -0.10, 0.20, len(bond.sim_grid) - 1, 10_000, seed=1)
    res_b = evaluate_paths(bond, paths)
    r, c = 0.03, 0.15
    analytic = np.exp(-r * bond.maturity_tau) * (1 + c * bond.maturity_tau)
    print("[退化债券] 全部路径应为情形2，PV应精确等于解析值:")
    print(f"  情形分布: {dict(pd.Series(res_b.scenario).value_counts())}")
    print(f"  MC PV = {res_b.pv(c, r):.8f}  vs 解析 e^(-rT)(1+CT) = {analytic:.8f}")
    print(f"  标准误 = {res_b.pv_stderr(c, r):.2e}（路径无分岔，应为0）\n")

    # ---- 测试3：MC 全景（用真实市场参数，看五种情形的概率分布） --------------
    try:
        from .futures_chain import FuturesChain
        from .market_params import build_market_params
    except ImportError:
        from futures_chain import FuturesChain
        from market_params import build_market_params
    snap = build_market_params(md, FuturesChain(md)).snapshot("2021-06-15")
    paths = snap.s * simulate_relative_paths(
        snap.repo, snap.sigma, len(terms.sim_grid) - 1, 100_000, seed=7)
    res_mc = evaluate_paths(terms, paths)
    print(f"[MC全景] 两年期70/103, repo={snap.repo:.2%}, sigma={snap.sigma:.2%}, 10万条路径:")
    print(res_mc.scenario_probs().round(4).to_string())
    print(f"\n  PV(C=15%) = {res_mc.pv(0.15, snap.r):.4f} ± {res_mc.pv_stderr(0.15, snap.r):.4f}")
    print(f"  PV(C=0%)  = {res_mc.pv(0.0, snap.r):.4f}  （票息越高PV越高，反解即找PV=1的C）")
