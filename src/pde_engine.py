# -*- coding: utf-8 -*-
"""
pde_engine.py —— 第2层：有限差分定价引擎（Crank-Nicolson）

核心思想：雪球虽然路径依赖，但只依赖"是否已敲入"这一个布尔状态。
因此拆成两层价值函数，在同一 (S, t) 网格上同步反向推进：

    V_in (S, t)   已敲入状态
    V_out(S, t)   未敲入状态

两层通过"区域赋值"耦合（每个时间步 = 一个交易日）：

    * 每个交易日（日频敲入观察）：S <= KI 区域令 V_out <- V_in
    * 每个敲出观察日（月频）  ：S >= KO 区域令两层同时 <- 1 + C*tau
    * 到期：V_in <- min(S_T/S0, 1)（=1+min(S_T/S0-1,0)），V_out <- 1 + C*T

【为什么是区域赋值而不是在 KI 处设 Dirichlet 边界】
把 V_out 的求解域截到 [KI, S_max] 并在 KI 施加边界条件，等价于把敲入
改成了连续观察：单日扩散中触及 KI 的路径全被杀掉。而合约只看每日收盘，
日内跌破再收回不构成敲入。区域赋值才与离散观察一致：单日扩散对全部
落点积分，落在 S<=KI 的部分接 V_in，其余接 V_out。（参考实现实测过
这一错误会把期权价值压低约8%。）

数值处理（沿用 Pooley-Vetzal-Forsyth 一系的标准手法）：
    * 网格对齐：调整 dx 使 ln(KI)、ln(KO) 恰好落在节点上；
    * 障碍节点取单元平均：敲出在 S=KO 处是 O(1) 跳变，点赋值只有一阶
      收敛，节点取左右极限的半和可恢复二阶；
    * Rannacher 起步：每次遇到间断（终值、每个敲出观察日）后的若干步
      用全隐式（theta=1）抹平振荡，其余用 CN（theta=0.5）；
    * 敲入线处的 delta 用单边差分：V_out 在 KI 处有真实跳变，跨越间断的
      中心差分会给出 (跳变)/(2dx) 的假 delta。未敲入头寸只会在 S>KI 处
      被查询，故取上侧导数。

双时钟（与 mc_engine 严格一致，保证两引擎可互验）：
    * 扩散：每个交易日 dt = 1/252（波动率的年化口径）；
    * 贴现与票息计提：自然日/365（合同口径）。相邻交易日隔几个自然日，
      该步的贴现就贴几天——周末/长假的资金时间价值不会丢。

方程（x = ln S，反向推进）：
    dV/dtau = (sigma^2/2) V_xx + (repo - sigma^2/2) V_x - r V
                └── 按交易日时钟 ──┘                  └自然日时钟┘

对 MC 的两点优势：单次求解约0.1秒（快约50倍），delta 无随机噪声、
且一次求解给出整条 (S -> V, delta) 曲线。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

try:
    from .snowball_contract import SnowballTerms
    from .market_params import MarketSnapshot
except ImportError:
    from snowball_contract import SnowballTerms
    from market_params import MarketSnapshot

# 默认数值参数
N_SPACE = 1201                 # 目标空间节点数
X_HALFWIDTH = float(np.log(5.0))   # 网格半宽：S0/5 ~ 5*S0
THETA = 0.5                    # CN
RANNACHER = 2                  # 间断后用全隐式的交易日区间数
# 每个 Rannacher 区间细分的全隐式子步数。
# 它是回测耗时的主要驱动：单次求解的三对角求解次数 =
#   剩余交易日数 + (1 + 剩余敲出观察日数) x RANNACHER x (SUBSTEPS - 1)
# 两年期合约上第二项占 60~70%。精度代价（对解析基准的 delta 偏差）：
#   16 -> +0.04pt,  8 -> +0.15pt,  4 -> +0.37pt
# 取 8：0.15pt 远低于回测中其他误差源（1亿名义下 IC 手数取整即 0.67pt），
# 而耗时较 16 减半。定价精度验证若需更严，可在调用处传 rannacher / 改本值。
RANNACHER_SUBSTEPS = 8
# 敲入节点(S=KI)的取值约定。敲入是"日度、按收盘、严格低于敲入线"的判定，
# 每个交易日都要施加一次，故该节点的处理会被累计 250 次，选择很敏感：
#   "out" —— 节点仍算未敲入（与 MC 的 S < KI 严格一致）
#   "avg" —— 取两层均值（障碍单元平均，标准做法；默认）
#   "in"  —— 节点算已敲入（等价于 S <= KI）
# 三者互为上下界、随 dx->0 收敛到同一极限。实测在默认网格上 "avg" 与 MC
# 的票息差最小（+8.6/+0.3/+3.3 bp，均在 MC 噪声内），故取为默认。
KI_NODE = "avg"
DT_DIFF = 1.0 / 252            # 扩散时钟：每交易日
DAYS_PER_YEAR = 365.0          # 贴现时钟：自然日


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Grid:
    x: np.ndarray      # ln(S) 节点
    s: np.ndarray      # S 节点
    dx: float
    i_ki: int          # ln(KI) 所在节点
    i_ko: int          # ln(KO) 所在节点


def _build_grid(s0: float, ki_level: float, ko_level: float,
                halfwidth: float, n_target: int) -> _Grid:
    """对数网格，两条障碍精确落在节点上。

    障碍若远在定价域外（退化测试常把 KO 设为 1e6），截断到域边界：
    等价于"无该障碍"（到达概率可忽略），否则障碍间跨度会撑爆网格。
    """
    x_c = np.log(s0)
    x_lo0, x_hi0 = x_c - halfwidth, x_c + halfwidth
    x_ki = float(np.clip(np.log(ki_level), x_lo0, x_hi0))
    x_ko = float(np.clip(np.log(ko_level), x_lo0, x_hi0))
    x_lo = min(x_lo0, x_ki - 0.1)
    x_hi = max(x_hi0, x_ko + 0.1)

    # 令两障碍之间恰好含整数个 dx => 都在节点上
    span = x_ko - x_ki
    m = max(20, int(round(n_target * span / (x_hi - x_lo))))
    dx = span / m
    n_below = int(np.ceil((x_ki - x_lo) / dx))
    n_above = int(np.ceil((x_hi - x_ko) / dx))
    x = x_ki + dx * np.arange(-n_below, m + n_above + 1)
    return _Grid(x=x, s=np.exp(x), dx=dx, i_ki=n_below, i_ko=n_below + m)


# ---------------------------------------------------------------------------
# theta 格式的一步
# ---------------------------------------------------------------------------

def _step_matrices(n: int, dx: float, sigma: float, repo: float, r: float,
                   dt_diff: float, dt_disc: float, theta: float):
    """LHS带状矩阵 + RHS三对角系数。空间算子按交易日时钟，-rV 按自然日时钟。"""
    a = 0.5 * sigma ** 2
    b = repo - 0.5 * sigma ** 2

    lower = dt_diff * (a / dx ** 2 - b / (2 * dx))
    diag = dt_diff * (-2 * a / dx ** 2) - r * dt_disc
    upper = dt_diff * (a / dx ** 2 + b / (2 * dx))

    ab = np.zeros((3, n))
    ab[0, 1:] = -theta * upper
    ab[1, :] = 1.0 - theta * diag
    ab[2, :-1] = -theta * lower
    ab[1, 0] = 1.0; ab[0, 1] = 0.0     # Dirichlet 下边界
    ab[1, -1] = 1.0; ab[2, -2] = 0.0   # Dirichlet 上边界
    return ab, (1.0 - theta) * np.array([lower, diag, upper])


def _step(v: np.ndarray, ab: np.ndarray, rhs_coef: np.ndarray,
          v_lo: float, v_hi: float) -> np.ndarray:
    lo, dg, up = rhs_coef
    rhs = v.copy()
    rhs[1:-1] += lo * v[:-2] + dg * v[1:-1] + up * v[2:]
    rhs[0] = v_lo
    rhs[-1] = v_hi
    return solve_banded((1, 1), ab, rhs)


# ---------------------------------------------------------------------------
# 求解结果：整条曲线 + 查询接口
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PDECurves:
    """一次求解的全部产出：估值日的 (S -> 价值, delta) 两层曲线。"""
    s: np.ndarray
    x: np.ndarray
    v_out: np.ndarray     # 未敲入层
    v_in: np.ndarray      # 已敲入层
    d_out: np.ndarray     # dV/dS（未敲入，KI处单边）
    d_in: np.ndarray      # dV/dS（已敲入）

    def value(self, spot: float, knocked_in: bool = False) -> float:
        v = self.v_in if knocked_in else self.v_out
        return float(np.interp(np.log(spot), self.x, v))

    def delta(self, spot: float, knocked_in: bool = False) -> float:
        d = self.d_in if knocked_in else self.d_out
        return float(np.interp(np.log(spot), self.x, d))


def _piecewise_delta(v: np.ndarray, s: np.ndarray, dx: float,
                     breaks: tuple) -> np.ndarray:
    """分段求导：段内中心差分、段端单边，绝不跨越间断点；返回 dV/dS。"""
    n = v.size
    g = np.full(n, np.nan)
    bs = [b for b in breaks if 0 < b < n - 1]
    starts = [0] + [b + 1 for b in bs]
    ends = [b - 1 for b in bs] + [n - 1]
    for a0, b0 in zip(starts, ends):
        if b0 > a0:
            g[a0:b0 + 1] = np.gradient(v[a0:b0 + 1], dx)
        elif b0 == a0:
            g[a0] = 0.0
    for b in bs:
        g[b] = g[b + 1]        # 障碍节点取上侧导数（未敲入头寸只在 S>KI 被查询）
    return g / s               # d/dx -> d/dS


# ---------------------------------------------------------------------------
# 主求解
# ---------------------------------------------------------------------------

def solve_curves(terms: SnowballTerms, snap: MarketSnapshot,
                 coupon: float = None, value_date=None,
                 n_space: int = N_SPACE, halfwidth: float = X_HALFWIDTH,
                 theta: float = THETA, rannacher: int = RANNACHER,
                 ki_node: str = KI_NODE) -> PDECurves:
    """从到期反向推进到估值日，返回两层曲线。

    coupon=None 时用 terms.coupon（须已填）。
    时间网格 = terms.grid_from(value_date)，与 MC 完全同一张网格；
    估值日（j=0）不施加任何观察，同 MC 的"列0不观察"约定。
    """
    c = terms.coupon if coupon is None else coupon
    if c is None:
        raise ValueError("需要票息：传入 coupon 或先 with_coupon")

    dates = pd.DatetimeIndex(terms.grid_from(value_date))
    n_steps = len(dates) - 1
    taus = np.array([terms.tau(d) for d in dates])          # 自然日时钟
    d_nat = np.diff((dates - dates[0]).days)                # 每步跨的自然日数

    # 敲出观察日 -> (网格步号, tau)
    ko_map = {}
    for d, t in zip(terms.ko_obs_dates, terms.ko_obs_taus):
        j = dates.get_indexer([d])[0]
        if j > 0:
            ko_map[int(j)] = t
    t_mat = terms.maturity_tau

    grid = _build_grid(terms.s0, terms.ki_level, terms.ko_level,
                       halfwidth, n_space)
    i_ki, i_ko = grid.i_ki, grid.i_ko

    # ---- 障碍处的区域赋值（单元平均） -------------------------------------
    def apply_ko(v: np.ndarray, val: float) -> None:
        node = 0.5 * (v[i_ko] + val)
        v[i_ko + 1:] = val
        v[i_ko] = node

    def apply_ki(v_out: np.ndarray, v_in: np.ndarray) -> None:
        # 敲入节点的三种约定，见 KI_NODE 说明
        if ki_node == "avg":
            node = 0.5 * (v_out[i_ki] + v_in[i_ki])
        elif ki_node == "in":
            node = v_in[i_ki]
        else:                      # "out"：仅严格低于敲入线才敲入（与MC的 < 一致）
            node = v_out[i_ki]
        v_out[:i_ki] = v_in[:i_ki]
        v_out[i_ki] = node

    # ---- 终值（到期日，j = n_steps） --------------------------------------
    v_in = np.minimum(grid.s / terms.s0, 1.0)               # 1 + min(S/S0-1, 0)
    v_out = np.full(grid.s.size, 1.0 + c * t_mat)
    if n_steps in ko_map:                                    # 到期日也是敲出观察日
        apply_ko(v_in, 1.0 + c * ko_map[n_steps])
        apply_ko(v_out, 1.0 + c * ko_map[n_steps])
    apply_ki(v_out, v_in)

    # ---- 每步矩阵（按 (自然日间隔, 细分数, theta) 缓存；取值只有少数几种） ----
    mat_cache = {}

    def matrices(dn: int, th: float, n_sub: int = 1):
        key = (dn, th, n_sub)
        if key not in mat_cache:
            mat_cache[key] = _step_matrices(grid.s.size, grid.dx, snap.sigma,
                                            snap.repo, snap.r,
                                            DT_DIFF / n_sub,
                                            dn / DAYS_PER_YEAR / n_sub, th)
        return mat_cache[key]

    # ---- 边界条件 ---------------------------------------------------------
    def lower_bc(j: int) -> float:
        # S 极小：必已敲入、不会敲出，到期兑付 S_T/S0。
        # E[S_T] = S*exp(repo*剩余交易日/252)，贴现按剩余自然日。
        m = n_steps - j
        return float(grid.s[0] / terms.s0
                     * np.exp(snap.repo * m * DT_DIFF - snap.r * (t_mat - taus[j])))

    def upper_bc(j: int) -> float:
        # S 极大：必在下一个敲出观察日敲出，取应付金额的贴现值
        nxt = [st for st in ko_map if st > j]
        if nxt:
            st = min(nxt)
            return float((1.0 + c * ko_map[st])
                         * np.exp(-snap.r * (taus[st] - taus[j])))
        return float((1.0 + c * t_mat) * np.exp(-snap.r * (t_mat - taus[j])))

    # ---- 反向推进 ---------------------------------------------------------
    # Rannacher 起步：间断（终值 / 每个敲出观察日）之后的 rannacher 个交易日区间，
    # 每个区间拆成 RANNACHER_SUBSTEPS 个全隐式子步（而非整日全隐式）。
    # 整日全隐式虽能阻尼CN振荡，但只有一阶时间精度；当估值日距观察日仅几步时，
    # 误差无从摊薄（实测可达 2 个delta敞口百分点）。半步配方兼顾阻尼与二阶精度。
    since_jump = 0
    for j in range(n_steps - 1, -1, -1):
        in_rannacher = since_jump < rannacher
        th = 1.0 if in_rannacher else theta
        n_sub = RANNACHER_SUBSTEPS if in_rannacher else 1
        ab, rhs = matrices(int(d_nat[j]), th, n_sub)
        since_jump += 1

        lo, hi = lower_bc(j), upper_bc(j)
        for _ in range(n_sub):
            v_in = _step(v_in, ab, rhs, lo, hi)
            v_out = _step(v_out, ab, rhs, lo, hi)

        if j > 0:                          # 估值日(j=0)不观察，同 MC 约定
            if j in ko_map:                # 月频敲出观察
                val = 1.0 + c * ko_map[j]
                apply_ko(v_in, val)
                apply_ko(v_out, val)
                since_jump = 0             # 新间断，重启 Rannacher
            apply_ki(v_out, v_in)          # 日频敲入观察

    d_in = _piecewise_delta(v_in, grid.s, grid.dx, ())
    d_out = _piecewise_delta(v_out, grid.s, grid.dx, (i_ki,))
    return PDECurves(s=grid.s, x=grid.x, v_out=v_out, v_in=v_in,
                     d_out=d_out, d_in=d_in)


# ---------------------------------------------------------------------------
# 与 pricer 同形的三个动作
# ---------------------------------------------------------------------------

def pde_value(terms: SnowballTerms, snap: MarketSnapshot,
              value_date=None, knocked_in: bool = False, **kw) -> float:
    return solve_curves(terms, snap, None, value_date, **kw).value(snap.s, knocked_in)


def pde_delta(terms: SnowballTerms, snap: MarketSnapshot,
              value_date=None, knocked_in: bool = False, **kw) -> float:
    return solve_curves(terms, snap, None, value_date, **kw).delta(snap.s, knocked_in)


def pde_solve_coupon(terms: SnowballTerms, snap: MarketSnapshot, **kw) -> float:  # noqa: D401
    """反解票息：PDE 的解对票息同样严格线性（终值/敲出值/边界全线性），
    两次求解即得 C* = (1 - PV(0)) / (PV(1) - PV(0))。"""
    if abs(snap.s / terms.s0 - 1) > 1e-9:
        raise ValueError("solve_coupon 需要发行日快照（snap.s 应等于 terms.s0）")
    p0 = solve_curves(terms, snap, 0.0, None, **kw).value(snap.s)
    p1 = solve_curves(terms, snap, 1.0, None, **kw).value(snap.s)
    return (1.0 - p0) / (p1 - p0)


if __name__ == "__main__":
    import time

    try:
        from .data_loader import load_market_data
        from .futures_chain import FuturesChain
        from .market_params import build_market_params
        from .snowball_contract import make_snowball
        from . import pricer
    except ImportError:
        from data_loader import load_market_data
        from futures_chain import FuturesChain
        from market_params import build_market_params
        from snowball_contract import make_snowball
        import pricer

    md = load_market_data()
    params = build_market_params(md, FuturesChain(md))
    d0 = "2021-06-15"
    snap0 = params.snapshot(d0)

    # ---- 1) 退化债券 vs 解析 ---------------------------------------------
    bond = make_snowball(md.trading_days, d0, s0=snap0.s,
                         ki_pct=1e-6, ko_pct=1e6, tenor_months=12, lockup_months=3)
    r, c = snap0.r, 0.15
    pv_pde = solve_curves(bond, snap0, c).value(snap0.s)
    pv_ana = np.exp(-r * bond.maturity_tau) * (1 + c * bond.maturity_tau)
    print(f"[退化债券] PDE {pv_pde:.8f} vs 解析 {pv_ana:.8f}  "
          f"(误差 {abs(pv_pde-pv_ana):.2e})\n")

    # ---- 2) 一年期 75/103：PDE vs MC 票息与delta ---------------------------
    t1y = make_snowball(md.trading_days, d0, s0=snap0.s)
    tic = time.perf_counter()
    c_pde = pde_solve_coupon(t1y, snap0)
    t_pde = time.perf_counter() - tic
    tic = time.perf_counter()
    c_mc = pricer.solve_coupon(t1y, snap0, n_paths=200_000)
    t_mc = time.perf_counter() - tic
    print(f"[一年期 75/103] 票息  PDE {c_pde:.4%} ({t_pde:.2f}s)  "
          f"vs MC {c_mc:.4%} ({t_mc:.2f}s)  差 {abs(c_pde-c_mc)*1e4:.1f}bp")

    t1y = t1y.with_coupon(c_pde)
    d_pde = pde_delta(t1y, snap0)
    d_mc = pricer.delta(t1y, snap0, n_paths=200_000)
    print(f"  发行日delta敞口  PDE {d_pde*snap0.s:.2%}  vs MC {d_mc*snap0.s:.2%}\n")

    # ---- 3) 存续期与敲入状态 vs MC ----------------------------------------
    t2y = make_snowball(md.trading_days, d0, s0=snap0.s,
                        ki_pct=0.70, ko_pct=1.03, tenor_months=24, lockup_months=2)
    c2 = pde_solve_coupon(t2y, snap0)
    t2y = t2y.with_coupon(c2)
    for dv, ki_flag in [("2021-09-14", False), ("2022-04-26", False), ("2022-04-26", True)]:
        snapv = params.snapshot(dv)
        v_pde = pde_value(t2y, snapv, value_date=dv, knocked_in=ki_flag)
        v_mc = pricer.value(t2y, snapv, value_date=dv, knocked_in=ki_flag, n_paths=200_000)
        print(f"[存续期 {dv} knocked_in={ki_flag}]  "
              f"PDE {v_pde:.4f} vs MC {v_mc:.4f}  差 {abs(v_pde-v_mc)*1e4:.1f}bp")

    # ---- 4) 空间网格收敛性 -------------------------------------------------
    print("\n[网格收敛] 一年期票息随 n_space:")
    for n in (301, 601, 1201, 2401):
        print(f"  n_space={n:>5}: C* = {pde_solve_coupon(t1y, snap0, n_space=n):.6%}")
