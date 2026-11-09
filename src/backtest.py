# -*- coding: utf-8 -*-
"""
backtest.py —— 第4层：回测编排

把前三层串成一条完整链路，在给定区间上滚动发行虚拟雪球，输出连续净值。
支持两种发行方式，由 n_slots 控制：

槽位制 ladder（n_slots > 1，默认 4）
------------------------------------
    组合是**一只净值化基金**：一条净值、一个期货账户。"槽位"只是发行日历的
    组织方式——n_slots 条顺序滚动链错峰并行，每条链任一时点至多一只雪球存续，
    组合最多 n_slots 只并存。

    建仓期：槽位 k 的首只雪球在回测起点后第 k × slot_stagger_weeks 周的周一
    网格日发行（周一逢假顺延至该周首个交易日），依次错峰启动。未启动的槽位
    空仓，净值如实反映建仓期的仓位不足。

    槽内滚动：腿终止后，在首个距终止 >= reissue_lag_days 个交易日的周一网格日
    发行下一只（monday_issue=True 时；False 退回"终止后第 lag 个交易日"）。
    再发行错峰（stagger_reissue，默认开）：发行日与其他槽位已用发行日冲突时
    顺延一周再试——多腿相近时点终止（如大行情连环敲出）会把各自的再发行推到
    同一个周一，若不错开，撞车槽位的条款从此完全同化（同 s0、同观察日程、
    同日终止，且再发行永远同日——同化是吸收态），ladder 退化为单雪球；顺延
    规则在冲突处按槽位次序隔周补齐，正是建仓期错峰机制在滚动期的延续。

    名义基数（notional_mode="leg" 下）：每只腿发行时一次性锁定为
        M^(k) = (1/n_slots) × M_0 × NAV_{发行日前收盘}
    腿内冻结。已实现盈亏闲置在净值里，直到任一槽位下次发行新腿时经由新腿的
    基数锚定被重新投入——即"新发雪球按当前总 AUM 的份额定名义"，隐含槽位间
    的资本互通。逐日对冲：各存续腿分别算 delta，聚合成策略级金额敞口后
    **一次调仓**（单一期货账户，手续费按净额成交计），见 hedger 多套筒模式。

顺序滚动 sequential（n_slots = 1）
----------------------------------
    任一时点至多一只虚拟雪球存续：第 k 只在 E_k 收盘终止（敲出或到期），
    第 k+1 只在 E_k 之后第 reissue_lag_days（默认5）个交易日（monday_issue
    时再落到周一网格）发行，期初价取发行日收盘；间隔期（含 E_k 当日）敞口为
    0、票息计提暂停。间隔的动机是执行可行性：敲出与否收盘才能确认，同日建新
    仓不可交易。reissue_lag_days=0 退回"终止当日接续、无空仓日"的旧行为。
    n_slots=1 + weekly_terms=False + monday_issue=False 与历史版本逐日恒等。

    回测覆盖到数据末端：末腿若在数据末端仍未了结（既未敲出也未到期），则在末端
    截断，记为情形 0，并按记账约定于当日平仓（这是记账动作，非合约事件）。
    该腿的条款仍按完整存续期构造——定价需要完整的敲出观察日程，故用
    "真实交易日历 + 末端之后的合成工作日"生成条款（见 _extended_calendar）。
    allow_truncated_leg=False 可退回"装不下完整一只就停止发行"的旧行为。
    ladder 下末端最多 n_slots 只腿同时截断。

条款日历（weekly_terms）
------------------------
    True（默认）= 周网格条款：周一发行、每 ko_interval_weeks 周的周五观察，
    见 snowball_contract.make_snowball_weekly。虚拟雪球的自定义日历，使错峰
    结构成为制度（理想态下每个周五组合恰有一只腿观察）。
    False = 月度对日条款（市场标准，历史行为）。
    两套条款的票息有 1~2pp 内的系统性差异，结果不可跨条款比较。

单腿流程
--------
    1. 取发行日市场快照                        market_params
    2. 造条款、反解并锁定票息                   snowball_contract + pricer
    3. 沿真实历史路径推进状态机                  每日判定敲入；观察日判定敲出
    4. 逐日重算 delta（当日现价 + 当日参数）      pricer

全周期
------
    5. 各槽位的腿 delta 各自首尾相接 → 按槽位分列对齐到全周期日历 →
       hedger（多套筒，单一账户一次调仓）→ 逐日损益与净值
    6. 票息对照线：每槽位在**自己的财富**上跨腿累计（客户滚动申购、本息滚入
       下一只），组合线 = 各槽位财富的等权平均：
         fixed     : sum_k w_k × [sum_{已完成腿} C_j* tau_j + C* tau(t)]
         leg/daily : sum_k w_k × [prod_{已完成腿}(1 + C_j* tau_j) × (1 + C* tau(t))] − 1
       leg 与 daily 共用同一条——雪球票息在自己的名义上按 tau 线性计提、只在
       腿间滚入，本就没有腿内复利。两端必须同口径，否则覆盖率（对冲/票息）会
       被系统性高估。n_slots=1 时退化为历史公式。

接口约定（为 notebook 工作流设计）
----------------------------------
    run_backtest(cfg) -> BacktestResult      函数进、结果对象出

    改参数用 dataclasses.replace，不改文件::

        from dataclasses import replace
        base = BacktestConfig(start_date="2018-01-02")
        r1 = run_backtest(base)
        r2 = run_backtest(replace(base, hedge_contract="当季"))
        compare([r1, r2])                     # -> 汇总对比表

    max_legs=1 退回"单只雪球"模式，便于逐腿诊断。
    数据、合约链、参数序列均按参数缓存，重复调用不会重读 Excel。

状态机与判定
------------
    敲入：自发行日次日起，任一交易日收盘 < 敲入线即置位，此后不可逆。
    敲出：仅在敲出观察日判定，收盘 >= 敲出线即终止。
    终止日 = 首个敲出日；否则为到期日；两者都未在数据内发生则为数据末端（截断）。

    情形由状态机直接判定。当整条路径在数据内可得时（即到期日不晚于数据末端），
    再用 payoff.evaluate_paths 对**同一条真实路径**独立复核一次，两者情形编号
    必须一致——回测状态机与定价所用判定规则由此保证同源。

已声明的缺省（见 README §0.2）
------------------------------
    无冲击成本、无保证金与现金管理收益、任一时点只有一只雪球
    （无多起点 ladder 平滑）。名义本金是否随净值复利由 notional_mode 控制。
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from .data_loader import load_market_data
    from .futures_chain import FuturesChain
    from .market_params import (build_market_params, REPO_WINDOW, REPO_CLIP,
                                VOL_WINDOW)
    from .snowball_contract import (make_snowball, make_snowball_weekly,
                                    next_issue_monday)
    from .pricer import solve_coupon, delta as price_delta, N_PATHS, SEED
    from .payoff import evaluate_paths, SCENARIO_LABELS
    from .pde_engine import N_SPACE
    from .curve_cache import solve_curves_cached, solve_coupon_cached
    from .hedger import (run_hedge, HedgeResult, IC_MULTIPLIER,
                         DEFAULT_NOTIONAL, DEFAULT_FEE_RATE)
except ImportError:
    from data_loader import load_market_data
    from futures_chain import FuturesChain
    from market_params import (build_market_params, REPO_WINDOW, REPO_CLIP,
                               VOL_WINDOW)
    from snowball_contract import (make_snowball, make_snowball_weekly,
                                   next_issue_monday)
    from pricer import solve_coupon, delta as price_delta, N_PATHS, SEED
    from payoff import evaluate_paths, SCENARIO_LABELS
    from pde_engine import N_SPACE
    from curve_cache import solve_curves_cached, solve_coupon_cached
    from hedger import (run_hedge, HedgeResult, IC_MULTIPLIER,
                        DEFAULT_NOTIONAL, DEFAULT_FEE_RATE)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """一次回测的全部参数。frozen => 可安全用 replace 生成变体。"""

    # -- 回测区间 --
    start_date: str = "2018-01-02"   # 首只雪球的发行日（monday_issue 时落到
                                     # 不早于该日的首个周一网格日）
    end_date: str = None             # 区间上界；None = 用到数据末端
    max_legs: int = None             # 每个槽位最多发行几只；None = 不限，
                                     # 1 = 单只模式（配 n_slots=1 用于逐腿诊断）
    allow_truncated_leg: bool = True  # 末腿允许在区间末端截断（情形0）；
                                      # False = 装不下完整一只就停止发行
    reissue_lag_days: int = 5        # 前一只终止后等待的交易日数，之后才发行下一只
                                     # （执行可行性约定，间隔期敞口为0）；0 = 终止当日接续
    stagger_reissue: bool = True     # 再发行错峰：发行日与其他槽位冲突时顺延一周
                                     # 补齐（保持观察日错开，见模块 docstring）；
                                     # False = 允许同日发行（撞车槽位条款同化，
                                     # ladder 退化为单雪球的旧行为）

    # -- 发行方式（槽位制 ladder；见模块 docstring） --
    n_slots: int = 4                 # 发行槽位数；1 = 单链条顺序滚动（旧行为）
    slot_stagger_weeks: int = 1      # 相邻槽位首发错开的周数（建仓期节奏）
    rampup_mode: str = "idle"        # 建仓期资金口径：
                                     # "idle"（默认）= 未启动槽位的资金闲置，
                                     #   组合在建仓期欠配（测策略性质，统计
                                     #   建议从 rampup_end 起算）
                                     # "full" = 已启动的 m 个槽位均分全部资金
                                     #   （权重 1 → 1/2 → ... → 1/K 逐级缩档，
                                     #   新槽位启动时存量腿基数同比例缩、腾出
                                     #   资金给新腿），第一天即满仓——模拟真实
                                     #   基金的建仓行为；代价是建仓期集中度高、
                                     #   结果对 start_date 更敏感
    weekly_terms: bool = True        # True = 周网格条款（周一发行、周五观察）；
                                     # False = 月度对日条款（历史行为）
    monday_issue: bool = True        # True = 发行落在周一网格（该周首个交易日）；
                                     # False = 任意交易日发行（历史行为）。
                                     # weekly_terms=True 时必须为 True
    # 完全退化到历史版本：n_slots=1, weekly_terms=False, monday_issue=False

    # -- 虚拟雪球条款（每只腿相同） --
    ki_pct: float = 0.70
    ko_pct: float = 1.03
    # 周网格条款参数（weekly_terms=True 时生效；含义见 make_snowball_weekly）
    tenor_weeks: int = 104           # 存续周数（≈24个月），末次观察=到期日
    first_ko_week: int = 12          # 首次敲出观察周（≈锁定2.8个月）
    ko_interval_weeks: int = 4       # 敲出观察间隔（周）
    ko_obs_weekday: object = 4       # 敲出观察日所在星期（0=周一 … 4=周五）；
                                     # 默认 4 与历史条款逐位一致。也可给长度为
                                     # n_slots 的元组，按槽位各自指定（如 (1,2,3,4)
                                     # = 四个槽位分别在周二至周五观察）
    # 月度对日条款参数（weekly_terms=False 时生效）
    tenor_months: int = 24
    lockup_months: int = 2           # =2 表示首次敲出观察在第3个月

    # -- 市场参数估计口径 --
    vol_window: int = VOL_WINDOW
    repo_window: int = REPO_WINDOW
    repo_clip: float = REPO_CLIP

    # -- 定价 --
    method: str = "pde"              # "pde" | "mc"
    n_space: int = 601               # 全周期回测的默认值；与1201差异约1bp
    n_paths: int = N_PATHS           # 仅 method="mc" 时生效
    seed: int = SEED
    param_mode: str = "daily"        # "daily": 每日用**前一交易日**的 repo/vol
                                     #   重算 delta（T-1 无前视口径：参数估计
                                     #   含当日收盘/结算信息，收盘前不可得，
                                     #   与执行层同钟；票息仍按发行日参数锁定）
                                     # "frozen": 每腿沿用其发行日参数（只变现价）
    keep_curves: bool = True         # method="pde" 时把逐腿日的整条价格曲线
                                     # 留在结果上（约 30KB/腿日，全周期约
                                     # 200MB），执行层（execution.py）零成本
                                     # 复用、免去二次求解；大批量参数扫描
                                     # 可关掉省内存
    curve_cache: bool = True         # method="pde" 时逐次求解结果按定价输入
                                     # 哈希落盘（data/pde_cache/），同键求解
                                     # 直接读盘、结果逐位一致（键的构成与
                                     # 失效规则见 curve_cache 模块 docstring）；
                                     # False = 每次现解

    # -- 对冲执行 --
    hedge_contract: str = "当月"      # 当月 / 次月 / 当季 / 下季
    roll_buffer_days: int = 5
    roll_window_days: int = 5        # 每次换月平摊到几个交易日完成（多天展期，
                                     # 见 futures_chain.build_schedule）；
                                     # 1 = 单日换月（历史行为）
    notional: float = DEFAULT_NOTIONAL
    notional_mode: str = "leg"       # 名义基数口径，见 hedger 模块 docstring
                                     # 与 README §0.1（三）：
                                     # "leg"（默认）= 每只雪球发行时按当时净值重置
                                     #   名义，腿内不变（已实现盈亏闲置到下一只
                                     #   建仓时才投入）；净值腿内算术、腿间连乘
                                     # "daily" = 名义逐日随净值缩放；净值逐日连乘
                                     # "fixed" = 名义恒为 notional；净值为算术和，
                                     #   各项损益归因可精确相加（分析口径）
    round_lots: bool = False         # False = 持仓为连续合约数（编指数口径，默认）；
                                     # True = 取整到整数手（执行层口径）
    delta_cap: float = None          # 敞口上限，None 为不截断
    fee_rate: float = DEFAULT_FEE_RATE  # 单边手续费率（成交名义额比例），默认万三；0 为免手续费

    @property
    def compounding(self) -> bool:
        """净值是否为连乘序列（leg 为腿内算术、腿间连乘，年化仍取几何）。"""
        return self.notional_mode != "fixed"

    def label(self) -> str:
        """用于对比表的简短标识。"""
        cap = "无cap" if self.delta_cap is None else f"cap{self.delta_cap:g}"
        tenor = (f"{self.tenor_weeks}W" if self.weekly_terms
                 else f"{self.tenor_months}M")
        slots = (f" L{self.n_slots}"
                 + ("F" if self.rampup_mode == "full" else "")
                 if self.n_slots > 1 else "")
        roll = f" R{self.roll_window_days}" if self.roll_window_days > 1 else ""
        wd = self.ko_obs_weekday
        obs = ("" if wd == 4 else " obs"
               + "".join("一二三四五"[int(w)] for w in (wd if isinstance(wd, (tuple, list)) else [wd])))
        return (f"{self.start_date} {self.ki_pct:.0%}/{self.ko_pct:.0%} "
                f"{tenor}{obs}{slots} {self.hedge_contract}{roll} {cap} "
                f"{self.notional_mode}")


# ---------------------------------------------------------------------------
# 缓存（notebook 中反复调用 run_backtest 时避免重复加载与重算）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _market_data():
    return load_market_data()


@lru_cache(maxsize=8)
def _chain(roll_buffer_days: int):
    return FuturesChain(_market_data(), roll_buffer_days=roll_buffer_days)


@lru_cache(maxsize=16)
def _params(repo_window: int, repo_clip: float, vol_window: int,
            roll_buffer_days: int):
    return build_market_params(_market_data(), _chain(roll_buffer_days),
                               repo_window=repo_window, repo_clip=repo_clip,
                               vol_window=vol_window)


@lru_cache(maxsize=4)
def _extended_calendar(extra_years: int = 4) -> pd.DatetimeIndex:
    """真实交易日历 + 数据末端之后的合成工作日。

    仅用于**构造条款**：末腿的存续期可能越过数据末端，而定价需要完整的敲出
    观察日程（少几次观察就是另一只雪球、票息也不同），故必须把日历补长。
    合成日按周一至周五生成，不含中国节假日——这只影响数据末端之后观察日的
    具体日期（至多平移一两天），对定价影响可忽略；且这些日期不参与任何取价。
    """
    days = _market_data().trading_days
    future = pd.bdate_range(days[-1] + pd.Timedelta(days=1),
                            days[-1] + pd.DateOffset(years=extra_years))
    return days.append(pd.DatetimeIndex(future))


def clear_cache() -> None:
    """更换 data/raw 下的数据后调用。"""
    for f in (_market_data, _chain, _params, _extended_calendar, _run_cached):
        f.cache_clear()


# 情形标签：payoff 层的五种，外加回测特有的"截断"
LEG_LABELS = {0: "0 截断(数据末端)", **SCENARIO_LABELS}


# ---------------------------------------------------------------------------
# 年化与覆盖率（口径随 notional_mode；analytics 复用）
# ---------------------------------------------------------------------------

def _annualize(total: float, years: float, geometric: bool) -> float:
    """总收益年化。geometric=False 线性（总收益/年数），True 几何。"""
    if years <= 0 or not np.isfinite(total):
        return np.nan
    if geometric:
        return ((1.0 + total) ** (1.0 / years) - 1.0) if 1.0 + total > 0 else np.nan
    return total / years


def _coverage_ann(res, years: float) -> float:
    """年化覆盖率 = 对冲端年化 / 票息年化。

    **复利口径下必须看这一项**：对冲端与票息端都是连乘序列，两者总收益之比
    会被凸性严重扭曲（如 12 只腿各累计约 20%，票息端连乘后总额是算术和的
    数倍），只有年化之比可读。算术口径下年化只是同除年数，该比值与总收益
    之比恒等，故两种口径下这一项都是可比的那个数。
    """
    geo = res.cfg.compounding
    a = _annualize(res.hedge_return, years, geo)
    b = _annualize(res.coupon_accrued, years, geo)
    return a / b if b else np.nan


# ---------------------------------------------------------------------------
# 单腿
# ---------------------------------------------------------------------------

def _run_leg(issue_date, cfg, md, params, kw, data_end):
    """跑一只雪球：定价 + 状态机 + 逐日 delta。

    Parameters
    ----------
    data_end : 可观测数据的末端。终止日不会晚于它；若合约在该日仍未了结，
               则记为截断（情形 0）。

    Returns
    -------
    (terms, delta_series, info, ki_state, journal, curves)
        delta_series 覆盖 [发行日, 终止日)，**不含终止日**——终止日由下一腿的
        发行日 delta 接管（末腿的终止日在外层补一行 0）。
        journal 为执行层（execution.py）所需的逐日状态清单：每日实际用于
        定价的参数（sigma/repo/r，daily 模式下为 T-1 口径）与敲入状态
        （前收盘 / 当日收盘两档），并含终止日一行（terminal=True）——终止日
        盘中合约仍存续（敲出/到期收盘才确认），执行层需要该日的参数为
        "仍存活"的合约算盘中 delta。
        curves 为 {日期: PDECurves}（method="pde" 且 keep_curves 时非空），
        与 journal 非终止行一一对应，供执行层零成本复用。
    """
    snap0 = params.snapshot(issue_date)
    # 用扩展日历构造条款：末腿的存续期可能越过数据末端，而定价需要完整日程
    if cfg.weekly_terms:
        terms = make_snowball_weekly(_extended_calendar(), issue_date,
                                     s0=snap0.s,
                                     ki_pct=cfg.ki_pct, ko_pct=cfg.ko_pct,
                                     tenor_weeks=cfg.tenor_weeks,
                                     first_ko_week=cfg.first_ko_week,
                                     ko_interval_weeks=cfg.ko_interval_weeks,
                                     ko_obs_weekday=cfg.ko_obs_weekday)
    else:
        terms = make_snowball(_extended_calendar(), issue_date, s0=snap0.s,
                              ki_pct=cfg.ki_pct, ko_pct=cfg.ko_pct,
                              tenor_months=cfg.tenor_months,
                              lockup_months=cfg.lockup_months)
    if cfg.method == "pde" and cfg.curve_cache:
        coupon0 = solve_coupon_cached(terms, snap0, **kw)
    else:
        coupon0 = solve_coupon(terms, snap0, method=cfg.method, **kw)
    terms = terms.with_coupon(coupon0)

    # ---- 真实路径上的状态机（只走可观测部分） ------------------------------
    grid = pd.DatetimeIndex(terms.sim_grid)
    obs = grid[grid <= data_end]
    if len(obs) < 2:
        raise ValueError(f"{issue_date} 发行的雪球可观测天数不足")
    path = md.index["close"].reindex(obs)
    if path.isna().any():
        raise ValueError(f"指数数据在 {issue_date} 发行的雪球存续期内存在缺失")

    ki_hit = path < terms.ki_level
    ki_hit.iloc[0] = False                       # 发行日不观察
    ki_state = ki_hit.cummax()

    ko_dates = pd.DatetimeIndex([d for d in terms.ko_obs_dates if d <= data_end])
    first_ko = None
    if len(ko_dates):
        ko_hit = path.reindex(ko_dates) >= terms.ko_level
        if ko_hit.any():
            first_ko = ko_dates[ko_hit.argmax()]

    # ---- 终止日与情形（由状态机直接判定） ----------------------------------
    truncated = False
    if first_ko is not None:
        end_date = first_ko
        # 敲出优先；仅当敲入严格发生在敲出日之前才记为情形5（与 payoff 层一致）
        ki_before = bool(ki_state.loc[:end_date].iloc[:-1].any())
        scenario = 5 if ki_before else 1
    elif terms.maturity_date <= data_end:
        end_date = terms.maturity_date
        s_t = float(path[end_date])
        if not bool(ki_state[end_date]):
            scenario = 2
        else:
            scenario = 3 if s_t >= terms.s0 else 4
    else:
        end_date = obs[-1]                       # 数据末端截断
        scenario = 0
        truncated = True

    # 整条路径在数据内可得时，用 payoff 层独立复核一次
    if grid[-1] <= data_end:
        res = evaluate_paths(terms, path.reindex(grid).to_numpy()[None, :])
        if int(res.scenario[0]) != scenario:
            raise ValueError(f"状态机情形 {scenario} 与 payoff 层判定 "
                             f"{int(res.scenario[0])} 不符")
        expected_end = terms.issue_date + pd.Timedelta(days=round(res.pay_tau[0] * 365))
        if abs((expected_end - end_date).days) > 1:
            raise ValueError(f"状态机终止日 {end_date.date()} 与 payoff 层判定 "
                             f"{expected_end.date()} 不符")

    # ---- 逐日 delta（不含终止日） -----------------------------------------
    # 参数口径：daily 模式取前一交易日（T-1 无前视，见 snapshot_prev）；
    # 现价仍为当日收盘——那是收盘时点可观测的量，参数才是"估计"。
    life = grid[(grid >= terms.issue_date) & (grid < end_date)]
    deltas = np.empty(len(life))
    sig = np.empty(len(life))
    rep = np.empty(len(life))
    rfr = np.empty(len(life))
    curves = {}
    if cfg.method == "pde":
        try:
            from .pde_engine import solve_curves
        except ImportError:
            from pde_engine import solve_curves
        _solve = solve_curves_cached if cfg.curve_cache else solve_curves
    for i, d in enumerate(life):
        snap = params.snapshot_prev(d) if cfg.param_mode == "daily" \
            else replace(snap0, date=d, s=md.spot(d))
        # 敞口比例 = 美元delta / 名义本金 = (dV/dS) x **当日现价**。
        # 注意归一化必须用当日现价而非期初价：hedger 随后按当日结算价把比例换回
        # 手数（N = delta x 名义 / (settle x 200)），两步的价格口径须一致，
        # 否则手数会多出一个 S0/settle_t 的因子（现价偏离期初价 20% 时即达 25%）。
        if cfg.method == "pde":
            # 解整条曲线、在收盘价处取 delta——与 pde_delta 同一代码路径、
            # 逐位同值；曲线留给执行层复用（keep_curves）
            cv = _solve(terms, snap, value_date=d, n_space=cfg.n_space)
            deltas[i] = cv.delta(snap.s, bool(ki_state[d])) * snap.s
            if cfg.keep_curves:
                curves[d] = cv
        else:
            deltas[i] = price_delta(terms, snap, value_date=d,
                                    knocked_in=bool(ki_state[d]),
                                    method=cfg.method, **kw) * snap.s
        sig[i], rep[i], rfr[i] = snap.sigma, snap.repo, snap.r

    # ---- 执行层状态清单（含终止日一行） ------------------------------------
    ki_prev = ki_state.shift(1, fill_value=False).astype(bool)
    snap_end = params.snapshot_prev(end_date) if cfg.param_mode == "daily" \
        else replace(snap0, date=end_date, s=md.spot(end_date))
    journal = pd.DataFrame({
        "date": np.append(life.to_numpy(), np.datetime64(end_date, "ns")),
        "ki_prev": np.append(ki_prev.reindex(life).to_numpy(),
                             bool(ki_prev[end_date])),
        "ki_close": np.append(ki_state.reindex(life).to_numpy(dtype=bool),
                              bool(ki_state[end_date])),
        "sigma": np.append(sig, snap_end.sigma),
        "repo": np.append(rep, snap_end.repo),
        "r": np.append(rfr, snap_end.r),
        "terminal": np.append(np.zeros(len(life), dtype=bool), True),
    })

    info = {
        "issue_date": terms.issue_date,
        "end_date": end_date,
        "s0": terms.s0,
        "coupon": terms.coupon,
        "scenario": scenario,
        "scenario_label": LEG_LABELS[scenario],
        "life_years": terms.tau(end_date),
        "maturity_date": terms.maturity_date,
        "truncated": truncated,
        "knocked_in_ever": bool(ki_state[end_date]),
        "ki_date": (ki_hit.idxmax() if ki_hit.any() else pd.NaT),
        "end_spot_ratio": float(path[end_date] / terms.s0),
    }
    return terms, pd.Series(deltas, index=life), info, ki_state, journal, curves


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    """一次回测的全部产出。"""

    cfg: BacktestConfig
    daily: pd.DataFrame             # 全周期逐日明细
    legs: pd.DataFrame              # 每只雪球一行（含 slot 列）
    hedge: HedgeResult
    terms_list: list = field(default_factory=list)   # 各腿的 SnowballTerms
                                                     # （与 legs 行序对应）
    slot_starts: tuple = ()         # 各槽位首只雪球的发行日
    leg_states: pd.DataFrame = None  # 执行层状态清单：每日每存续腿一行
                                     # [date, slot, leg, ki_prev, ki_close,
                                     #  sigma, repo, r, terminal]，leg 为
                                     # legs/terms_list 的行号；terminal=True
                                     # 行是腿的终止日（收盘确认敲出/到期，
                                     # 盘中仍按存活合约算 delta）。
                                     # sigma/repo/r 即当日定价实际所用参数
                                     # （daily 模式下为 T-1 口径）
    curves: dict = None              # (腿号, 日期) -> PDECurves 整条价格曲线
                                     # （method="pde" 且 keep_curves 时非空，
                                     # 不含终止日）。执行层 CurveBook 直接
                                     # 复用，免二次求解

    @property
    def nav(self) -> pd.Series:
        return self.daily["nav"]

    @property
    def rampup_end(self) -> pd.Timestamp:
        """建仓完成日 = 最后一个槽位首只雪球的发行日。此前组合未满仓
        （净值如实保留建仓期），业绩统计建议从该日起算（见
        analytics.nav_metrics 的 start 参数）。"""
        return max(self.slot_starts) if self.slot_starts \
            else self.daily.index[0]

    @property
    def hedge_return(self) -> float:
        """对冲端总收益（净值口径）。"""
        return float(self.daily["nav"].iloc[-1] - 1.0)

    @property
    def coupon_accrued(self) -> float:
        """全周期票息计提总额（跨腿累计）——对照基准。

        口径随 cfg.notional_mode：fixed 为算术和，leg / daily 同为滚动申购的连乘净值
        （减 1）。与对冲端同口径，故覆盖率两侧可比。
        """
        return float(self.daily["coupon_line"].iloc[-1])

    @property
    def pnl_base(self) -> pd.Series:
        """逐日损益的归一化基数，见 HedgeResult.pnl_base。"""
        return self.hedge.pnl_base

    def summary(self) -> pd.Series:
        d, lg = self.daily, self.legs
        h = self.hedge.summary()
        years = (d.index[-1] - d.index[0]).days / 365.0
        sc = lg["scenario"].value_counts()
        base = self.pnl_base
        spot_pnl = float((d["pnl_spot"] / base).sum())
        basis_pnl = float((d["pnl_basis"] / base).sum())
        fee_total = float((d["fee"] / base).sum())
        return pd.Series({
            "标识": self.cfg.label(),
            "起始日": d.index[0].date(),
            "结束日": d.index[-1].date(),
            "覆盖年数": round(years, 2),
            "名义口径": self.cfg.notional_mode,
            "槽位数": self.cfg.n_slots,
            "建仓完成日": self.rampup_end.date(),
            "雪球只数": len(lg),
            "敲出只数": int(sc.get(1, 0) + sc.get(5, 0)),
            "到期未敲入只数": int(sc.get(2, 0)),
            "敲入亏损只数": int(sc.get(4, 0)),
            "末端截断只数": int(sc.get(0, 0)),
            "平均票息": lg["coupon"].mean(),
            "平均存续年数": lg["life_years"].mean(),
            "对冲端总收益": self.hedge_return,
            "票息计提总额": self.coupon_accrued,
            "差额(对冲-票息)": self.hedge_return - self.coupon_accrued,
            "覆盖率": (self.hedge_return / self.coupon_accrued
                       if self.coupon_accrued else np.nan),
            # 复利口径下两端皆为连乘序列，总收益之比被凸性扭曲，须看年化之比
            # （算术口径下两者恒等）。定义见 analytics._coverage_ann。
            "覆盖率(年化)": _coverage_ann(self, years),
            # 年化口径随净值口径：fixed 线性（总收益/年数）、daily 几何。年数一律 /365
            "对冲端年化": _annualize(self.hedge_return, years,
                                     self.cfg.compounding),
            "方向性损益": spot_pnl,
            "基差损益": basis_pnl,
            "手续费": fee_total,
            # fixed 下三项之和恒等于对冲端总收益（交叉项为 0）；
            # daily 下三项之和为算术归因和，与几何总收益的差即复利交叉项。
            "复利交叉项": self.hedge_return - (spot_pnl + basis_pnl - fee_total),
            "年化波动": h["年化波动"],
            "最大回撤": h["最大回撤"],
            "平均delta": d["delta"].mean(),
            "最大delta": d["delta"].max(),
            "换月次数": h["换月次数"],
            "累计成交手数": h["累计成交手数"],
        })


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_backtest(cfg: BacktestConfig = None, **overrides) -> BacktestResult:
    """在给定区间上滚动发行虚拟雪球并回测（默认：周网格 + 4 槽位 ladder）。

    用法::

        r = run_backtest()                                    # 全部默认（ladder）
        r = run_backtest(start_date="2019-01-04")             # 关键字覆盖
        r = run_backtest(replace(cfg, hedge_contract="当季"))  # 显式配置
        r = run_backtest(n_slots=1)                           # 单链条顺序滚动
        r = run_backtest(n_slots=1, max_legs=1)               # 单只模式（诊断）
        r = run_backtest(n_slots=1, weekly_terms=False,
                         monday_issue=False)                  # 历史版本行为

    结果按 cfg 缓存（BacktestConfig 为 frozen dataclass，可哈希），
    notebook 中重复运行同一配置的 cell 不会重算。换数据后调 clear_cache()。
    """
    cfg = replace(cfg or BacktestConfig(), **overrides) if overrides \
        else (cfg or BacktestConfig())
    return _run_cached(cfg)


def _run_slot_legs(slot_start, cfg, md, params, kw, window_end, days,
                   reserved=None):
    """跑一个槽位：一条顺序滚动链的逐腿推进（全部 PDE 成本所在）。

    不含对冲记账——记账在组合层对全部槽位聚合后统一做（单一期货账户）。

    reserved 非 None 时为其他槽位已占用的发行日集合（stagger_reissue）：
    本槽位计算出的再发行日若在其中，顺延到下一个周一网格日再试。冲突
    解析按槽位次序确定（先跑的槽位优先占位），同日多腿重建被摊成逐周
    补齐——与建仓期错峰同一节奏。

    Returns
    -------
    dict 或 None（区间内一只都发不出来时）：
        delta      : 敞口比例序列，覆盖 [首腿发行日, 末腿终止日]，
                     腿间空仓段为 0，末行补 0（终止平仓）
        ki         : 逐日敲入状态（空仓段 False；末行 = 末腿终态）
        active     : 逐日是否有存续腿（空仓段与末行 False）
        coupon     : 该槽位票息对照线（客户滚动申购、本息滚入下一只）。
                     复利口径下为**财富值** W_t（起点 1，腿间乘 1 + C*tau），
                     算术口径下为累计计提 A_t（起点 0）；组合层等权平均后
                     （复利口径再减 1）得到对照线
        leg_id     : 逐日所属腿序号（空仓行归刚终止的腿）
        leg_rows   : 每腿一行的 dict 列表
        terms_list : 各腿 SnowballTerms
        issue_dates: 各腿发行日（= "leg" 口径的基数锚定日）
    """
    delta_parts, ki_parts, act_parts, coupon_parts, leg_ids = [], [], [], [], []
    leg_rows, terms_list, journals, curves_list = [], [], [], []
    issue = slot_start
    cum_coupon = 0.0        # 算术口径：已完成各腿的票息计提之和
    coupon_base = 1.0       # 复利口径：已完成各腿的连乘因子 prod(1 + C_j tau_j)
    pending_gap = None      # 上一腿终止后、下一腿发行前的空仓区段（交易日索引）

    # 槽位内票息财富线，口径与 notional_mode 一致——见模块 docstring
    def _coupon_frozen():
        """已完成部分的水平：空仓区段与末行取此值（计提暂停、定格）。"""
        return coupon_base if cfg.compounding else cum_coupon

    def _coupon_line(terms, idx):
        """存续期内逐日的票息计提线（本槽位财富口径）。"""
        if cfg.compounding:
            vals = [coupon_base * (1.0 + terms.coupon * terms.tau(d))
                    for d in idx]
        else:
            vals = [cum_coupon + terms.coupon * terms.tau(d) for d in idx]
        return pd.Series(vals, index=idx)

    while issue < window_end:
        if cfg.max_legs is not None and len(leg_rows) >= cfg.max_legs:
            break
        try:
            terms, dser, info, ki_state, jnl, cvs = _run_leg(issue, cfg, md,
                                                             params, kw,
                                                             window_end)
        except ValueError as e:
            if "可观测天数不足" in str(e):
                break                       # 区间末端只剩一两天，不再发行
            raise
        # 不允许截断时，装不下完整一只即停止发行（旧行为）
        if info["truncated"] and not cfg.allow_truncated_leg:
            break

        # 下一腿确定发行，上一腿终止后的空仓区段 [E_k, I_{k+1}) 才落账：
        # 敞口 0、无敲入状态、票息计提定格，行归属于刚终止的腿。
        # 若下一腿未能发行（区间末端等），该区段被丢弃，由"末腿补一行"收尾。
        if pending_gap is not None:
            delta_parts.append(pd.Series(0.0, index=pending_gap))
            ki_parts.append(pd.Series(False, index=pending_gap))
            act_parts.append(pd.Series(False, index=pending_gap))
            coupon_parts.append(pd.Series(_coupon_frozen(), index=pending_gap))
            leg_ids.append(pd.Series(len(leg_rows) - 1, index=pending_gap))
            pending_gap = None

        idx = dser.index
        delta_parts.append(dser)
        ki_parts.append(ki_state.reindex(idx))
        act_parts.append(pd.Series(True, index=idx))
        coupon_parts.append(_coupon_line(terms, idx))
        leg_ids.append(pd.Series(len(leg_rows), index=idx))
        leg_accrual = terms.coupon * info["life_years"]
        cum_coupon += leg_accrual
        coupon_base *= (1.0 + leg_accrual)

        leg_rows.append(info)
        terms_list.append(terms)
        journals.append(jnl)
        curves_list.append(cvs)
        if info["truncated"]:
            break                           # 末腿已顶到区间末端，无法再发行

        # 终止后等待 reissue_lag_days 个交易日（monday_issue 时再落到下一个
        # 周一网格日）才发行下一只。执行可行性约定：敲出与否收盘才能确认，
        # 同日建新仓不可交易。reissue_lag_days=0 且不落网格即终止当日接续。
        end_pos = days.searchsorted(info["end_date"])
        nxt_pos = end_pos + cfg.reissue_lag_days
        if nxt_pos >= len(days):
            break                           # 等待期越过数据末端，不再发行
        if cfg.monday_issue:
            try:
                nxt_pos = int(days.searchsorted(
                    next_issue_monday(days, days[nxt_pos])))
            except ValueError:
                break                       # 周一网格日越过数据末端
        if reserved is not None:
            # 再发行错峰：发行日已被其他槽位占用则顺延一周再试
            try:
                while nxt_pos < len(days) and days[nxt_pos] in reserved:
                    if cfg.monday_issue:
                        nxt_pos = int(days.searchsorted(next_issue_monday(
                            days, days[nxt_pos] + pd.Timedelta(days=1))))
                    else:
                        nxt_pos += 5
            except ValueError:
                nxt_pos = len(days)
        if nxt_pos >= len(days):
            break                           # 顺延越过数据末端，不再发行
        if nxt_pos > end_pos:
            pending_gap = days[end_pos:nxt_pos]
        issue = days[nxt_pos]               # 新腿期初 = 发行日收盘

    if not leg_rows:
        return None

    # 末腿终止日补一行：敞口归零、票息计提定格。
    # 截断腿在此处平仓属记账约定（数据到此为止），非合约事件。
    last_end = leg_rows[-1]["end_date"]
    delta_parts.append(pd.Series([0.0], index=[last_end]))
    ki_parts.append(pd.Series([leg_rows[-1]["knocked_in_ever"]], index=[last_end]))
    act_parts.append(pd.Series([False], index=[last_end]))
    coupon_parts.append(pd.Series([_coupon_frozen()], index=[last_end]))
    leg_ids.append(pd.Series(len(leg_rows) - 1, index=[last_end]))

    delta_series = pd.concat(delta_parts)
    delta_series.name = "delta"
    return {
        "delta": delta_series,
        "ki": pd.concat(ki_parts),
        "active": pd.concat(act_parts),
        "coupon": pd.concat(coupon_parts),
        "leg_id": pd.concat(leg_ids),
        "leg_rows": leg_rows,
        "terms_list": terms_list,
        "issue_dates": [r["issue_date"] for r in leg_rows],
        "journals": journals,
        "curves": curves_list,
    }


@lru_cache(maxsize=64)
def _run_cached(cfg: BacktestConfig) -> "BacktestResult":
    md = _market_data()
    chain = _chain(cfg.roll_buffer_days)
    params = _params(cfg.repo_window, cfg.repo_clip, cfg.vol_window,
                     cfg.roll_buffer_days)
    kw = {"n_space": cfg.n_space} if cfg.method == "pde" \
        else {"n_paths": cfg.n_paths, "seed": cfg.seed}

    if cfg.n_slots < 1:
        raise ValueError("n_slots 须 >= 1")
    if cfg.weekly_terms and not cfg.monday_issue:
        raise ValueError("weekly_terms=True 要求 monday_issue=True："
                         "周网格条款只能在周一网格日发行")
    if cfg.rampup_mode not in ("idle", "full"):
        raise ValueError(f'rampup_mode 须为 "idle" 或 "full"，'
                         f'收到 {cfg.rampup_mode!r}')

    days = md.trading_days
    window_end = pd.Timestamp(cfg.end_date) if cfg.end_date else days[-1]
    first = days[days.searchsorted(pd.Timestamp(cfg.start_date))]
    if cfg.monday_issue:
        first = next_issue_monday(days, first)
    # 定价用 T-1 参数（snapshot_prev）：首个发行日的前一交易日也须在预热之后
    if first <= params.repo.first_valid_index():
        raise ValueError(f"start_date 须晚于参数预热完成日 "
                         f"{params.repo.first_valid_index().date()}"
                         f"（定价用前一交易日参数）")

    # ---- 槽位起始日：依次错开 slot_stagger_weeks 周（建仓期错峰） ----------
    slot_starts = []
    for k in range(cfg.n_slots):
        target = first + pd.Timedelta(weeks=k * cfg.slot_stagger_weeks)
        if cfg.monday_issue:
            start_k = next_issue_monday(days, target)
        else:
            pos = days.searchsorted(target)
            if pos >= len(days):
                raise ValueError(f"槽位 {k} 的起始日越过数据末端")
            start_k = days[pos]
        slot_starts.append(start_k)
    if len(set(slot_starts)) != len(slot_starts):
        raise ValueError("槽位起始日重复：slot_stagger_weeks 太小或日历异常")

    # ---- 逐槽位推进腿序列（全部 PDE 成本在此） -----------------------------
    # stagger_reissue：占位表按槽位次序累积各槽位的发行日，后跑的槽位对
    # 冲突日顺延一周（初始起始日已错峰，一并入表）
    slots = []
    reserved = set(slot_starts) if cfg.stagger_reissue else None
    obs_wd = cfg.ko_obs_weekday
    if isinstance(obs_wd, (tuple, list)):
        if len(obs_wd) != cfg.n_slots:
            raise ValueError(f"ko_obs_weekday 元组长度须等于 n_slots（{cfg.n_slots}）")
    for k, start_k in enumerate(slot_starts):
        cfg_k = (replace(cfg, ko_obs_weekday=int(obs_wd[k]))
                 if isinstance(obs_wd, (tuple, list)) else cfg)
        out = _run_slot_legs(start_k, cfg_k, md, params, kw, window_end, days,
                             reserved=reserved)
        if out is None:
            raise ValueError(
                f"槽位 {k}（起始 {start_k.date()}）在区间内无法发行任何雪球，"
                f"区间太短或 start/end_date 有误")
        if reserved is not None:
            reserved.update(pd.Timestamp(d) for d in out["issue_dates"])
        slots.append(out)
    names = [f"slot{k}" for k in range(cfg.n_slots)]

    # ---- 对齐到全周期日历 --------------------------------------------------
    start_all = min(s["delta"].index[0] for s in slots)
    end_all = max(s["delta"].index[-1] for s in slots)
    idx = days[(days >= start_all) & (days <= end_all)]

    # ---- 对冲执行与记账（单一期货账户） ------------------------------------
    # "leg" 口径的基数锚定日 = 各腿发行日，由本层交给 hedger——该模块不感知
    # "腿"的概念（见 hedger.run_hedge 的 reset_dates）。
    # n_slots=1 走单套筒 Series 路径（与历史版本逐位一致）；多槽位走多套筒
    # DataFrame 路径：聚合总手数一次调仓、净额计费，腿基数锚定组合净值。
    if cfg.n_slots == 1:
        delta_input = slots[0]["delta"]
        resets = (slots[0]["issue_dates"]
                  if cfg.notional_mode == "leg" else None)
    else:
        delta_input = pd.DataFrame(
            {nm: s["delta"].reindex(idx).fillna(0.0)
             for nm, s in zip(names, slots)})
        resets = ({nm: s["issue_dates"] for nm, s in zip(names, slots)}
                  if cfg.notional_mode == "leg" else None)

    # 建仓期权重路径："full" 模式下已启动的 m 个槽位均分全部资金
    # （1 → 1/2 → ... → 1/K 缩档，第一天即满仓）；"idle"（默认）等权 1/K，
    # 未启动槽位的资金闲置在自己的份额里。
    weights = None
    if cfg.n_slots > 1 and cfg.rampup_mode == "full":
        firsts = [s["issue_dates"][0] for s in slots]
        started = np.column_stack([idx >= f for f in firsts])   # (天数, K)
        m_live = started.sum(axis=1)
        weights = pd.DataFrame(
            np.where(started, 1.0, 0.0) / m_live[:, None],
            index=idx, columns=names)

    hedge = run_hedge(delta_input,
                      chain.build_schedule(cfg.hedge_contract,
                                           roll_window_days=cfg.roll_window_days),
                      spot=md.index["close"], notional=cfg.notional,
                      multiplier=IC_MULTIPLIER, round_lots=cfg.round_lots,
                      delta_cap=cfg.delta_cap, close_at_end=True,
                      fee_rate=cfg.fee_rate, notional_mode=cfg.notional_mode,
                      reset_dates=resets, sleeve_weights=weights)

    # ---- 票息对照线（口径与对冲端一致） ------------------------------------
    if cfg.n_slots > 1 and cfg.rampup_mode == "full":
        # "full" 模式：组合票息财富递归。腿的计提基数锚定**组合票息财富**
        # （发行日权重 × 前日收盘财富），权重缩档时基数同比例重定标——与
        # 对冲端的基数机制完全镜像，财富曲线在缩档日连续（缩档只是把一部分
        # 计提基数从旧腿挪给新腿，不消灭已计提的财富）。
        w_np = weights.to_numpy()
        lids = [s["leg_id"].reindex(idx).to_numpy(dtype=float) for s in slots]
        acts = [s["active"].reindex(idx, fill_value=False).astype(bool)
                .to_numpy() for s in slots]
        wealth = 1.0                    # 复利口径的财富水平
        acc = 0.0                       # 算术口径的累计计提
        anchors = [0.0] * cfg.n_slots   # 各槽位当前腿的计提基数（财富单位）
        cur = [None] * cfg.n_slots
        prev_tau = [0.0] * cfg.n_slots
        vals = np.empty(len(idx))
        for i, d in enumerate(idx):
            wealth_prev = wealth        # 锚定用前日收盘财富
            day_accr = 0.0
            for k, s in enumerate(slots):
                if not acts[k][i]:
                    cur[k] = None
                    continue
                lid = int(lids[k][i])
                terms_k = s["terms_list"][lid]
                if cur[k] != lid:       # 新腿发行日
                    cur[k] = lid
                    anchors[k] = w_np[i, k] * (wealth_prev
                                               if cfg.compounding else 1.0)
                    prev_tau[k] = 0.0
                elif i > 0 and w_np[i, k] != w_np[i - 1, k]:
                    anchors[k] *= w_np[i, k] / w_np[i - 1, k]   # 权重缩档
                tau = terms_k.tau(d)
                day_accr += anchors[k] * terms_k.coupon * (tau - prev_tau[k])
                prev_tau[k] = tau
            wealth += day_accr
            acc += day_accr
            vals[i] = (wealth - 1.0) if cfg.compounding else acc
        coupon_line = pd.Series(vals, index=idx)
    else:
        # "idle"（默认）：各槽位自成一体的滚动申购财富，等权平均
        wgt = 1.0 / cfg.n_slots
        init = 1.0 if cfg.compounding else 0.0
        coupon_cols = [s["coupon"].reindex(idx).ffill().fillna(init)
                       for s in slots]
        coupon_line = sum(wgt * c for c in coupon_cols)
        if cfg.compounding:
            coupon_line = coupon_line - 1.0

    # ---- 腿表（slot 列；归因后按发行日排序） -------------------------------
    rows, terms_all = [], []
    for k, s in enumerate(slots):
        for r, t in zip(s["leg_rows"], s["terms_list"]):
            rr = dict(r)
            rr["slot"] = k
            rows.append(rr)
            terms_all.append(t)
    legs = pd.DataFrame(rows)

    # ---- 组装明细 ----------------------------------------------------------
    daily = hedge.daily.copy()
    daily["spot"] = md.index["close"].reindex(idx).to_numpy()
    daily["coupon_line"] = coupon_line.to_numpy()
    daily["repo"] = params.repo.reindex(idx).to_numpy()
    daily["vol"] = params.vol.reindex(idx).to_numpy()

    # 每腿的对冲端损益（净值口径，已扣手续费）。
    # 归属口径：pnl[t] 由 lots[t-1] 产生，故记给 **t-1 日**的活跃腿；
    # 若按当日归组，腿切换日的损益会被错记到新腿（该日仓位实为旧腿建立）。
    # 手续费则相反：fee[t] 由 t 日的调仓产生（目标敞口来自当日活跃腿），
    # 故按**当日**的腿归属——腿切换日的换仓费记给新腿。
    if cfg.n_slots == 1:
        # 单套筒：与历史版本一致的路径。
        # 归一化基数见 HedgeResult.pnl_base：fixed 下为名义本金，两项之和恒等
        # 于全周期净收益；daily 下为逐日缩放的名义基数，各腿得到的是该腿期间
        # 的**算术收益率之和**，跨腿相加等于算术归因和，与几何总收益之差即
        # summary() 中的"复利交叉项"（fixed 下该项恒为 0）。
        daily["leg"] = slots[0]["leg_id"].to_numpy()
        daily["leg_s0"] = legs["s0"].to_numpy()[daily["leg"].to_numpy()]
        daily["spot_ratio"] = daily["spot"] / daily["leg_s0"]
        daily["knocked_in"] = slots[0]["ki"].to_numpy()
        pos_leg = daily["leg"].shift(1).fillna(daily["leg"].iloc[0]).astype(int)
        # pnl[t] 的基数与归组同取 t-1 日：leg 口径下基数在腿切换日跳变，
        # reissue_lag_days=0（终止当日接续）时切换日的非零损益除以新腿基数
        # 会失真——与多套筒分支的 base_k.shift(1) 同口径。daily 的 pnl_base
        # 定义本身已是前收盘净值基数（见 HedgeResult.pnl_base），无需再错位；
        # fixed 恒定，shift 为恒等变换。fee[t] 属当日腿、用当日基数，不变。
        base = hedge.pnl_base
        base_pnl = base if hedge.notional_mode == "daily" \
            else base.shift(1).fillna(base.iloc[0])
        leg_pnl = (daily["pnl"] / base_pnl).groupby(pos_leg).sum()
        leg_fee = (daily["fee"] / base).groupby(daily["leg"]).sum()
        legs["leg_hedge_pnl"] = (leg_pnl.reindex(legs.index).fillna(0.0)
                                 - leg_fee.reindex(legs.index).fillna(0.0)
                                 ).to_numpy()
        legs["leg_fee"] = leg_fee.reindex(legs.index).fillna(0.0).to_numpy()
    else:
        # 多槽位：用套筒分列（pnl_<slot> / fee_<slot> / base_<slot>）逐槽位
        # 归因，每腿得到**自己冻结基数上的收益率**，与单槽位口径同义。
        ki_df = pd.DataFrame(
            {nm: s["ki"].reindex(idx, fill_value=False).astype(bool)
             for nm, s in zip(names, slots)})
        act_df = pd.DataFrame(
            {nm: s["active"].reindex(idx, fill_value=False).astype(bool)
             for nm, s in zip(names, slots)})
        daily["n_live"] = act_df.sum(axis=1).astype(int)
        daily["ki_count"] = ki_df.sum(axis=1).astype(int)
        daily["knocked_in"] = daily["ki_count"] > 0
        pnl_parts, fee_parts = [], []
        for nm, s in zip(names, slots):
            lid = s["leg_id"].reindex(idx)          # float，槽位生命期外 NaN
            daily[f"leg_{nm}"] = lid.to_numpy()
            base_k = daily[f"base_{nm}"]
            n_legs = len(s["leg_rows"])
            gp = ((daily[f"pnl_{nm}"] / base_k.shift(1))
                  .groupby(lid.shift(1)).sum())     # NaN 组自动丢弃
            gf = (daily[f"fee_{nm}"] / base_k).groupby(lid).sum()
            gp.index = gp.index.astype(int)
            gf.index = gf.index.astype(int)
            pnl_parts.append(gp.reindex(range(n_legs)).fillna(0.0).to_numpy())
            fee_parts.append(gf.reindex(range(n_legs)).fillna(0.0).to_numpy())
        legs["leg_fee"] = np.concatenate(fee_parts)
        legs["leg_hedge_pnl"] = np.concatenate(pnl_parts) - legs["leg_fee"]

    legs["leg_coupon_accrued"] = legs["coupon"] * legs["life_years"]
    legs["leg_gap"] = legs["leg_hedge_pnl"] - legs["leg_coupon_accrued"]

    # 按发行日排序（单槽位下本就有序，行序不变；terms_list 与行序保持对应）
    order = legs["issue_date"].to_numpy().argsort(kind="stable")
    legs = legs.iloc[order].reset_index(drop=True)
    terms_all = [terms_all[i] for i in order]

    # ---- 执行层状态清单（腿号 = 排序后 legs/terms_list 的行号） -------------
    inv = np.empty(len(order), dtype=int)
    inv[order] = np.arange(len(order))
    jparts, g = [], 0
    curve_map = {}          # (腿号, 日期) -> PDECurves，供执行层复用
    for k, s in enumerate(slots):
        for jnl, cvs in zip(s["journals"], s["curves"]):
            jj = jnl.copy()
            jj["slot"] = k
            leg_no = int(inv[g])
            jj["leg"] = leg_no
            g += 1
            jparts.append(jj)
            for d, cv in cvs.items():
                curve_map[(leg_no, pd.Timestamp(d))] = cv
    leg_states = (pd.concat(jparts, ignore_index=True)
                  .sort_values(["date", "slot"], kind="stable")
                  .reset_index(drop=True))

    return BacktestResult(cfg=cfg, daily=daily, legs=legs, hedge=hedge,
                          terms_list=terms_all,
                          slot_starts=tuple(s["issue_dates"][0] for s in slots),
                          leg_states=leg_states, curves=curve_map)


# ---------------------------------------------------------------------------
# 批量对比（notebook 主用法）
# ---------------------------------------------------------------------------

def compare(results) -> pd.DataFrame:
    """把多次回测的 summary 拼成一张对比表。"""
    return pd.DataFrame([r.summary() for r in results]).set_index("标识")


def sweep(base: BacktestConfig, param: str, values) -> list:
    """沿单个参数扫描，返回 BacktestResult 列表。

    用法::

        results = sweep(BacktestConfig(), "hedge_contract",
                        ["当月", "次月", "当季", "下季"])
        compare(results)
    """
    return [run_backtest(replace(base, **{param: v})) for v in values]


if __name__ == "__main__":
    import time

    print("=" * 78)
    print("槽位制 ladder 回测：2021-01-04 起至 2023-12-29，周网格条款，"
          "4 槽位按周错峰")
    print("=" * 78)
    tic = time.perf_counter()
    r = run_backtest(start_date="2021-01-04", end_date="2023-12-29")
    print(f"（耗时 {time.perf_counter() - tic:.0f}s）\n")
    print(r.summary().to_string())

    print("\n" + "=" * 78)
    print("分腿明细")
    print("=" * 78)
    cols = ["slot", "issue_date", "end_date", "s0", "coupon", "scenario_label",
            "life_years", "truncated", "leg_hedge_pnl", "leg_coupon_accrued",
            "leg_gap"]
    lg = r.legs[cols].copy()
    lg["issue_date"] = lg["issue_date"].dt.date
    lg["end_date"] = lg["end_date"].dt.date
    print(lg.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("自检")
    print("=" * 78)
    td = _market_data().trading_days

    # 1) 发行日全部落在周一网格；观察日错相分布
    from snowball_contract import is_week_start
    grid_ok = all(is_week_start(td, d) for d in r.legs["issue_date"])
    print(f"  发行日全为周一网格日: {grid_ok}")
    obs_wk = pd.Series([d.day_name()[:3] for t in r.terms_list
                        for d in t.ko_obs_dates])
    print(f"  全部敲出观察日星期分布: {obs_wk.value_counts().to_dict()}")

    # 2) 聚合恒等（fixed、免手续费下严格成立；有费时差额=净额抵消的节省）：
    #    策略级净值 vs 手工子账户平均
    r0 = run_backtest(start_date="2021-01-04", end_date="2023-12-29",
                      notional_mode="fixed", fee_rate=0.0)
    chain = _chain(r0.cfg.roll_buffer_days)
    sched = chain.build_schedule(r0.cfg.hedge_contract,
                                 roll_window_days=r0.cfg.roll_window_days)
    sub_navs = []
    for nm in (f"slot{k}" for k in range(r0.cfg.n_slots)):
        h = run_hedge(r0.daily[f"delta_{nm}"], sched, fee_rate=0.0,
                      close_at_end=True, notional_mode="fixed")
        sub_navs.append(h.daily["nav"])
    sub_avg = pd.concat(sub_navs, axis=1).mean(axis=1)
    diff = float((r0.nav - sub_avg).abs().max())
    print(f"  [fixed/免手续费] 策略级净值 vs 子账户均值 最大差 {diff:.2e}"
          f"（应为机器精度）")

    # 3) leg 口径：策略级与子账户平均之差 = 槽位间资本再平衡效应（报告值）
    #    ——策略级把新腿名义锚定组合净值，子账户各自复利，两者是不同的资金账
    sub_navs, w = [], 1.0 / r.cfg.n_slots
    for k in range(r.cfg.n_slots):
        nm = f"slot{k}"
        resets_k = list(r.legs.loc[r.legs["slot"] == k, "issue_date"])
        h = run_hedge(r.daily[f"delta_{nm}"], sched, fee_rate=r.cfg.fee_rate,
                      close_at_end=True, notional_mode="leg",
                      reset_dates=resets_k)
        sub_navs.append(h.daily["nav"])
    sub_avg = pd.concat(sub_navs, axis=1).mean(axis=1)
    print(f"  [leg] 策略级终值 {r.nav.iloc[-1]:.6f} vs 子账户均值 "
          f"{sub_avg.iloc[-1]:.6f}   差 {(r.nav.iloc[-1]-sub_avg.iloc[-1])*1e4:+.1f}bp"
          f"（再平衡效应 + 净额计费，非误差）")

    # 4) 索引连续性与票息线
    dup = int(r.daily.index.duplicated().sum())
    holes = int((~td[(td >= r.daily.index[0]) & (td <= r.daily.index[-1])]
                 .isin(r.daily.index)).sum())
    cl = r.daily["coupon_line"]
    print(f"  日线索引重复: {dup}，空缺: {holes}（均应为0）")
    print(f"  票息线单调: {bool((cl.diff().dropna() >= -1e-12).all())}")

    # 5) 槽内腿间隔 >= reissue_lag_days（周一网格使间隔可大于设定值）
    bad = 0
    for k in range(r.cfg.n_slots):
        lk = r.legs[r.legs["slot"] == k]
        for i in range(len(lk) - 1):
            gap_td = int(td.searchsorted(lk["issue_date"].iloc[i + 1])
                         - td.searchsorted(lk["end_date"].iloc[i]))
            bad += gap_td < r.cfg.reissue_lag_days
    print(f"  槽内腿间隔 < {r.cfg.reissue_lag_days} 个交易日的异常处: {bad}"
          f"（应为0）")

    # 6) 存续腿数与敲入统计
    print(f"  存续腿数: 均值 {r.daily['n_live'].mean():.2f} / "
          f"最大 {int(r.daily['n_live'].max())}（上限 {r.cfg.n_slots}）")
