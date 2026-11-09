"""对冲执行层：把"收盘算信号、结算价成交"的指数规则变成盘中可执行的交易，
并量化执行与指数之间的差异。

框架
----
一次回测（run_execution）同时跑多个**执行策略**，共享全部昂贵的计算：

    每日一个 DayContext（分钟行情 + 当日各存续腿的 PDE 曲线，曲线按
    (腿, 日) 缓存、跨策略共享）
        └─ 各策略独立决策：plan_day(ctx, 自己的持仓) -> 订单列表
              └─ 统一记账引擎逐笔成交、盯市、计费 -> 各自的净值

指数腿（日频回测 BacktestResult）是参照系：所有策略的净值与它比较，
逐日差异可加分解为三项（见 ExecutionResult）：

    数量项 te_qty   = 隔夜持仓差 x 当日结算价变动 —— "拿错仓位的钱"
    价格项 te_price = 成交量 x (结算价 − 成交价)   —— "跑赢/跑输结算价的钱"
    费差项 te_fee   = 指数腿手续费 − 执行腿手续费

约定（与设计草案一致）
--------------------
* 敲入状态收盘确认：盘中信号用**前收盘**已确认状态查曲线；
* 市场参数无前视：sigma/repo/r 直接读 leg_states——日频回测在 daily 模式
  下即用**前一交易日**参数定价（market_params.snapshot_prev），指数与
  执行层同参数、同曲线（bt.curves 复用），差异分解不含参数时点差；
  名义基数在 notional_mode="daily" 下取前一交易日（leg/fixed 本就
  T-1 可知）；
* 敲出收盘确认：敲出终止日盘中策略仍按"存活"合约报目标（PDE 曲线），
  收盘指数腿归零、执行腿次日才能平——该滞后是数量项的诚实来源（无前视）；
* 到期日（事先确定，无前视问题）：PDE 网格不含到期日当天，盘中用终端
  payoff 的解析 delta（已敲入且 S<S0 时为 S/S0，否则 0）；
* delta 截断属于**执行侧**：指数端默认不截断（报告纯模型 delta 的理论
  净值），敞口上限（保证金约束）是策略参数（内置策略默认 2.0），
  其代价被 TE 分解显式量化——"cap 造成的偏离"是可测量的执行成本，
  不是指数定义里的隐藏假设；
* 发行日无前视：新腿条款以发行日收盘定（s0 盘中未知），盘中信号用
  **查询时点价格**等比构造临时条款（障碍线随 s0 缩放、票息按同套 T-1
  参数现解）并现解曲线；次日起条款为已知量，回到真实条款的共享曲线。
  指数腿仍用真实条款（其定义即收盘口径）；
* 记账口径与 hedger 完全一致：日终按官方结算价盯市，
  NAV = 1 + (累计损益 − 累计手续费)/名义本金；日内成交贡献
  手数 x (当日结算价 − 成交价)；手续费单边按成交名义额计。

验收（__main__ 自检）
--------------------
* 曲线重建恒等：journal 重建的收盘 delta 与日频回测存储值逐位一致；
* 退化恒等：SettleReplica 策略（收盘信号、结算价成交）的净值与指数腿
  逐位一致——执行与指数的一切差异都来自明确建模的执行因素。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

try:
    from .backtest import BacktestResult, _chain
    from .curve_cache import solve_curves_cached, solve_coupon_cached
    from .hedger import IC_MULTIPLIER
    from .intraday_data import IntradayData, load_intraday
    from .market_params import MarketSnapshot
    from .pde_engine import pde_solve_coupon, solve_curves
except ImportError:
    from backtest import BacktestResult, _chain
    from curve_cache import solve_curves_cached, solve_coupon_cached
    from hedger import IC_MULTIPLIER
    from intraday_data import IntradayData, load_intraday
    from market_params import MarketSnapshot
    from pde_engine import pde_solve_coupon, solve_curves

# 时段常量（bar 以结束时刻标记）：TWAP 执行窗 (14:00, 14:55]，收尾窗 (14:55, 15:00]
SIGNAL_TIME = 135500
EXEC_WINDOW = (140100, 145500)
FINAL_SIGNAL_TIME = 145500
FINAL_WINDOW = (145600, 150000)
# 最后一小时整窗 (14:00, 15:00]，与结算价计算窗重合。15:00 bar 为完整一分钟
# 且通常是尾盘量最大的 bar（见 execution_backtest.ipynb 的完整性检查）
HOUR_WINDOW = (140100, 150000)


# ---------------------------------------------------------------------------
# PDE 曲线簿：按 (腿, 日) 缓存，跨策略共享
# ---------------------------------------------------------------------------

class CurveBook:
    """执行层的 delta 查询后端。

    solve_curves 返回估值日整条 S -> (价值, delta) 曲线（两层敲入状态），
    与日频回测的 pde_delta 同一代码路径、同一 n_space——收盘处的查询值
    与日频存储的 delta 逐位一致（见 validate_curves）。

    日频回测（keep_curves=True）已把逐腿日曲线留在 bt.curves 上，此处
    直接作为缓存种子——指数与执行共用同一批 PDE 求解，执行层只对缓存
    未命中的少数查询（腿的终止日等）现解。
    """

    def __init__(self, bt: BacktestResult):
        if bt.cfg.method != "pde":
            raise ValueError("执行层需要 method='pde' 的日频回测结果"
                             "（PDE 曲线可在任意盘中价位查询）")
        if bt.leg_states is None:
            raise ValueError("BacktestResult 缺 leg_states——请用加入状态"
                             "清单后的 backtest.py 重跑（clear_cache() 后）")
        self._terms = bt.terms_list
        self._n_space = bt.cfg.n_space
        # 与日频回测同开关：未命中的少数求解（终止日、发行日临时条款等）
        # 同样经盘存档（curve_cache），跨进程复用
        self._solve = (solve_curves_cached if bt.cfg.curve_cache
                       else solve_curves)
        self._solve_coupon = (solve_coupon_cached if bt.cfg.curve_cache
                              else pde_solve_coupon)
        self._cache: dict = {}
        if bt.curves:
            # 以 (腿, 日) 在 leg_states 中的实际定价参数补全缓存键——
            # 与 curves() 的查询键构造完全一致，保证逐位同源
            prm = {(int(row.leg), row.date):
                   (float(row.sigma), float(row.repo), float(row.r))
                   for row in bt.leg_states.itertuples()}
            for (leg, d), cv in bt.curves.items():
                sig, rep, rfr = prm[(leg, d)]
                self._cache[(leg, d, sig, rep, rfr)] = cv

    def terms(self, leg: int):
        return self._terms[int(leg)]

    def curves(self, leg: int, date, sigma: float, repo: float, r: float):
        # 键含参数：同一 (腿, 日) 可能被不同口径查询（执行层用 T-1 参数，
        # validate_curves 用当日参数），仅按 (腿, 日) 缓存会串值
        key = (int(leg), pd.Timestamp(date),
               float(sigma), float(repo), float(r))
        out = self._cache.get(key)
        if out is None:
            terms = self._terms[int(leg)]
            snap = MarketSnapshot(date=pd.Timestamp(date), s=terms.s0,
                                  r=r, repo=repo, sigma=sigma)
            out = self._solve(terms, snap, value_date=date,
                              n_space=self._n_space)
            self._cache[key] = out
        return out

    def issue_curves(self, leg: int, date, s0: float,
                     sigma: float, repo: float, r: float):
        """发行日的临时条款曲线（无前视）。

        真实条款的 s0 = 发行日收盘价，盘中不可知。以查询时点价格为 s0
        等比构造临时条款（ki/ko 线为 s0 的固定百分比，随之缩放；观察日程
        不变），票息按同一套 T-1 参数现解，再解曲线。每个发行日约多三次
        PDE 求解（票息两点 + 曲线一次），仅腿的首日走此路径。
        """
        key = ("issue", int(leg), pd.Timestamp(date), round(float(s0), 6),
               float(sigma), float(repo), float(r))
        out = self._cache.get(key)
        if out is None:
            prov = replace(self._terms[int(leg)], s0=float(s0), coupon=None)
            snap = MarketSnapshot(date=pd.Timestamp(date), s=float(s0),
                                  r=r, repo=repo, sigma=sigma)
            prov = prov.with_coupon(
                self._solve_coupon(prov, snap, n_space=self._n_space))
            out = self._solve(prov, snap, value_date=date,
                              n_space=self._n_space)
            self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# 每日上下文：策略能看到的全部信息
# ---------------------------------------------------------------------------

@dataclass
class DayContext:
    """一个交易日的决策上下文（跨策略共享，策略不得修改）。"""

    date: pd.Timestamp
    times: np.ndarray            # 240 个 bar 结束时刻
    legs: pd.DataFrame           # 当日存续腿：leg/slot/ki_prev/ki_close/
                                 # sigma/repo/r/terminal（即日频定价参数，
                                 # daily 模式下为 T-1 口径，无前视）
    bases: dict                  # slot -> 当日名义基数（元）
    code: str                    # 单合约视角的目标合约（手数换算价所属）
    code_out: str                # 展期窗口的两切片合约（非窗口日同 code）
    code_in: str
    w_in: float                  # 进入合约的目标仓位占比
    settle: dict                 # {code: 当日官方结算价}（含窗口两合约）
    index_target: dict           # 指数腿当日收盘目标 {code: lots}
    index_delta_cap: float       # 指数配置自带的 cap（默认 None = 不截断）
    is_last: bool                # 回测末日（指数腿强制平仓，执行腿跟随）
    patched: bool                # 当日指数分钟数据是否含补丁段
    fill_mode: str
    event_unwind: bool = False               # 观察日盘中 delta 口径（见 ExecutionConfig）
    ko_buffer: float = 0.0
    event_check_times: tuple = (135500,)
    _itd: IntradayData = field(repr=False, default=None)
    _book: CurveBook = field(repr=False, default=None)

    def unwound_legs(self, time: int) -> set:
        """t 时刻已因敲出观察触发而置零的腿号：判定时刻（默认仅 13:55）中
        不晚于 t 的任一时刻指数价过线即锁存（只用 t 之前的信息，与策略调用
        顺序和策略自身的目标计算时点无关）。"""
        if not self.event_unwind:
            return set()
        out = set()
        checks = [t for t in self.event_check_times if t <= int(time)]
        if not checks:
            return set()
        for row in self.legs.itertuples():
            terms = self._book.terms(row.leg)
            if self.date >= terms.maturity_date:         # 到期日：收盘结算，无条件置零
                out.add(int(row.leg))
                continue
            if self.date not in terms.ko_obs_dates:
                continue
            line = terms.ko_level * (1.0 + self.ko_buffer)
            if any(self.index_price(t) >= line for t in checks):
                out.add(int(row.leg))
        return out

    # ---- 行情查询 ----------------------------------------------------------

    def index_price(self, time: int) -> float:
        """t 时刻的指数价（bar close）。"""
        return self._itd.index_price(self.date, time)

    def fut_price(self, code: str, time: int) -> float:
        """t 时刻的合约价（bar close，观察用；成交价见 fill_price）。"""
        return self._itd.fut_price(self.date, code, time)

    def fill_price(self, code: str, time: int) -> float:
        """在 t 时刻 bar 内成交的价格假设（bar_vwap / bar_close）。"""
        return self._itd.bar_fill(self.date, code, time, self.fill_mode)

    def bars_between(self, t0: int, t1: int) -> np.ndarray:
        """[t0, t1] 闭区间内的全部 bar 标记。"""
        return self.times[(self.times >= t0) & (self.times <= t1)]

    def leg_terms(self, leg: int):
        """该腿的雪球条款（策略侧事件规则用：观察日程、障碍线、到期日）。"""
        return self._book.terms(leg)

    # ---- 目标手数 ----------------------------------------------------------

    def target_lots(self, time: int, delta_cap: float = None,
                    leg_scale: dict = None, ki_override: set = None) -> dict:
        """按 t 时刻的盘中价格重算指数规则下的目标持仓（无前视）。

        ki_override：盘中按"已敲入"曲线取 delta 的腿号集合（策略判定盘中价格已
        低于敲入线时传入；日频状态 ki_prev 仍为未敲入）。
        leg_scale：逐腿目标倍率 {腿号: 倍率}，未列出的腿为 1。执行侧的
        事件规则由策略判定、此处只做乘法：0 = 事件清仓（敲出已定局 /
        到期必平的腿提前置零），>1 = 事件加注（放大触发腿的敞口）。
        倍率乘在该腿敞口比例上、再经 cap 截断——敞口上限不因加注放松；
        判定所用信息须在 t 时刻可得，无前视由策略保证。

        与日频口径的对应关系（hedger 步骤 3）：
        逐腿 敞口比例 = dV/dS(盘中S) x 盘中S；金额 = Σ 比例 x 腿基数；
        总手数 = 金额 / (目标合约盘中价 x 乘数)；再按 w_in 切成
        （退出, 进入）两片。敲入状态用前收盘已确认值。

        delta_cap 是**执行侧**的敞口上限（保证金约束的体现，逐套筒
        ±截断，None = 不截断）——指数端默认不截断，截断属于执行层
        风控，其代价由 TE 分解量化。若指数配置本身带 cap
        （bt.cfg.delta_cap，敏感性扫描用），两者先后各截一次，
        效果等于取较严者。
        """
        s = self.index_price(time)
        unwound = self.unwound_legs(time)
        amount = 0.0
        for row in self.legs.itertuples():
            scale = (1.0 if leg_scale is None
                     else float(leg_scale.get(int(row.leg), 1.0)))
            if int(row.leg) in unwound and (leg_scale is None
                                            or int(row.leg) not in leg_scale):
                scale = 0.0                  # 观察日口径：过线腿置零（策略显式给倍率时以策略为准）
            if scale == 0.0:
                continue                     # 事件清仓：该腿目标记 0
            terms = self._book.terms(row.leg)
            if row.terminal and self.date >= terms.maturity_date:
                # 到期日：合约当日收盘结算（事先确定，非 KO 那种收盘才知道），
                # PDE 网格不含到期日当天，直接用终端 payoff 的解析 delta：
                #   已敲入（含盘中价低于敲入线的情形）且 S<S0 时
                #   payoff = 1 + (S/S0 − 1) => 敞口比例 = S/S0；其余为 0。
                # 敲入线附近保留真实的数字期权式跳变（最后观察日的 cliff）。
                knocked = bool(row.ki_prev) or s < terms.ki_level
                ratio = (s / terms.s0) if (knocked and s < terms.s0) else 0.0
            elif self.date == terms.issue_date:
                # 发行日：条款（s0 = 发行日收盘）盘中未定，用查询时点价格
                # 等比构造的临时条款查曲线（见 CurveBook.issue_curves）；
                # 次日起条款已知，回到真实条款的共享曲线
                cv = self._book.issue_curves(row.leg, self.date, s,
                                             row.sigma, row.repo, row.r)
                ratio = cv.delta(s, False) * s
            else:
                cv = self._book.curves(row.leg, self.date,
                                       row.sigma, row.repo, row.r)
                ki_state = bool(row.ki_prev) or (ki_override is not None
                                                 and int(row.leg) in ki_override)
                ratio = cv.delta(s, ki_state) * s
            ratio *= scale                           # 事件倍率（清仓/加注）
            if self.index_delta_cap is not None:     # 指数规则自带的 cap
                ratio = float(np.clip(ratio, -self.index_delta_cap,
                                      self.index_delta_cap))
            if delta_cap is not None:                # 执行侧 cap
                ratio = float(np.clip(ratio, -delta_cap, delta_cap))
            amount += ratio * self.bases[row.slot]
        if self.is_last:
            amount = 0.0
        f = self.fut_price(self.code, time)
        tot = amount / (f * IC_MULTIPLIER)
        return self._split(tot)

    def _split(self, tot: float) -> dict:
        if self.code_out == self.code_in:
            return {self.code_in: tot}
        return {self.code_in: tot * self.w_in,
                self.code_out: tot * (1.0 - self.w_in)}


# ---------------------------------------------------------------------------
# 策略接口与内置策略
# ---------------------------------------------------------------------------

class ExecutionStrategy(ABC):
    """执行策略基类。

    plan_day 返回 (orders, info)：
        orders : list[(time, code, lots)]，time=None 表示按结算价成交
                 （仅退化验收用），其余按该 bar 的 fill_price 成交；
        info   : dict，并入当日明细行（诊断列，如各时点的目标手数）。
    订单按列表顺序成交；框架负责记账，策略只负责"何时、何合约、多少手"。
    """

    name: str = "base"

    @abstractmethod
    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        """pos: 该策略昨日收盘后的持仓 {code: lots}（副本，可改）。"""


def _diff_orders(target: dict, pos: dict) -> dict:
    """目标持仓 − 当前持仓，含仅存在于一侧的合约。"""
    out = {}
    for c in set(target) | set(pos):
        q = target.get(c, 0.0) - pos.get(c, 0.0)
        if abs(q) > 1e-12:
            out[c] = q
    return out


class SettleReplica(ExecutionStrategy):
    """退化验收策略：收盘信号（直接取指数腿目标）、结算价成交。

    净值须与指数腿逐位一致——它是执行框架记账正确性的总闸，
    不是一个真实的执行方式。
    """

    name = "settle_replica"

    def plan_day(self, ctx: DayContext, pos: dict) -> tuple:
        orders = [(None, c, q)
                  for c, q in _diff_orders(ctx.index_target, pos).items()]
        return orders, {}


# 可交易的执行策略在 strategies/ 包中注册（一个策略一个文件，
# 见 strategies/__init__.py 的新增步骤）；本模块只保留框架：
# 策略基类、窗口常量、_diff_orders 与验收用的 SettleReplica。


# ---------------------------------------------------------------------------
# 记账引擎
# ---------------------------------------------------------------------------

class _Account:
    """一个策略的期货账户：持仓、逐日盯市、逐笔计费。"""

    def __init__(self):
        self.pos: dict = {}           # {code: lots}
        self.last_settle: dict = {}   # 持仓合约的上一结算价
        self.cum_pnl = 0.0
        self.cum_fee = 0.0

    def process_day(self, orders, settle: dict, fill, fee_rate: float,
                    log: list = None):
        """结算顺序与 hedger 一致：先用昨仓盯今日结算，再逐笔成交。

        log 非 None 时逐笔追加成交明细
        {time, code, lots, fill_px, settle_px}（time=None 表示按结算价成交）。

        Returns (pnl, fee, traded_lots, slip_cost)
            slip_cost = Σ q x (成交价 − 结算价) x 乘数，正 = 比结算价差。
        """
        pnl = 0.0
        for c, lots in self.pos.items():
            if c not in settle:
                raise KeyError(f"持仓合约 {c} 无当日结算价（已摘牌？"
                               f"策略未按目标及时移仓）")
            pnl += lots * (settle[c] - self.last_settle[c]) * IC_MULTIPLIER
        fee = 0.0
        traded = 0.0
        slip = 0.0
        for t, c, q in orders:
            if c not in settle:
                raise KeyError(f"订单合约 {c} 无当日结算价")
            p = settle[c] if t is None else fill(c, t)
            pnl += q * (settle[c] - p) * IC_MULTIPLIER
            slip += q * (p - settle[c]) * IC_MULTIPLIER
            fee += abs(q) * p * IC_MULTIPLIER * fee_rate
            traded += abs(q)
            if log is not None:
                log.append({"time": t, "code": c, "lots": q,
                            "fill_px": p, "settle_px": settle[c]})
            self.pos[c] = self.pos.get(c, 0.0) + q
        self.pos = {c: l for c, l in self.pos.items() if abs(l) > 1e-9}
        self.last_settle = {c: settle[c] for c in self.pos}
        self.cum_pnl += pnl
        self.cum_fee += fee
        return pnl, fee, traded, slip


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionConfig:
    fill_mode: str = "bar_vwap"     # bar_vwap | bar_close
    fee_rate: float = None          # None = 沿用日频回测的 fee_rate
    # 观察日盘中 delta 口径：曲线按"当日观察尚未发生"求解（j=0 不观察），
    # 敲出观察日盘中价格已过线时该腿本应接近零 delta。开关打开后，判定时刻
    # （event_check_times，默认仅 13:55）指数价 >= 敲出线 x (1 + ko_buffer) 即把
    # 该腿目标置零并锁存至收盘；到期日的腿无条件置零（收盘结算，不应持有过夜）。
    # 对全部策略统一生效，与策略自身的目标计算时点无关；策略在 t 时刻算目标时
    # 只用不晚于 t 的判定时刻。False = 历史口径（逐位复现既有档案）。
    event_unwind: bool = False
    ko_buffer: float = 0.0
    event_check_times: tuple = (135500,)


@dataclass(frozen=True)
class ExecutionResult:
    """多策略执行回测的全部产出。

    daily[策略名] 的列：
        nav / pnl / fee / traded_lots : 该策略自己的账户
        lots_eod                      : 日终总手数
        lots_index                    : 指数腿日终总手数
        slip_cost                     : 当日滑点成本（元，正 = 跑输结算价）
        te_qty / te_price / te_fee    : 相对指数腿的净值差三项分解（净值单位，
                                        逐日相加精确还原 nav − nav_index）
        以及策略 plan_day 返回的 info 诊断列

    orders[策略名] 为分钟订单流（逐笔成交明细）：
        date / time / code / lots / fill_px / settle_px
        time 为分钟 bar 结束标记（如 143000），Int64，<NA> = 按结算价成交
        （退化策略）；lots 带方向；settle_px 为该合约当日结算价
    """

    bt: BacktestResult
    cfg: ExecutionConfig
    daily: dict
    dates: pd.DatetimeIndex
    orders: dict = None

    @property
    def nav_index(self) -> pd.Series:
        return self.bt.daily.loc[self.dates, "nav"]

    @property
    def navs(self) -> pd.DataFrame:
        """指数腿 + 各策略的净值，一张表（notebook 主看板）。

        注意：start/end 子区间运行时各策略净值从 1 重新起算，与指数腿的
        **水平**不可比（差一个起点常数）；此时请看 summary() 或 te_* 分解，
        两者对子区间同样精确。全区间运行无此问题。
        """
        out = {"index": self.nav_index}
        for name, d in self.daily.items():
            out[name] = d["nav"]
        return pd.DataFrame(out)

    def summary(self) -> pd.DataFrame:
        """逐策略汇总：相对指数腿的年化差异及三项分解。"""
        years = len(self.dates) / 252.0
        rows = {}
        for name, d in self.daily.items():
            # 逐日净值差 = 三项分解之和（构造上精确，对子区间运行同样成立；
            # 直接用 nav − nav_index 在子区间会混入起点的基准偏移）
            te = d["te_qty"] + d["te_price"] + d["te_fee"]
            notional_traded = (d["traded_lots"] * d["settle_px"]).sum()
            slip_bp = (d["slip_cost"].sum() / (notional_traded * IC_MULTIPLIER)
                       * 1e4 if notional_traded else np.nan)
            rows[name] = {
                "累计净值差": te.sum(),
                "年化差(pp)": te.sum() / years * 100,
                "年化跟踪误差(pp)": te.std() * np.sqrt(252) * 100,
                "数量项年化(pp)": d["te_qty"].sum() / years * 100,
                "价格项年化(pp)": d["te_price"].sum() / years * 100,
                "费差项年化(pp)": d["te_fee"].sum() / years * 100,
                "加权滑点(bp)": slip_bp,
                "日均成交(手)": d["traded_lots"].mean(),
                "日均持仓缺口(手)": (d["lots_eod"]
                                     - d["lots_index"]).abs().mean(),
            }
        return pd.DataFrame(rows).T


def _index_target_lots(bt_row) -> dict:
    """指数腿当日收盘目标持仓（直接读 hedger 的切片明细）。"""
    if bt_row["code_out"] == bt_row["code_in"]:
        return {bt_row["code_in"]: float(bt_row["lots"])}
    return {bt_row["code_in"]: float(bt_row["lots_in"]),
            bt_row["code_out"]: float(bt_row["lots_out"])}


def run_execution(bt: BacktestResult, strategies: list,
                  cfg: ExecutionConfig = None,
                  start=None, end=None) -> ExecutionResult:
    """在日频回测结果之上跑多个执行策略（共享曲线与行情，一次遍历）。

    Parameters
    ----------
    bt : run_backtest 的结果（method='pde'，含 leg_states）
    strategies : ExecutionStrategy 实例列表，名字须唯一
    start / end : 只跑子区间（诊断用）。子区间起点的持仓以指数腿前一日
        持仓为种子（比较从同一起点出发），净值从 1 重新起算，
        te_* 分解仍逐日精确。
    """
    cfg = cfg or ExecutionConfig()
    fee_rate = bt.cfg.fee_rate if cfg.fee_rate is None else cfg.fee_rate
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError(f"策略名重复: {names}")

    itd = load_intraday()
    chain = _chain(bt.cfg.roll_buffer_days)
    daily_bt = bt.daily
    dates = daily_bt.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) == 0:
        raise ValueError("start/end 与回测区间无交集")
    missing = dates.difference(itd.index_px.index)
    if len(missing):
        raise ValueError(
            f"回测区间有 {len(missing)} 个交易日缺分钟数据"
            f"（首个: {missing[0].date()}）。执行回测仅支持 "
            f"{itd.index_px.index[0].date()} 之后的区间——指数分钟数据"
            f"自该日起；更早的日频回测无法做执行层复现")

    book = CurveBook(bt)
    # 无前视口径由日频回测统一保证：leg_states 存的 sigma/repo/r 就是当日
    # 定价实际所用参数——daily 模式下为**前一交易日**值（snapshot_prev，
    # 参数估计含当日收盘/结算信息、13:55 不可得），执行层直接读取即可，
    # 且与指数腿同参数（差异分解因此只含执行因素，不含参数时点差）。
    states = bt.leg_states
    # 建仓前的行（terminal 行之后无腿）不需要曲线；按日期分组一次取好
    states_by_day = {d: g for d, g in states.groupby("date")}
    sleeves = bt.hedge.sleeves          # None = 单套筒

    # 名义基数同理：leg / fixed 口径的 notional_t 本就是 T-1 可知量
    # （重置日取前收盘净值、窗口内冻结），直接用；daily 口径按当日结算
    # 盯市（hedger 步骤 2），盘中不可知 -> 取前一交易日的基数。
    base_cols = (["notional_t"] if sleeves is None
                 else [f"base_{nm}" for nm in sleeves])
    bases_df = daily_bt[base_cols]
    if bt.cfg.notional_mode == "daily":
        bases_df = bases_df.shift(1).fillna(bases_df.iloc[0])

    # 子区间起点的持仓种子：指数腿前一日的持仓
    seed_pos, seed_settle = {}, {}
    if dates[0] != daily_bt.index[0]:
        prev = daily_bt.index[daily_bt.index.get_loc(dates[0]) - 1]
        row = daily_bt.loc[prev]
        seed_pos = {c: l for c, l in _index_target_lots(row).items()
                    if abs(l) > 1e-9}
        seed_settle = {c: float(chain.settle(prev, c)) for c in seed_pos}

    accounts = {}
    for s in strategies:
        acc = _Account()
        acc.pos = dict(seed_pos)
        acc.last_settle = dict(seed_settle)
        accounts[s.name] = acc
    records = {s.name: [] for s in strategies}
    order_logs = {s.name: [] for s in strategies}

    empty_legs = states.iloc[0:0]
    for d in dates:
        row = daily_bt.loc[d]
        legs_today = states_by_day.get(d, empty_legs)

        # 各腿名义基数：多套筒读 base_<slot>，单套筒读 notional_t
        # （daily 口径已在上方整体错位到 T-1）
        brow = bases_df.loc[d]
        bases = {k: float(brow[c]) for k, c in enumerate(base_cols)}

        codes = {row["code"], row["code_out"], row["code_in"]}
        settle = {c: float(chain.settle(d, c)) for c in codes}
        # 昨日持仓合约今日可能不在 codes 里（窗口结束越界等）——补价
        for acc in accounts.values():
            for c in acc.pos:
                if c not in settle:
                    settle[c] = float(chain.settle(d, c))

        ctx = DayContext(
            date=d, times=itd.times, legs=legs_today, bases=bases,
            code=row["code"], code_out=row["code_out"],
            code_in=row["code_in"], w_in=float(row["w_in"]),
            settle=settle, index_target=_index_target_lots(row),
            index_delta_cap=bt.cfg.delta_cap,
            is_last=(d == daily_bt.index[-1]),
            patched=bool(itd.patched.get(d, False)),
            fill_mode=cfg.fill_mode, event_unwind=cfg.event_unwind,
            ko_buffer=cfg.ko_buffer, event_check_times=tuple(cfg.event_check_times),
            _itd=itd, _book=book)

        fill = lambda c, t: itd.bar_fill(d, c, t, cfg.fill_mode)
        idx_lots = sum(ctx.index_target.values())
        for s in strategies:
            acc = accounts[s.name]
            pos_prev = dict(acc.pos)
            orders, info = s.plan_day(ctx, dict(acc.pos))
            day_log = []
            pnl, fee, traded, slip = acc.process_day(orders, settle, fill,
                                                     fee_rate, log=day_log)
            for r in day_log:
                r["date"] = d
            order_logs[s.name].extend(day_log)
            # ---- 相对指数腿的当日净值差三项分解（元 -> 净值单位） ----------
            # 指数腿当日 pnl 全部来自隔夜（结算价成交对当日 pnl 贡献为 0）
            pnl_qty = (pnl + slip) - float(row["pnl"])   # 隔夜项差
            te = {
                "te_qty": pnl_qty / bt.cfg.notional,
                "te_price": -slip / bt.cfg.notional,
                "te_fee": (float(row["fee"]) - fee) / bt.cfg.notional,
            }
            records[s.name].append({
                "date": d,
                "nav": 1.0 + (acc.cum_pnl - acc.cum_fee) / bt.cfg.notional,
                "pnl": pnl, "fee": fee, "traded_lots": traded,
                "slip_cost": slip,
                "lots_eod": sum(acc.pos.values()),
                "lots_index": idx_lots,
                "settle_px": settle[row["code"]],
                "patched": ctx.patched,
                "is_roll": bool(row["is_roll"]),
                **te, **info,
            })

    out_daily = {}
    for name, recs in records.items():
        df = pd.DataFrame(recs).set_index("date")
        # 分解自洽性：三项之和须精确还原逐日净值差（相对指数腿）
        nav_idx = daily_bt.loc[dates, "nav"]
        base0 = 1.0 - float(nav_idx.iloc[0]) + (
            float(df["nav"].iloc[0]) - float(df["te_qty"].iloc[0]
                                             + df["te_price"].iloc[0]
                                             + df["te_fee"].iloc[0]))
        recon = base0 + (df["te_qty"] + df["te_price"] + df["te_fee"]).cumsum()
        resid = (df["nav"] - nav_idx - recon + 1.0).abs().max()
        # not(<=) 而非 (>)：NaN 与任何数比较均为 False，写 (>) 会让被 NaN
        # 污染的净值链静默通过自检
        if not (resid <= 1e-8):
            raise ValueError(f"策略 {name} 的 TE 分解不自洽，"
                             f"最大残差 {resid:.3g}")
        out_daily[name] = df

    out_orders = {}
    for name, rows in order_logs.items():
        odf = pd.DataFrame(rows, columns=["date", "time", "code", "lots",
                                          "fill_px", "settle_px"])
        odf["time"] = odf["time"].astype("Int64")   # <NA> = 按结算价成交
        out_orders[name] = odf

    return ExecutionResult(bt=bt, cfg=cfg, daily=out_daily, dates=dates,
                           orders=out_orders)


# ---------------------------------------------------------------------------
# 验收工具
# ---------------------------------------------------------------------------

def validate_curves(bt: BacktestResult, sample: int = 200,
                    seed: int = 0) -> float:
    """P1 恒等验收：journal 重建的收盘 delta vs 日频回测存储值。

    随机抽样非 terminal 的 (日, 腿) 状态行，用 CurveBook 在
    (收盘价, 收盘敲入状态) 处查询敞口比例，与日频存储的逐套筒 delta
    （截断前后一致处理）比对。返回最大绝对偏差；> 1e-9 报错。
    """
    book = CurveBook(bt)
    states = bt.leg_states
    rows = states[~states["terminal"]]
    rng = np.random.default_rng(seed)
    take = rows.iloc[rng.choice(len(rows), size=min(sample, len(rows)),
                                replace=False)]
    sleeves = bt.hedge.sleeves
    worst = 0.0
    for row in take.itertuples():
        d = row.date
        s_close = float(bt.daily.at[d, "spot"])
        cv = book.curves(row.leg, d, row.sigma, row.repo, row.r)
        ratio = cv.delta(s_close, bool(row.ki_close)) * s_close
        if bt.cfg.delta_cap is not None:
            ratio = float(np.clip(ratio, -bt.cfg.delta_cap,
                                  bt.cfg.delta_cap))
        col = "delta" if sleeves is None else f"delta_slot{row.slot}"
        stored = float(bt.daily.at[d, col])
        worst = max(worst, abs(ratio - stored))
    if not (worst <= 1e-9):        # not(<=)：NaN 也触发
        raise ValueError(f"曲线重建与日频 delta 不一致，最大偏差 {worst:.3g}"
                         f"（参数或代码路径有分歧）")
    return worst


def validate_replica(bt: BacktestResult, start=None, end=None) -> float:
    """P2 退化恒等验收：SettleReplica 净值 vs 指数腿净值，返回最大偏差。"""
    res = run_execution(bt, [SettleReplica()], start=start, end=end)
    nav_e = res.daily["settle_replica"]["nav"]
    nav_i = res.nav_index
    if start is None and end is None:
        worst = float((nav_e - nav_i).abs().max())
    else:  # 子区间从 1 起算，比对形状（逐日差分）
        worst = float((nav_e.diff() - nav_i.diff()).abs().max())
    if not (worst <= 1e-8):        # not(<=)：NaN 也触发
        raise ValueError(f"退化恒等失败：SettleReplica 与指数腿净值最大偏差 "
                         f"{worst:.3g}")
    return worst


if __name__ == "__main__":
    import time as _time

    try:
        from .backtest import run_backtest
        from .strategies import TwapTwoStage, MarketOnSignal
    except ImportError:
        from backtest import run_backtest
        from strategies import TwapTwoStage, MarketOnSignal

    print("跑小型日频回测（单槽位、2 腿，首腿为敲入后到期的情形 4）...")
    t0 = _time.time()
    bt = run_backtest(n_slots=1, weekly_terms=False, monday_issue=False,
                      max_legs=2, start_date="2018-01-02")
    print(f"  {_time.time() - t0:.0f}s, {len(bt.legs)} 腿, "
          f"{len(bt.daily)} 天, leg_states {len(bt.leg_states)} 行")
    print(f"  腿情形: {bt.legs['scenario_label'].tolist()}")

    print("P1 曲线重建恒等（抽样 120 个腿日）...")
    t0 = _time.time()
    w = validate_curves(bt, sample=120)
    print(f"  通过，最大偏差 {w:.3g}（{_time.time() - t0:.0f}s）")

    print("P2 退化恒等（SettleReplica vs 指数腿，全区间）...")
    t0 = _time.time()
    w = validate_replica(bt)
    print(f"  通过，最大偏差 {w:.3g}（{_time.time() - t0:.0f}s）")

    print("基准策略冒烟测试（跨首腿到期日的 40 天窗口）...")
    t0 = _time.time()
    end0 = bt.legs["end_date"].iloc[0]        # 首腿到期日（情形 4 回归覆盖）
    pos0 = bt.daily.index.get_loc(end0)
    d0 = bt.daily.index[max(0, pos0 - 35)]
    d1 = bt.daily.index[min(len(bt.daily) - 1, pos0 + 5)]
    res = run_execution(bt, [TwapTwoStage(), TwapTwoStage("twap_norecalc",
                                                          recalc=False),
                             MarketOnSignal()], start=d0, end=d1)
    print(f"  {_time.time() - t0:.0f}s（窗口 {d0.date()} ~ {d1.date()}，"
          f"含到期日 {end0.date()}）")
    print(res.summary().round(3).to_string())
