# -*- coding: utf-8 -*-
"""
snowball_contract.py —— 第2层：雪球条款定义

只描述"这是一只什么样的雪球"，不含任何定价逻辑。
MC 引擎（payoff.py）和 PDE 引擎（pde_engine.py）共享这唯一的条款来源，
两个引擎对同一份 SnowballTerms 给出一致价格，就是互验的依据。
两个引擎只消费 ko_obs_dates / ki_obs_dates / maturity_date / tau()——观察日
按什么日历生成对定价层透明，因此本模块可提供多种观察日历的构造器。

条款约定（经典雪球，对应 payoff.py 的五种情形）：
    - strike = 期初价 S0（100%），亏损腿 payoff = min(S_T/S0 − 1, 0)
    - 敲出观察日历，二选一：
        * make_snowball()        月度对日（发行日在每月的"对日"，遇非交易日
                                 顺延至下一交易日），市场标准条款
        * make_snowball_weekly() 周网格（周一发行、每 ko_interval_weeks 周的
                                 周五观察），虚拟雪球的自定义日历，见构造器
                                 docstring
      两者下锁定期内均不观察；到期日均是最后一个敲出观察日
    - 敲入观察：每日（发行日次日起至到期日的每个交易日，收盘价判定）
    - 判定惯例（payoff层实现，此处备忘）：敲出 = 收盘 >= 敲出线；
      敲入 = 收盘 < 敲入线
    - 凭证口径：发行日现值 = 1（本金随凭证交付，票息含本金利息）

时间口径：票息计提与贴现均用 自然日/365（repo 漂移为交易日时钟，
换算见 market_params.estimate_repo）。
"""

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class SnowballTerms:
    """一只虚拟雪球的完整条款。用 make_snowball() 或 make_snowball_weekly()
    构造，不要手工填。

    Attributes
    ----------
    issue_date : 发行日（定价基准日，S0的采样日）
    s0         : 期初价（发行日指数收盘）
    ki_pct / ko_pct : 敲入/敲出线（期初价的百分比，如 0.75 / 1.03）
    maturity_date : 到期日（最后一个敲出观察日）
    ko_obs_dates  : 敲出观察日（锁定期后开始，最后一个=到期日）
    ki_obs_dates  : 敲入观察日（发行日次日起的每个交易日，含到期日）
    tenor_months / lockup_months : 存续期 / 锁定期（月）。仅月度对日条款
             （make_snowball）填写，周网格条款下为 None。
    tenor_weeks / first_ko_week / ko_interval_weeks : 周网格条款
             （make_snowball_weekly）的存续周数 / 首次观察周 / 观察间隔周。
             仅周网格条款填写，月度对日条款下为 None。
    coupon : 年化票息。发行时未知（None），由 pricer 反解后经
             with_coupon() 填入。

    定价引擎只消费日期与 tau，两套日历字段仅供描述与回测层记录。
    """

    issue_date: pd.Timestamp
    s0: float
    ki_pct: float
    ko_pct: float
    maturity_date: pd.Timestamp
    ko_obs_dates: tuple
    ki_obs_dates: tuple
    tenor_months: int = None
    lockup_months: int = None
    tenor_weeks: int = None
    first_ko_week: int = None
    ko_interval_weeks: int = None
    coupon: float = None

    # -- 价格水平（绝对点位） ------------------------------------------------

    @property
    def ki_level(self) -> float:
        """敲入线（指数点位）。"""
        return self.s0 * self.ki_pct

    @property
    def ko_level(self) -> float:
        """敲出线（指数点位）。"""
        return self.s0 * self.ko_pct

    # -- 时间换算（两个引擎都要用） ------------------------------------------

    def tau(self, date) -> float:
        """发行日到某日的年化时长（自然日/365），用于票息计提与贴现。"""
        return (pd.Timestamp(date) - self.issue_date).days / 365.0

    @property
    def maturity_tau(self) -> float:
        """总存续年限 T。"""
        return self.tau(self.maturity_date)

    @property
    def ko_obs_taus(self) -> tuple:
        """各敲出观察日的年化时点（敲出票息 = coupon × 对应tau）。"""
        return tuple(self.tau(d) for d in self.ko_obs_dates)

    @property
    def sim_grid(self) -> tuple:
        """模拟/判定时间网格：发行日 + 全部敲入观察日（即存续期每个交易日）。
        MC 路径的列与此一一对应；列0 = 发行日（S0），列1起 = 敲入观察日。"""
        return self.grid_from(None)

    def grid_from(self, value_date=None) -> tuple:
        """从估值日出发的剩余网格：估值日 + 之后的全部敲入观察日。

        value_date=None 或发行日 => 完整网格 sim_grid。
        存续期中估值（每日重算delta）时，MC 只需模拟剩余寿命的路径，
        列0 = 估值日（当前现价），列1起 = 未来的敲入观察日。
        """
        v = self.issue_date if value_date is None else pd.Timestamp(value_date)
        if not self.issue_date <= v < self.maturity_date:
            raise ValueError(f"估值日 {v.date()} 须在 [发行日, 到期日) 内")
        return (v,) + tuple(d for d in self.ki_obs_dates if d > v)

    # -- 工具 ---------------------------------------------------------------

    def with_coupon(self, coupon: float) -> "SnowballTerms":
        """填入反解出的票息，返回新对象（条款本身不可变）。"""
        return replace(self, coupon=coupon)

    def describe(self) -> str:
        """人类可读的条款摘要。"""
        c = f"{self.coupon:.2%}" if self.coupon is not None else "待反解"
        if self.tenor_weeks is not None:
            grid = (f"周网格（锁定至第{self.first_ko_week}周，"
                    f"每{self.ko_interval_weeks}周观察）")
        else:
            grid = f"月度对日（锁定{self.lockup_months}个月）"
        lines = [
            f"雪球条款  发行 {self.issue_date.date()}  到期 {self.maturity_date.date()}"
            f"  (T={self.maturity_tau:.3f}年)",
            f"  期初价 {self.s0:.2f} | 敲入 {self.ki_pct:.0%} ({self.ki_level:.2f})"
            f" | 敲出 {self.ko_pct:.0%} ({self.ko_level:.2f}) | 票息 {c}",
            f"  敲出观察 {len(self.ko_obs_dates)} 次"
            f"（{grid}，首次 {self.ko_obs_dates[0].date()}）"
            f" | 敲入观察 {len(self.ki_obs_dates)} 天（每日）",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------

def _next_trading_day(calendar: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    """calendar 中不早于 target 的第一个交易日（对日遇假顺延的实现）。"""
    pos = calendar.searchsorted(target)
    if pos >= len(calendar):
        raise ValueError(f"目标日 {target.date()} 超出交易日历末端 "
                         f"({calendar[-1].date()})，该雪球无法在数据内走完")
    return calendar[pos]


def make_snowball(calendar: pd.DatetimeIndex, issue_date, s0: float,
                  ki_pct: float = 0.75, ko_pct: float = 1.03,
                  tenor_months: int = 12, lockup_months: int = 3) -> SnowballTerms:
    """按"月度对日、遇假顺延"惯例生成全部观察日，构造条款。

    Parameters
    ----------
    calendar : 交易日历（md.trading_days）
    issue_date : 发行日（必须是交易日）
    s0 : 期初价（调用方传入 md.spot(issue_date)；不直接依赖 MarketData，
         保持本层可独立测试）
    其余为条款参数。
    """
    issue = pd.Timestamp(issue_date)
    if issue not in calendar:
        raise ValueError(f"发行日 {issue.date()} 不是交易日")
    if not 0 < ki_pct < ko_pct:
        raise ValueError(f"需满足 0 < 敲入线({ki_pct}) < 敲出线({ko_pct})")
    if not 0 <= lockup_months < tenor_months:
        raise ValueError("需满足 0 <= 锁定期 < 存续期")

    # 敲出观察日：第 lockup+1 ... tenor 个月的对日（顺延），最后一个即到期日
    ko_dates = []
    for k in range(lockup_months + 1, tenor_months + 1):
        target = issue + pd.DateOffset(months=k)   # 自动处理月末（1/31+1月=2/28）
        ko_dates.append(_next_trading_day(calendar, target))
    maturity = ko_dates[-1]

    # 敲入观察日：发行日次日起至到期日（含）的全部交易日
    i0, i1 = calendar.searchsorted(issue), calendar.searchsorted(maturity)
    ki_dates = tuple(calendar[i0 + 1: i1 + 1])

    return SnowballTerms(
        issue_date=issue, s0=float(s0),
        ki_pct=float(ki_pct), ko_pct=float(ko_pct),
        tenor_months=int(tenor_months), lockup_months=int(lockup_months),
        maturity_date=maturity,
        ko_obs_dates=tuple(ko_dates), ki_obs_dates=ki_dates,
    )


# ---------------------------------------------------------------------------
# 周网格条款（虚拟雪球的自定义日历）
# ---------------------------------------------------------------------------

def _iso_week(ts: pd.Timestamp) -> tuple:
    """(ISO年, ISO周) —— 判断两个日期是否同一周的键。"""
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def is_week_start(calendar: pd.DatetimeIndex, date) -> bool:
    """date 是否为其所在周的首个交易日（周一网格：周一逢假顺延后的等价日）。"""
    ts = pd.Timestamp(date)
    pos = calendar.searchsorted(ts)
    if pos >= len(calendar) or calendar[pos] != ts:
        raise ValueError(f"{ts.date()} 不是交易日")
    return pos == 0 or _iso_week(calendar[pos - 1]) != _iso_week(ts)


def next_issue_monday(calendar: pd.DatetimeIndex, target) -> pd.Timestamp:
    """calendar 中不早于 target 的第一个"周一网格日"（该周首个交易日）。

    周一网格是周网格条款的发行日约定：只在每周的首个交易日发行（周一，
    逢假顺延后即该周第一个交易日）。回测层的建仓错峰与再发行都用本函数
    把任意日期落到网格上。
    """
    ts = pd.Timestamp(target)
    pos = calendar.searchsorted(ts)
    while pos < len(calendar):
        if pos == 0 or _iso_week(calendar[pos - 1]) != _iso_week(calendar[pos]):
            return calendar[pos]
        pos += 1
    raise ValueError(f"{ts.date()} 之后的日历中已无周一网格日")


def make_snowball_weekly(calendar: pd.DatetimeIndex, issue_date, s0: float,
                         ki_pct: float = 0.70, ko_pct: float = 1.03,
                         tenor_weeks: int = 104, first_ko_week: int = 12,
                         ko_interval_weeks: int = 4,
                         ko_obs_weekday: int = 4) -> SnowballTerms:
    """按"周一发行、周五观察"的周网格生成观察日，构造条款。

    ko_obs_weekday：观察日所在星期（0=周一 … 4=周五，默认 4 与历史行为
    逐位一致）。取 0–3 时目标日逢假**顺延**至该周内下一个交易日（周内不
    跨周）；该周其后已无交易日时回退到目标日前最后一个交易日。

    观察日历
    --------
    发行日须为周一网格日（其所在周的首个交易日）。敲出观察日取发行周之后
    第 first_ko_week, first_ko_week + ko_interval_weeks, ..., tenor_weeks 周的
    **周五**；周五逢假**提前**至该周最后一个交易日（周内收缩、不跨周），
    以保证"每周至多一次观察"的结构。整周无交易日的长假周（如国庆调休）
    是唯一例外：此时回退到上一个交易日（跨周兜底），实际数据中极少发生。

    默认参数 (104, 12, 4)：24 个月存续、首观第 12 周（≈2.8 个月，对应月度
    条款"锁定 2 个月、第 3 个月首观"）、每 4 周观察一次，共 24 次观察，
    末次观察即到期日。

    约束：(tenor_weeks − first_ko_week) 须被 ko_interval_weeks 整除，
    否则末次观察落不到 tenor_weeks 周，"末次观察=到期日"即被破坏。
    """
    issue = pd.Timestamp(issue_date)
    if not 0 < ki_pct < ko_pct:
        raise ValueError(f"需满足 0 < 敲入线({ki_pct}) < 敲出线({ko_pct})")
    if not 1 <= first_ko_week <= tenor_weeks:
        raise ValueError("需满足 1 <= 首次观察周 <= 存续周数")
    if ko_interval_weeks < 1:
        raise ValueError("观察间隔须 >= 1 周")
    if (tenor_weeks - first_ko_week) % ko_interval_weeks != 0:
        raise ValueError(
            f"(tenor_weeks({tenor_weeks}) − first_ko_week({first_ko_week})) "
            f"须被 ko_interval_weeks({ko_interval_weeks}) 整除，"
            f"否则末次观察不落在到期周")
    if not is_week_start(calendar, issue):   # 内部同时校验 issue 是交易日
        raise ValueError(f"发行日 {issue.date()} 不是周一网格日"
                         f"（须为其所在周的首个交易日）")

    if not 0 <= int(ko_obs_weekday) <= 4:
        raise ValueError("ko_obs_weekday 须为 0–4（周一至周五）")
    # 发行周的周一（自然日锚点）；第 k 周的目标日 = 锚点 + k 周 + 星期偏移
    monday0 = issue - pd.Timedelta(days=int(issue.weekday()))
    ko_dates = []
    for k in range(first_ko_week, tenor_weeks + 1, ko_interval_weeks):
        target = monday0 + pd.Timedelta(weeks=k, days=int(ko_obs_weekday))
        if target > calendar[-1]:
            raise ValueError(f"目标观察日 {target.date()} 超出交易日历末端 "
                             f"({calendar[-1].date()})，该雪球无法在数据内走完")
        pos = calendar.searchsorted(target, side="right") - 1   # 最后一个 <= 目标日
        if ko_obs_weekday < 4:
            p2 = calendar.searchsorted(target)                  # 第一个 >= 目标日
            if p2 < len(calendar) and _iso_week(calendar[p2]) == _iso_week(target):
                pos = p2                                        # 周内顺延
        ko_dates.append(calendar[pos])
    if len(set(ko_dates)) != len(ko_dates) or ko_dates[0] <= issue:
        raise ValueError("周网格观察日退化（重复或早于发行日），请检查日历")
    maturity = ko_dates[-1]

    # 敲入观察日：发行日次日起至到期日（含）的全部交易日（与月度条款一致）
    i0, i1 = calendar.searchsorted(issue), calendar.searchsorted(maturity)
    ki_dates = tuple(calendar[i0 + 1: i1 + 1])

    return SnowballTerms(
        issue_date=issue, s0=float(s0),
        ki_pct=float(ki_pct), ko_pct=float(ko_pct),
        tenor_weeks=int(tenor_weeks), first_ko_week=int(first_ko_week),
        ko_interval_weeks=int(ko_interval_weeks),
        maturity_date=maturity,
        ko_obs_dates=tuple(ko_dates), ki_obs_dates=ki_dates,
    )


if __name__ == "__main__":
    import numpy as np

    try:
        from .data_loader import load_market_data
    except ImportError:
        from data_loader import load_market_data

    md = load_market_data()

    terms = make_snowball(md.trading_days, "2021-06-15", s0=md.spot("2021-06-15"))
    print(terms.describe())
    print("\n敲出观察日与对应票息时点:")
    for d, t in zip(terms.ko_obs_dates, terms.ko_obs_taus):
        flag = "  <- 对日顺延" if d.day != terms.issue_date.day else ""
        print(f"  {d.date()} ({d.day_name()[:3]})  tau={t:.4f}{flag}")

    # 边界1：月末发行（对日不存在的月份自动收缩）
    t2 = make_snowball(md.trading_days, "2021-08-31", s0=md.spot("2021-08-31"))
    print(f"\n月末发行 2021-08-31 的观察日: {[str(d.date()) for d in t2.ko_obs_dates[:3]]} ...")

    # 边界2：到期超出日历必须报错
    try:
        make_snowball(md.trading_days, "2026-06-15", s0=1.0)
    except ValueError as e:
        print(f"\n超出日历拦截成功: {e}")

    # 填票息
    print("\n" + terms.with_coupon(0.15).describe())

    # ---- 周网格条款 --------------------------------------------------------
    monday = next_issue_monday(md.trading_days, "2021-06-15")
    tw = make_snowball_weekly(md.trading_days, monday, s0=md.spot(monday))
    print("\n" + tw.describe())
    dows = [d.day_name()[:3] for d in tw.ko_obs_dates]
    shrunk = [d for d in tw.ko_obs_dates if d.weekday() != 4]
    weeks = [_iso_week(d) for d in tw.ko_obs_dates]
    print(f"  观察日星期分布: {dict(pd.Series(dows).value_counts())}"
          f"（非周五 = 周内收缩，共 {len(shrunk)} 次: "
          f"{[str(d.date()) for d in shrunk]}）")
    print(f"  观察次数 {len(tw.ko_obs_dates)}（应为 "
          f"{(104 - 12) // 4 + 1}）| 各次观察不同周: "
          f"{len(set(weeks)) == len(weeks)} | 末次=到期日: "
          f"{tw.ko_obs_dates[-1] == tw.maturity_date}")
    gaps = np.diff([md.trading_days.searchsorted(d) for d in tw.ko_obs_dates])
    print(f"  相邻观察间隔（交易日）: min {gaps.min()} / max {gaps.max()}"
          f"（应在 4 周 ≈ 18~20 附近）")

    # 周一网格：非网格日发行必须报错（2021-06-16 是周三，且当周 15 日已有交易）
    try:
        make_snowball_weekly(md.trading_days, "2021-06-16", s0=1.0)
    except ValueError as e:
        print(f"  非周一网格日拦截成功: {e}")
    # 2021-06-15（周二）是端午后当周首个交易日，属合法网格日（顺延等价周一）
    print(f"  假日顺延网格日 2021-06-15 合法: "
          f"{is_week_start(md.trading_days, '2021-06-15')}")
    # 整除约束
    try:
        make_snowball_weekly(md.trading_days, monday, s0=1.0, tenor_weeks=103)
    except ValueError as e:
        print(f"  整除约束拦截成功: {str(e)[:40]}...")
