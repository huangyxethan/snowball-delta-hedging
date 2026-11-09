# -*- coding: utf-8 -*-
"""
futures_chain.py —— 第0层：IC合约选择与换月日历

回答一个核心问题：
    任意交易日 t，按给定规则（当月/次月/当季/下季），我应该持有哪个合约？
    今天是否需要换月？

核心概念：槽位（slot）
    中金所每天挂牌恰好4个IC合约：当月、次月、随后两个季月。
    把当日在市合约按到期年月升序排列，位置即槽位：
        slot 0 = 当月, slot 1 = 次月, slot 2 = 当季, slot 3 = 下季
    "持有当季"的含义 = 每天持有落在 slot 2 的那个合约。
    合约会随时间在槽位间迁移（下季→当季→次月→当月→到期摘牌），
    当某槽位上的合约代码发生变化时，该规则自然产生一次换月。

换月的两个触发来源：
    1. 挂牌结构变化：每月合约到期摘牌、新合约上市，槽位映射整体平移。
       这是 次月/当季/下季 规则唯一的换月来源——这些槽位上的合约
       距自身到期永远还有1个月以上，不存在被迫临近交割的问题。
    2. 临近交割保护（仅影响"当月"规则）：当选中合约的剩余交易日
       ≤ roll_buffer_days 时，提前切换到下一个合约，避免持仓进入交割。

持仓与损益的时间约定（与 hedger 层对齐）：
    schedule 中日期 t 的 code，指"t 日结算时刻起持有、直到 t+1 日结算"的合约。
    因此 t+1 日的损益 = settle_{t+1}(code_t) − settle_t(code_t)；
    换月日 t 的换仓动作以 t 日结算价平旧开新，本身不产生损益。
    （即：旧合约记换仓前的变动、新合约记换仓后的变动。）

多天展期（roll_window_days > 1）：
    把一次换月事件平摊到 n 个交易日完成：窗口第 k 天收盘后，新合约占
    目标仓位的 k/n（等步长过渡），窗口内新旧两合约同时持有。
    窗口锚定按换月的触发来源分两类：
      * 当月/次月（交割保护/摘牌驱动）：窗口**倒推**——完成日仍是单日
        换月的原换月日 R，从 R-(n-1) 开始分批。保住"距最后交易日
        roll_buffer_days 个交易日完成移仓"的安全边际不变；
      * 当季/下季（新合约上市驱动）：新合约在原换月日 R 才上市，无法
        提前买入，窗口**前推**为 [R, R+n-1]（旧合约距到期尚有约 2 个月，
        延后完成无交割风险）。
    统一实现：窗口起点 = max(R-(n-1), 新合约首个交易日)，终点 = 起点+n-1。
    schedule 以两切片列描述窗口内的持仓（code_out/code_in/w_in 及各自价格），
    hedger 层据此逐切片记账——价差仍然永不跨合约。
"""

import calendar

import numpy as np
import pandas as pd

try:
    from .data_loader import MarketData, load_market_data
except ImportError:  # 直接运行本文件做冒烟测试时走这个分支
    from data_loader import MarketData, load_market_data

# 规则名 -> 槽位
RULES = {"当月": 0, "次月": 1, "当季": 2, "下季": 3}


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """某年某月的第三个周五（IC理论最后交易日，未考虑节假日顺延）。"""
    fridays = [d for d in calendar.Calendar().itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return pd.Timestamp(fridays[2])


class FuturesChain:
    """IC合约链：合约信息、槽位判定、换月日历、价格查询。

    Parameters
    ----------
    md : MarketData
        data_loader.load_market_data() 的产出。
    roll_buffer_days : int
        临近交割保护：选中合约剩余交易日 ≤ 此值时切换到下一合约。
        实际只对"当月"规则生效（其余槽位的合约离到期都还很远）。
        默认5：在最后交易日前第5个交易日收盘完成移仓。
    """

    def __init__(self, md: MarketData, roll_buffer_days: int = 5):
        self.md = md
        self.buffer = int(roll_buffer_days)
        self.days = md.trading_days

        self.info = self._build_contract_info()

        # 宽表：date x code 的结算价/收盘价，供 O(1) 查价
        self._settle = md.ic.pivot(index="date", columns="code", values="settle")
        self._close = md.ic.pivot(index="date", columns="code", values="close")

        # 每日槽位：date -> [当月, 次月, 当季, 下季]（按到期年月升序）
        tmp = md.ic[["date", "code"]].copy()
        tmp["ym"] = tmp["code"].map(self.info["ym"])
        tmp = tmp.sort_values(["date", "ym"])
        self._slots = {d: list(codes) for d, codes in tmp.groupby("date")["code"]}

        # 剩余交易日计算用的位置索引
        self._day_pos = {d: i for i, d in enumerate(self.days)}
        self._last_pos = {
            code: int(np.searchsorted(self.days.values, np.datetime64(lt)))
            for code, lt in self.info["last_trade"].items()
        }

        self._schedule_cache: dict[tuple, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # 合约静态信息
    # ------------------------------------------------------------------

    def _build_contract_info(self) -> pd.DataFrame:
        """每个合约一行：上市/摘牌日期、到期年月、最后交易日。

        最后交易日的确定：
        - 已到期合约：数据中最后一次出现的日期就是真实最后交易日
          （自动吸收了节假日顺延，比按"第三个周五"推算更可靠）；
        - 数据末端仍存续的合约：用第三个周五估计（仅影响回测末尾几天，可接受）。
        """
        g = self.md.ic.groupby("code")["date"]
        info = pd.DataFrame({"first_date": g.min(), "last_date": g.max()})

        ym = info.index.str.extract(r"IC(\d{2})(\d{2})")
        info["year"] = 2000 + ym[0].astype(int).values
        info["month"] = ym[1].astype(int).values
        info["ym"] = info["year"] * 100 + info["month"]  # 排序键，如 202109

        data_end = self.days[-1]
        expiry_est = [_third_friday(y, m) for y, m in zip(info["year"], info["month"])]
        alive = info["last_date"] >= data_end
        info["last_trade"] = info["last_date"]
        info.loc[alive, "last_trade"] = [
            max(ld, ee) for ld, ee
            in zip(info.loc[alive, "last_date"], pd.Series(expiry_est, index=info.index)[alive])
        ]
        return info

    # ------------------------------------------------------------------
    # 核心查询
    # ------------------------------------------------------------------

    def remaining_days(self, code: str, date) -> int:
        """date 起（含当日）到该合约最后交易日的剩余交易日数。最后交易日当天为0。"""
        return self._last_pos[code] - self._day_pos[pd.Timestamp(date)]

    def contract_on(self, date, rule: str) -> str:
        """交易日 date 按规则应持有的合约代码。"""
        slots = self._slots[pd.Timestamp(date)]
        k = RULES[rule]
        code = slots[k]
        # 临近交割保护：顺延到下一个合约（实际只会在"当月"规则触发）
        if self.remaining_days(code, date) <= self.buffer and k + 1 < len(slots):
            code = slots[k + 1]
        return code

    def settle(self, date, code: str) -> float:
        """合约在指定日期的结算价（损益记账与换仓的价格基准）。

        宽表中"日期在、合约列在、但该日无此合约"的格子是 NaN——静默返回
        会击穿下游"持仓合约无价即报错"的防线（execution 层曾预填结算价，
        NaN 会无声进入 pnl），故此处直接 fail loud。
        """
        v = self._settle.at[pd.Timestamp(date), code]
        if pd.isna(v):
            raise KeyError(f"{code} 在 {pd.Timestamp(date).date()} 无结算价"
                           f"（未上市或已摘牌）")
        return float(v)

    def close(self, date, code: str) -> float:
        """合约在指定日期的收盘价（备用）。缺价直接报错，同 settle。"""
        v = self._close.at[pd.Timestamp(date), code]
        if pd.isna(v):
            raise KeyError(f"{code} 在 {pd.Timestamp(date).date()} 无收盘价"
                           f"（未上市或已摘牌）")
        return float(v)

    # ------------------------------------------------------------------
    # 换月日历（供 hedger 层使用的主接口）
    # ------------------------------------------------------------------

    def build_schedule(self, rule: str, roll_window_days: int = 1) -> pd.DataFrame:
        """整条持仓日历：每个交易日持有哪个（哪些）合约、结算价、换月信息。

        Parameters
        ----------
        rule : 当月 / 次月 / 当季 / 下季
        roll_window_days : int
            每次换月平摊到几个交易日完成（见模块 docstring"多天展期"）。
            1（默认）= 单日换月，与历史行为完全一致。

        Returns
        -------
        pd.DataFrame，index = 共同交易日。

        单合约视角的列（描述"单日换月"口径，n>1 时窗口内仅供报表参考）：
            code        : t日结算起的目标合约
            settle      : 该合约t日结算价（= 建/换仓的成交价基准）
            close       : 该合约t日收盘价
            is_roll     : t日是否为换月**事件日**（多天展期时窗口只算一次事件）
            settle_exit : code_{t-1} 在 t 日的结算价。非换月日等于 settle；
                          换月日为旧合约当日结算价。
                              PnL_t = 手数_{t-1} × (settle_exit_t − settle_{t-1})

        两切片视角的列（hedger 的记账依据；n=1 时退化为单合约）：
            code_out / code_in : 退出中 / 进入中的合约。非窗口日两者相同
            w_in               : t日收盘后进入合约应占目标仓位的比例
                                 （窗口第 k 天 = k/n，非窗口日 = 1）
            settle_out / settle_in           : 两合约的 t 日结算价（成交价基准）
            settle_exit_out / settle_exit_in : code_out_{t-1} / code_in_{t-1} 在
                t 日的结算价——各切片的损益价差**仍在同一合约内部**计算。
                窗口结束次日旧合约可能已摘牌（价格缺失），该日 out 切片持仓
                必为 0，缺价以 settle_out 填充、不参与任何计算
            in_roll_window     : t日是否处于某个展期窗口内
        """
        n = int(roll_window_days)
        if rule not in RULES:
            raise ValueError(f"未知规则 {rule!r}，可选: {list(RULES)}")
        if n < 1:
            raise ValueError("roll_window_days 须 >= 1")
        key = (rule, n)
        if key in self._schedule_cache:
            return self._schedule_cache[key]

        codes = [self.contract_on(d, rule) for d in self.days]
        sched = pd.DataFrame(index=self.days, data={"code": codes})

        st = self._settle.to_numpy()
        row = self._settle.index.get_indexer(sched.index)
        col = self._settle.columns.get_indexer(sched["code"])
        sched["settle"] = st[row, col]
        sched["close"] = self._close.to_numpy()[row, col]

        sched["is_roll"] = sched["code"].ne(sched["code"].shift(1))
        sched.iloc[0, sched.columns.get_loc("is_roll")] = False  # 首日是建仓不是换月

        prev_codes = sched["code"].shift(1).fillna(sched["code"].iloc[0])
        col_prev = self._settle.columns.get_indexer(prev_codes)
        sched["settle_exit"] = st[row, col_prev]

        if sched[["settle", "close", "settle_exit"]].isna().any().any():
            raise ValueError(f"规则 {rule} 的持仓日历存在缺价，数据或选择逻辑有误")

        # ---- 两切片列：把每次换月事件展开为 n 天的过渡窗口 -----------------
        # 窗口起点 = max(R-(n-1), 新合约首个交易日, 1)，终点 = 起点+n-1：
        #   当月/次月：新合约早已挂牌，起点即 R-(n-1)，完成日仍是 R（倒推）；
        #   当季/下季：新合约 R 日才上市，起点被抬到 R，窗口前推为 [R, R+n-1]。
        # w_in 每天走 1/n 步；窗口被数据末端截断时过渡停在中途（真实未完成）。
        base = sched["code"].to_numpy()
        code_out = base.copy()
        code_in = base.copy()
        w_in = np.ones(len(base))
        in_win = np.zeros(len(base), dtype=bool)
        prev_end = -1
        for r in np.flatnonzero(sched["is_roll"].to_numpy()):
            a, b = base[r - 1], base[r]
            first_b = self._day_pos[pd.Timestamp(self.info.at[b, "first_date"])]
            # 起点依次避让：倒推 n-1 天、新合约上市日、上一窗口终点、日历首日
            start = max(r - (n - 1), first_b, prev_end + 1, 1)
            if start > r:
                # 本事件落在上一窗口内部（三个合约同时在途），不支持——
                # 实际只会在相邻换月间隔 < 2 天的数据异常处发生
                raise ValueError(
                    f"换月事件 {self.days[r].date()}（{a}->{b}）落在上一展期"
                    f"窗口内，roll_window_days={n} 与该处换月间隔不兼容")
            # 终点：倒推窗口在事件日 R 完成；被上市日/上一窗口挤短的窗口
            # 向后补足到 n 天（当季/下季即前推为 [R, R+n-1]），但不越过
            # 旧合约的最后交易日
            end = min(max(r, start + n - 1), self._last_pos[a])
            prev_end = end
            end = min(end, len(base) - 1)      # 数据末端截断（过渡真实未完成）
            nn = prev_end - start + 1          # 计划步数（据未截断的终点）
            for k, t in enumerate(range(start, end + 1), 1):
                code_out[t] = a
                code_in[t] = b
                w_in[t] = k / nn
                in_win[t] = True

        sched["code_out"] = code_out
        sched["code_in"] = code_in
        sched["w_in"] = w_in
        sched["in_roll_window"] = in_win
        col_out = self._settle.columns.get_indexer(code_out)
        col_in = self._settle.columns.get_indexer(code_in)
        sched["settle_out"] = st[row, col_out]
        sched["settle_in"] = st[row, col_in]
        co_prev = np.concatenate([code_out[:1], code_out[:-1]])
        ci_prev = np.concatenate([code_in[:1], code_in[:-1]])
        sched["settle_exit_out"] = st[row, self._settle.columns.get_indexer(co_prev)]
        sched["settle_exit_in"] = st[row, self._settle.columns.get_indexer(ci_prev)]
        # 窗口结束次日旧合约可能已摘牌：该日 out 切片持仓必为 0（w 已到 1），
        # 缺价不参与计算，填当日值以维持"无 NaN"不变式
        sched["settle_exit_out"] = sched["settle_exit_out"].fillna(sched["settle_out"])
        if sched[["settle_out", "settle_in", "settle_exit_out",
                  "settle_exit_in"]].isna().any().any():
            raise ValueError(f"规则 {rule} 的展期窗口存在缺价，窗口推导有误")

        self._schedule_cache[key] = sched
        return sched

    # ------------------------------------------------------------------
    # 供 market_params 层估计 repo 的接口
    # ------------------------------------------------------------------

    def quarterly_for_repo(self, date) -> tuple[str, float, float]:
        """repo 估计所需的当季合约信息：(代码, 结算价, 剩余期限年数)。

        直接取 slot 2 的原始合约（不做交割保护顺延——该槽位剩余期限
        天然在约2~4个月之间，不存在 tau->0 的问题）。
        期限按自然日/365 返回；换算到引擎的交易日时钟
        （x 252/年均交易日数）在 market_params.estimate_repo 完成。
        """
        d = pd.Timestamp(date)
        code = self._slots[d][2]
        tau = (self.info.at[code, "last_trade"] - d).days / 365.0
        return code, self.settle(d, code), tau


if __name__ == "__main__":
    md = load_market_data()
    chain = FuturesChain(md, roll_buffer_days=5)

    n_years = (md.trading_days[-1] - md.trading_days[0]).days / 365.25
    print(f"合约总数: {len(chain.info)}，回测区间约 {n_years:.1f} 年\n")

    for rule in RULES:
        sched = chain.build_schedule(rule)
        n_roll = int(sched["is_roll"].sum())
        hold = sched.groupby((sched["is_roll"]).cumsum()).size()
        print(f"[{rule}] 换月 {n_roll} 次（约 {n_roll / n_years:.1f} 次/年），"
              f"单合约平均持有 {hold.mean():.0f} 个交易日")

    # 抽查一次换月现场：换月日前后各2天
    sched_q = chain.build_schedule("当季")
    roll_days = sched_q.index[sched_q["is_roll"]]
    r = roll_days[len(roll_days) // 2]
    pos = chain._day_pos[r]
    print(f"\n[当季] 抽查换月日 {r.date()} 前后：")
    print(sched_q.iloc[pos - 2: pos + 3].to_string())

    # 当季槽位的剩余期限范围（验证不存在 tau->0）
    taus = [chain.quarterly_for_repo(d)[2] for d in md.trading_days]
    print(f"\n当季槽位剩余期限: 最小 {min(taus) * 12:.1f} 个月, "
          f"最大 {max(taus) * 12:.1f} 个月（repo 分母安全）")

    # 交割保护抽查：当月规则在某合约最后交易日附近的行为
    sched_m = chain.build_schedule("当月")
    some_code = sched_m["code"].iloc[100]
    lt = chain.info.at[some_code, "last_trade"]
    pos_lt = chain._day_pos[pd.Timestamp(lt)]
    print(f"\n[当月] 合约 {some_code} 最后交易日 {pd.Timestamp(lt).date()} 附近的持仓：")
    print(sched_m.iloc[pos_lt - 4: pos_lt + 2][["code", "is_roll"]].to_string())

    # ---- 多天展期窗口自检 ---------------------------------------------------
    print("\n" + "=" * 70)
    print("多天展期窗口自检（roll_window_days=5）")
    print("=" * 70)
    for rule in RULES:
        s1 = chain.build_schedule(rule, roll_window_days=1)
        s5 = chain.build_schedule(rule, roll_window_days=5)
        # (a) 单合约视角的列不受窗口影响；n=1 的切片列退化为单合约
        legacy_same = all(s1[c].equals(s5[c])
                          for c in ("code", "settle", "settle_exit", "is_roll"))
        deg = bool((s1["code_out"] == s1["code_in"]).all()
                   or ((s1["w_in"] == 1.0).all()))
        # (b) 每个窗口：w 走 1/5 阶梯、事件数 = 窗口数、窗口内两合约都有价
        n_ev = int(s5["is_roll"].sum())
        win_days = int(s5["in_roll_window"].sum())
        wvals = sorted(s5.loc[s5["in_roll_window"], "w_in"].unique())
        # (c) 窗口锚定：每个事件要么倒推（事件日 R 完成，w[R]=1，进入合约
        #     早已挂牌——当月/次月/当季皆如此），要么前推（进入合约 R 日才
        #     上市——下季规则），要么被上一窗口/数据末端挤压（罕见，数据
        #     末端的伪换月处）
        bad_anchor = 0
        for d in s5.index[s5["is_roll"]]:
            listed_at_r = (chain.info.at[s5.at[d, "code"], "first_date"] == d)
            if s5.at[d, "w_in"] < 1.0 and not listed_at_r \
                    and d > s5.index[5] and d < s5.index[-5]:
                bad_anchor += 1
        print(f"[{rule}] 事件 {n_ev} 次，窗口日 {win_days} 天"
              f"（应≈{n_ev * 5}），w 阶梯 {[f'{w:.1f}' for w in wvals]}，"
              f"单合约列一致={legacy_same}，n=1退化={deg}，锚定异常={bad_anchor}")

    # 抽查一个当月窗口现场
    s5 = chain.build_schedule("当月", roll_window_days=5)
    r = s5.index[s5["is_roll"]][30]
    pos = chain._day_pos[r]
    print(f"\n[当月] 抽查换月事件 {r.date()} 的展期窗口（前推4日开始，事件日完成）：")
    print(s5.iloc[pos - 6: pos + 3][["code", "code_out", "code_in", "w_in",
                                     "settle_out", "settle_in",
                                     "in_roll_window"]].to_string())
