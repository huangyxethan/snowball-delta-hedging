# -*- coding: utf-8 -*-
"""
hedger.py —— 第3层：对冲执行与逐日损益记账

把定价层给出的 delta 序列转成 IC 期货持仓，并逐日盯市，产出净值。
本模块不含任何定价逻辑，也不判断敲入敲出——它只做"给定每日目标敞口，
执行并记账"这一件事。

时间约定（与 futures_chain.build_schedule 一致）
------------------------------------------------
    t 日结算时刻: 先用 t-1 日建立的持仓结算当日损益，再按 t 日的目标敞口调仓。
      PnL_t   = 手数_{t-1} × (settle_exit_t − settle_{t-1}) × 乘数
      手数_t  = delta_t × 名义基数_t / (settle_t × 乘数)

    settle_exit_t 是"自 t-1 起持有的合约"在 t 日的结算价：
      * 非换月日 = settle_t（同一合约）
      * 换月日   = 旧合约当日结算价
    这样价差永远在同一合约内部计算。换月动作（以 settle_exit_t 平旧、
    以 settle_t 开新）本身不产生损益，但会改变后续的基差水平。

    delta_t 由 t 日收盘价与 t 日市场参数算出，仅用于 t 日之后的损益 ——
    因此不存在前视偏差。

损益分解（传入 spot 序列时提供）
--------------------------------
    F = S + B（B 为基差），故
      PnL_t = 手数_{t-1} × [(S_t − S_{t-1}) + (B_t − B_{t-1})] × 乘数
            = 方向性损益 + 基差损益
    其中 B_t 用**旧合约**的口径：B_t = settle_exit_t − S_t，
    B_{t-1} = settle_{t-1} − S_{t-1}，两项之和恒等于总损益。

手续费
------
    按**单边成交名义额**的固定比例（fee_rate，默认万三）计费：
      fee_t = 当日各笔成交的 |手数| × 成交价基准 × 乘数 × fee_rate
    成交价基准与换仓口径一致：非换月日为 settle_t；换月日旧仓按 settle_exit_t
    平仓、新仓按 settle_t 开仓，分别计费。fee_rate=0 退回无成本口径。

名义基数与净值口径（notional_mode）
----------------------------------
    "fixed"（固定名义）: 名义基数_t ≡ 名义本金。每日损益始终相对期初本金计，
        故净值是每日收益的**算术和**：NAV_T = 1 + sum_t delta_{t-1} r_t。
        与名义规模无关、与路径顺序无关，各项损益可精确相加——归因分析口径。

    "leg"（每腿重置；回测层 BacktestConfig 的默认）: 名义基数在 reset_dates
        （各腿发行日）重置为**前一
        交易日收盘净值**，腿内保持不变。已实现盈亏闲置到下一只雪球建仓时才投入，
        故净值是**腿内算术、腿间连乘**：
            NAV(腿k末) = NAV(腿k初) × (1 + sum_{t in 腿k} delta_{t-1} r_t)
        对应"复制一只名义固定的虚拟雪球"的叙事：delta 乘的是该腿发行时确定的
        名义，腿内的对冲盈亏不改变对冲头寸规模。

        基数取前一日收盘而非当日盯市净值：默认 reissue_lag_days=5 时腿间有空仓
        （无持仓、无成交，净值不动），两者恒等；reissue_lag_days=0 时前者能让
        "各腿收益连乘 = 全周期总收益"精确成立。

    "daily"（每日复利）: 名义基数_t = 名义本金 × NAV_t（盯市后、付费前）。
        当日盈亏立刻计入次日的建仓规模，即净值化基金的运作方式：管理人收盘时
        按当前 AUM 的 delta 比例配敞口，账上无闲置的已实现盈亏。
        由 PnL_t = delta_{t-1} × 名义 × NAV_{t-1} × r_t 直接推出
            NAV_t = NAV_{t-1} × (1 + delta_{t-1} r_t)
        故净值自动变为**逐日连乘**。

    三种模式下净值公式本身完全相同：
        NAV_t = 1 + (累计损益_t − 累计手续费_t) / 名义本金
    差别只在建仓时用的名义基数（daily 列 notional_t 记录了逐日取值）。复利性质
    完全由基数的缩放产生，不需要另改净值公式。

    代价是 leg / daily 成为递推：基数依赖历史净值，不能先算完整条 pnl 再
    cumsum，必须逐日推进。两者下净值均可被推至 0 以下（delta 可超 100%），
    故基数重置时若净值 <= 0 则报错。

    不含冲击成本、保证金占用与现金管理收益（见 README §0.2 的缺省声明）。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

IC_MULTIPLIER = 200.0     # 中证500股指期货合约乘数（元/点）
DEFAULT_NOTIONAL = 1e8    # 名义本金（元）。影响取整误差，不影响净值口径
DEFAULT_FEE_RATE = 3e-4   # 单边手续费率（成交名义额的比例），默认万三
NOTIONAL_MODES = ("fixed", "leg", "daily")   # 见模块 docstring


@dataclass(frozen=True)
class HedgeResult:
    """逐日对冲明细与汇总。"""

    daily: pd.DataFrame   # 见 run_hedge 的 Returns 说明
    notional: float
    multiplier: float
    notional_mode: str = "fixed"
    sleeves: tuple = None   # 多套筒时为列名元组；单套筒（Series 输入）为 None

    @property
    def compounding(self) -> bool:
        """净值是否为连乘序列（leg 为腿内算术、腿间连乘，年化仍取几何）。"""
        return self.notional_mode != "fixed"

    @property
    def pnl_base(self) -> pd.Series:
        """逐日损益的归一化基数 —— 使 (pnl − fee) / pnl_base 为该基数上的收益。

        单套筒：
            fixed: 恒为名义本金        => 1 + cumsum(收益) 精确还原净值
            leg  : 当腿的名义基数      => 腿内累加、腿间连乘精确还原净值
            daily: 名义本金 × 前一日**收盘**净值 => cumprod(1+收益) 精确还原净值

        用同一个基数归一 pnl / fee / pnl_spot / pnl_basis，各归因项才能精确
        相加。fixed / leg 下它就是 notional_t（腿窗口内恒定；但 leg 下基数
        在腿切换日跳变，切换日的 pnl 属 t-1 日建立的旧腿持仓，腿级归因须配合
        shift(1) 使用——见 backtest 归因处。仅当切换日前空仓时无需错位）；
        daily 下与 notional_t（建仓基数，取盯市后、付费前）相差一个当日手续费
        ——那是建仓口径，这是归因口径，用途不同。

        多套筒：leg 下各套筒锚定日交错，净值没有连乘重述式，但恒有
        NAV_t = 1 + (累计损益 − 累计手续费)/notional，故归因基数取**恒定的
        期初名义**——(pnl − fee)/notional 逐日相加精确还原净值，各归因项
        严格可加（daily 下仍取前一日收盘净值 × notional）。
        """
        if self.notional_mode == "daily":
            return self.notional * self.daily["nav"].shift(1).fillna(1.0)
        if self.sleeves is not None and self.notional_mode == "leg":
            return pd.Series(self.notional, index=self.daily.index)
        return self.daily["notional_t"]

    @property
    def nav(self) -> pd.Series:
        return self.daily["nav"]

    @property
    def total_return(self) -> float:
        return float(self.daily["nav"].iloc[-1] - 1.0)

    def summary(self) -> pd.Series:
        d = self.daily
        n_days = len(d)
        years = n_days / 252.0
        ret = self.total_return
        nav = d["nav"]
        dd = (nav / nav.cummax() - 1.0).min()
        # fixed: 净值为线性累计，日收益取绝对差、年化取线性
        # daily: 净值为连乘，日收益取相对变化、年化取几何
        daily_ret = nav.pct_change().fillna(0.0) if self.compounding \
            else nav.diff().fillna(0.0)
        if years <= 0:
            ann_ret = np.nan
        elif self.compounding:
            ann_ret = float(nav.iloc[-1]) ** (1.0 / years) - 1.0
        else:
            ann_ret = ret / years
        return pd.Series({
            "起始日": d.index[0].date(),
            "结束日": d.index[-1].date(),
            "交易日数": n_days,
            "名义口径": self.notional_mode,
            "总收益": ret,
            "年化收益": ann_ret,
            "年化波动": daily_ret.std() * np.sqrt(252),
            "最大回撤": dd,
            "平均delta敞口": d["delta"].mean(),
            "最大delta敞口": d["delta"].max(),
            "换月次数": int(d["is_roll"].sum()),
            "累计成交手数": d["traded_lots"].sum(),
            "日均换手(手)": d["traded_lots"].mean(),
            # 按名义基数逐日归一（fixed 下即退回 fee/名义本金）
            "手续费/名义": (d["fee"] / self.pnl_base).sum(),
        })


def run_hedge(delta, schedule: pd.DataFrame,
              spot: pd.Series = None,
              notional: float = DEFAULT_NOTIONAL,
              multiplier: float = IC_MULTIPLIER,
              round_lots: bool = False,
              delta_cap: float = None,
              close_at_end: bool = True,
              fee_rate: float = DEFAULT_FEE_RATE,
              notional_mode: str = "fixed",
              reset_dates=None,
              sleeve_weights=None) -> HedgeResult:
    """按 delta 序列执行对冲并逐日记账（单套筒或多套筒）。

    Parameters
    ----------
    delta : pd.Series | pd.DataFrame
        每日目标 **delta 敞口比例**（如 0.82 表示 82% 名义本金的多头），
        index 为交易日，须为 schedule 索引的子集。序列的首日即建仓日、
        末日即终止日（close_at_end=True 时强制平仓）。

        * pd.Series —— 单套筒，与历史行为完全一致；
        * pd.DataFrame —— **多套筒**（ladder 回测）：每列一个套筒（槽位），
          列值为该套筒名义基数上的敞口比例（未启动/空仓段填 0，不允许 NaN）。
          各套筒逐日聚合成**单一账户的总目标手数**后一次调仓：

              总手数_t = sum_k delta_t^(k) × 基数_t^(k) / (settle_t × 乘数)

          损益、换月、净值均按总持仓记账（一个期货账户）；手续费按**净额
          成交**计——同日一个套筒加仓、另一个减仓自然抵消，这是相对分账户
          执行的真实费用优势。逐套筒的 lots/pnl/base 单列输出供归因
          （fee 按各套筒成交量占比分摊）。
    schedule : pd.DataFrame
        FuturesChain.build_schedule(rule) 的产出，须含
        code / settle / settle_exit / is_roll 四列。
    spot : pd.Series, optional
        指数收盘价序列。给出时额外产出方向性/基差损益分解。
    notional : float
        名义本金（元）。仅通过手数取整影响结果；净值口径与之无关
        （复利模式下建仓基数为 notional × 当日净值，比例关系不变，故同样无关）。
    multiplier : float
        合约乘数（IC = 200 元/点）。
    round_lots : bool
        是否取整到整数手。默认 False（编指数口径）：持仓为连续合约数，
        乘数在净值中自动消去，无执行颗粒度约束；True 为执行层口径，
        更贴近实盘但引入跟踪误差（量级见 README §5.5）。
    delta_cap : float, optional
        敞口上限（如 2.0 表示 200%）。None 为不截断。
        敲入线附近的模型 delta 可远超 100%，实盘受保证金约束通常需设限。
    close_at_end : bool
        末日是否强制平仓（产品终止）。默认 True。
    fee_rate : float
        单边手续费率（成交名义额的比例），默认万三（3e-4）。0 为免手续费。
        换月日旧仓按 settle_exit、新仓按 settle 分别计费。
    notional_mode : str
        建仓所用名义基数的口径，见模块 docstring：
        "fixed"（本函数默认）= 恒为 notional，净值为算术和；
        "leg"                = 在 reset_dates 按前一日收盘净值重置，窗口内不变，
                               净值为窗口内算术、窗口间连乘；
        "daily"              = notional × 当日盯市净值，净值为逐日连乘。

        多套筒下的 "leg"：套筒 k 在**自己的**重置日把基数锚定为
        权重_k × notional × 前一日收盘的**组合净值**，此后冻结——即"新发
        雪球按当前总 AUM 的份额定名义"，隐含套筒间的资本互通。此时净值不再
        有连乘的重述式（各套筒锚定日交错），但恒有
        NAV_t = 1 + (累计损益 − 累计手续费)/notional。
    reset_dates : optional
        仅 "leg" 模式使用且必填：基数重置日（回测中即各腿发行日）。本模块不
        感知"腿"的概念，重置点须由调用方给出。
        单套筒（Series 输入）传日期序列；多套筒（DataFrame 输入）传
        {列名: 日期序列} 的映射，每个套筒一套重置日。首日总是隐含的重置点
        （基数 = 权重 × notional）；不在 delta 索引内的日期报错。
    sleeve_weights : optional
        各套筒的资本权重，每日和须为 1。None = 等权（单套筒即 1.0）。
        静态权重传 {列名: 权重} 或与列序对应的序列；**逐日权重**传
        DataFrame（index 覆盖 delta 日期、列同 delta 列）——权重可随时间
        变化（如满仓建仓的 1 → 1/2 → ... → 1/K 缩档）。"leg" 口径下，
        权重变化会同比例重定标存量腿的冻结基数（基数 = 当日权重 × notional
        × 锚定时的净值水平），即"缩旧腿的仓、腾资金给新腿"。

    Returns
    -------
    HedgeResult，其中 daily 的列：
        delta        : 截断后的目标敞口比例（多套筒下 = 聚合金额敞口 /
                       聚合名义基数，即组合敞口比例）
        notional_t   : 当日建仓所用的名义基数（fixed 下恒为 notional；
                       多套筒下为各套筒基数之和）
        code         : 当日收盘后持有的合约
        settle       : 该合约当日结算价（建/换仓价）
        settle_exit  : 自前一日起持有的合约在当日的结算价（记账价）
        is_roll      : 当日是否换月
        lots         : 当日收盘后的持仓手数（正=多头）
        pnl          : 当日盯市损益（元，未扣手续费）
        cum_pnl      : 累计盯市损益（元，未扣手续费）
        fee          : 当日手续费（元）
        cum_fee      : 累计手续费（元）
        nav          : 净值 = 1 + (累计损益 − 累计手续费) / 名义本金
        traded_lots  : 当日成交手数（换月日 = 平旧 + 开新；多套筒下为净额；
                       多天展期下窗口内逐日分批）
        w_in / lots_out / lots_in / code_out / code_in :
                       多天展期的切片明细（进入合约权重、两合约上的手数与
                       代码；单日换月下 lots_out 恒为 0）
        pnl_spot     : 方向性损益（元，仅当传入 spot）
        pnl_basis    : 基差损益（元，仅当传入 spot；逐切片在各自合约内计算）
    多套筒（DataFrame 输入）时另有逐套筒列：
        delta_<列> / lots_<列> / pnl_<列> / fee_<列> / base_<列>
        （lots 为连续手数；fee 按套筒成交量占比分摊，和 = 总 fee；
        round_lots=True 时总手数在聚合后取整，套筒 lots/pnl 保持连续口径，
        套筒 pnl 之和与总 pnl 差一个取整残差）
    """
    if notional_mode not in NOTIONAL_MODES:
        raise ValueError(f"notional_mode 须为 {NOTIONAL_MODES} 之一，"
                         f"收到 {notional_mode!r}")
    # 校验 fee_rate 单位："成交名义额的比例"，万五 = 5e-4。期货手续费率不可能
    # 超过 1%——超过即视为单位写错（如把万五误写成 5）。
    if not 0.0 <= fee_rate <= 0.01:
        raise ValueError(f"fee_rate={fee_rate:g} 超出合理范围 [0, 0.01]。"
                         f"单位是成交名义额的比例：万五请写 5e-4（0.0005），"
                         f"千一请写 1e-3")

    # ---- 输入归一：Series = 单套筒，DataFrame = 每列一个套筒 ----------------
    single = isinstance(delta, pd.Series)
    dframe = delta.to_frame("delta") if single else delta
    cols = list(dframe.columns)
    K = len(cols)
    if K == 0:
        raise ValueError("delta DataFrame 至少需要一列")

    dates = pd.DatetimeIndex(dframe.index)
    if not dates.isin(schedule.index).all():
        missing = dates[~dates.isin(schedule.index)]
        raise ValueError(f"delta 序列有 {len(missing)} 个日期不在持仓日历中，"
                         f"首个: {missing[0].date()}")
    if not dates.is_monotonic_increasing:
        raise ValueError("delta 序列的日期必须升序")
    n = len(dates)

    # 套筒权重（每日行和为 1；单套筒即 1.0）。
    # 静态：None（等权）/ {列名: 权重} / 序列；
    # 逐日：DataFrame（index 覆盖 delta 日期、列同 delta 列）——权重可随时间
    # 变化（如 rampup_mode="full" 的建仓期 1 → 1/2 → ... → 1/K 缩档），
    # "leg" 口径下权重变化会同比例重定标存量腿的冻结基数（缩旧腿让新腿）。
    if isinstance(sleeve_weights, pd.DataFrame):
        wdf = sleeve_weights.reindex(index=dates)
        if set(wdf.columns) != set(cols):
            raise ValueError(f"sleeve_weights 的列 {list(wdf.columns)} "
                             f"须与 delta 列 {cols} 一致")
        w_mat = wdf[cols].to_numpy(dtype=float)
        if np.isnan(w_mat).any():
            raise ValueError("sleeve_weights 在 delta 的日期上存在缺失")
    else:
        if sleeve_weights is None:
            w = np.full(K, 1.0 / K)
        elif isinstance(sleeve_weights, dict):
            w = np.asarray([sleeve_weights[c] for c in cols], dtype=float)
        else:
            w = np.asarray(sleeve_weights, dtype=float)
        if len(w) != K:
            raise ValueError(f"sleeve_weights 须与套筒数({K})对应")
        w_mat = np.broadcast_to(w, (n, K))
    if np.abs(w_mat.sum(axis=1) - 1.0).max() > 1e-9:
        raise ValueError("sleeve_weights 每日的权重和须为 1")

    # 基数重置点（首日总是重置点）。仅 "leg" 模式需要，逐套筒各一套。
    is_reset = np.zeros((n, K), dtype=bool)
    if notional_mode == "leg":
        if reset_dates is None:
            raise ValueError('notional_mode="leg" 须提供 reset_dates（各腿发行日）')
        if isinstance(reset_dates, dict):
            if set(reset_dates) != set(cols):
                raise ValueError(f"reset_dates 的键 {sorted(reset_dates)} "
                                 f"须与 delta 列 {cols} 一致")
            reset_map = reset_dates
        elif single:
            reset_map = {cols[0]: reset_dates}
        else:
            raise ValueError("多套筒（DataFrame 输入）的 reset_dates "
                             "须为 {列名: 日期序列} 映射")
        for k, c in enumerate(cols):
            resets = pd.DatetimeIndex(reset_map[c])
            if not resets.isin(dates).all():
                bad = resets[~resets.isin(dates)]
                raise ValueError(f"套筒 {c!r} 的 reset_dates 有 {len(bad)} 个"
                                 f"日期不在 delta 索引内，首个: {bad[0].date()}")
            is_reset[:, k] = dates.isin(resets)
    else:
        if reset_dates is not None:
            raise ValueError(f"reset_dates 仅用于 notional_mode='leg'，"
                             f"当前为 {notional_mode!r}")

    sch = schedule.loc[dates]
    tgt = dframe.to_numpy(dtype=float).copy()          # (n, K)
    if np.isnan(tgt).any():
        raise ValueError("delta 含 NaN：未启动/已终止的套筒段请先填 0")
    if delta_cap is not None:
        # 逐套筒截断（单套筒语义不变；聚合敞口 <= cap × 聚合基数）
        tgt = np.clip(tgt, -delta_cap, delta_cap)
    if close_at_end:
        tgt[-1, :] = 0.0                   # 终止日平掉全部头寸

    settle = sch["settle"].to_numpy(dtype=float)
    settle_exit = sch["settle_exit"].to_numpy(dtype=float)
    is_roll = sch["is_roll"].to_numpy(dtype=bool)
    # 两切片列（多天展期；见 futures_chain.build_schedule docstring）。
    # 兼容缺列的手工 schedule：合成单合约切片（w_in 恒为 1）。
    if "w_in" in sch.columns:
        c_out = sch["code_out"].to_numpy()
        c_in = sch["code_in"].to_numpy()
        w_arr = sch["w_in"].to_numpy(dtype=float)
        s_out = sch["settle_out"].to_numpy(dtype=float)
        s_in = sch["settle_in"].to_numpy(dtype=float)
        sx_out = sch["settle_exit_out"].to_numpy(dtype=float)
        sx_in = sch["settle_exit_in"].to_numpy(dtype=float)
    else:
        c_out = c_in = sch["code"].to_numpy()
        w_arr = np.ones(n)
        s_out, s_in = settle, settle
        sx_out, sx_in = settle_exit, settle_exit

    lots_slv = np.zeros((n, K))     # 逐套筒连续手数（归因口径，总手数等价量）
    lots_tot = np.zeros(n)          # 账户总手数（round_lots 在此层取整）
    lots_out = np.zeros(n)          # 总手数在退出/进入合约上的切片
    lots_in = np.zeros(n)
    pnl = np.zeros(n)
    pnl_slv = np.zeros((n, K))
    traded = np.zeros(n)
    fee = np.zeros(n)
    fee_slv = np.zeros((n, K))
    base_slv = np.zeros((n, K))     # 逐套筒名义基数
    nav_prev = 1.0                  # 上一交易日收盘净值（已扣当日手续费）
    base_nav = np.ones(K)           # 各套筒当前生效的基数水平（净值单位）

    # leg / daily 是递推：base 依赖历史净值，而净值依赖 pnl[:i]，
    # 故 pnl 无法先整条算完再 cumsum，必须在循环内逐日推进。
    for i in range(n):
        # 1) 先用昨日持仓结算今日损益：逐切片在**各自合约内部**做价差
        #    （settle_exit_out/in 按昨日切片合约取今日结算价），blend 即
        #    昨日每一手（按昨日切片比例分布在两合约上）的加权价差。
        #    非窗口日 w=1，blend 退化为单合约价差。
        if i > 0:
            d_out = sx_out[i] - s_out[i - 1]
            d_in = sx_in[i] - s_in[i - 1]
            blend = (1.0 - w_arr[i - 1]) * d_out + w_arr[i - 1] * d_in
            pnl[i] = lots_tot[i - 1] * blend * multiplier
            pnl_slv[i] = lots_slv[i - 1] * blend * multiplier

        # 2) 确定今日建仓所用的名义基数。
        #    daily: 全部套筒取盯市后、付费前的净值 —— 这个顺序解开了"仓位→
        #      手续费→净值→仓位"的循环依赖，且与实际操作一致：收盘先知道
        #      账户值多少钱，据此下单，再付这笔单子的费。
        #    leg  : 套筒 k 仅在自己的重置日按**前一日收盘的组合净值**换挡，
        #      窗口内冻结（多套筒下即"新发雪球按当前总 AUM 的份额定名义"）。
        #    fixed: 恒为 1。
        nav_mtm = nav_prev + pnl[i] / notional
        if notional_mode == "daily":
            base_nav[:] = nav_mtm
        elif notional_mode == "leg":
            hit = is_reset[i]
            if hit.any():
                base_nav[hit] = nav_prev
        if notional_mode != "fixed" and (base_nav <= 0.0).any():
            raise ValueError(
                f"{dates[i].date()} 净值 {base_nav.min():.4f} <= 0，"
                f"notional_mode={notional_mode!r} 下名义基数不可为非正"
                f"（考虑设 delta_cap 限制杠杆）")
        base_slv[i] = w_mat[i] * notional * base_nav

        # 3) 再按今日目标敞口调仓：套筒金额敞口聚合成账户总手数，再按 w_in
        #    切成（退出合约, 进入合约）两片。手数换算统一用单合约视角的
        #    settle（目标合约价）——切片间的换算价差为二阶小量，换取单套筒
        #    口径与历史行为严格一致。
        raw = tgt[i] * base_slv[i] / (settle[i] * multiplier)
        lots_slv[i] = raw
        agg = raw.sum()
        lots_tot[i] = np.round(agg) if round_lots else agg
        lots_in[i] = lots_tot[i] * w_arr[i]
        lots_out[i] = lots_tot[i] * (1.0 - w_arr[i])

        # 4) 逐切片调仓与手续费：每片的成交 = 今日目标 − 昨日落在该片合约上
        #    的持仓，按该片合约自己的结算价计费。窗口期内展期动作与 delta
        #    调仓自然合并为净额；单日换月（w 一步到 1）自动退化为
        #    "平旧按旧合约结算价、开新按新合约结算价"的历史口径。
        if c_out[i] == c_in[i]:            # 非窗口日：单合约，全部归 in 片
            tgt_out_l, tgt_in_l = 0.0, lots_tot[i]
        else:
            tgt_in_l = lots_in[i]
            tgt_out_l = lots_out[i]
        if i == 0:
            prev_on_out = prev_on_in = 0.0
        else:
            prev_on_out = 0.0
            if c_out[i] != c_in[i]:
                if c_out[i] == c_out[i - 1]:
                    prev_on_out += lots_out[i - 1]
                if c_out[i] == c_in[i - 1]:
                    prev_on_out += lots_in[i - 1]
            prev_on_in = 0.0
            if c_in[i] == c_out[i - 1]:
                prev_on_in += lots_out[i - 1]
            if c_in[i] == c_in[i - 1]:
                prev_on_in += lots_in[i - 1]
            # 昨日持仓须被今日两片完整接住（合约衔接不允许漏仓）
            lost = (lots_out[i - 1] + lots_in[i - 1]) - (prev_on_out + prev_on_in)
            if abs(lost) > 1e-9 * max(1.0, abs(lots_tot[i - 1])):
                raise ValueError(f"{dates[i].date()} 切片衔接丢失持仓 "
                                 f"{lost:.6g} 手，schedule 的窗口结构有误")
        trade_out = tgt_out_l - prev_on_out
        trade_in = tgt_in_l - prev_on_in
        traded[i] = abs(trade_out) + abs(trade_in)
        fee[i] = abs(trade_out) * s_out[i] + abs(trade_in) * s_in[i]
        fee[i] *= multiplier * fee_rate
        # 套筒间分摊基数：窗口日/换月日各套筒的展期成交与持仓规模相关，
        # 取 |昨仓|+|今仓|；平日取 |变化量|（与历史单日换月口径一致）
        if i == 0:
            t_slv = np.abs(lots_slv[i])
        elif c_out[i] != c_in[i] or c_in[i] != c_in[i - 1]:
            t_slv = np.abs(lots_slv[i - 1]) + np.abs(lots_slv[i])
        else:
            t_slv = np.abs(lots_slv[i] - lots_slv[i - 1])
        t_sum = t_slv.sum()
        if t_sum > 0.0:
            fee_slv[i] = fee[i] * t_slv / t_sum   # 归因分摊，和 = 总 fee

        # 5) 收盘净值（扣除当日手续费）
        nav_prev = nav_mtm - fee[i] / notional

    cum = np.cumsum(pnl)
    cum_fee = np.cumsum(fee)
    # 净值公式三种模式完全相同；复利性质由 base 的缩放自动产生。
    # 循环内的滚动 nav 与事后 cumsum 必然一致，此处断言之。
    if abs((1.0 + (cum[-1] - cum_fee[-1]) / notional) - nav_prev) > 1e-9:
        raise ValueError("滚动净值与累计损益不自洽")

    base_tot = base_slv.sum(axis=1)
    if K == 1:
        agg_delta = tgt[:, 0]
    else:
        # 组合敞口比例 = 聚合金额敞口 / 聚合基数
        with np.errstate(invalid="ignore", divide="ignore"):
            agg_delta = np.where(base_tot > 0.0,
                                 (tgt * base_slv).sum(axis=1) / base_tot, 0.0)
    out = pd.DataFrame({
        "delta": agg_delta,
        "notional_t": base_tot,
        "code": sch["code"].to_numpy(),
        "settle": settle,
        "settle_exit": settle_exit,
        "is_roll": is_roll,
        "lots": lots_tot,
        "pnl": pnl,
        "cum_pnl": cum,
        "fee": fee,
        "cum_fee": cum_fee,
        "nav": 1.0 + (cum - cum_fee) / notional,
        "traded_lots": traded,
        "w_in": w_arr,
        "lots_out": lots_out,
        "lots_in": lots_in,
        "code_out": c_out,
        "code_in": c_in,
    }, index=dates)

    if not single:
        for k, c in enumerate(cols):
            out[f"delta_{c}"] = tgt[:, k]
            out[f"lots_{c}"] = lots_slv[:, k]
            out[f"pnl_{c}"] = pnl_slv[:, k]
            out[f"fee_{c}"] = fee_slv[:, k]
            out[f"base_{c}"] = base_slv[:, k]

    if spot is not None:
        s = spot.reindex(dates).to_numpy(dtype=float)
        if np.isnan(s).any():
            raise ValueError("spot 序列在 delta 的日期上存在缺失")
        lots_prev = np.concatenate([[0.0], lots_tot[:-1]])
        lo_prev = np.concatenate([[0.0], lots_out[:-1]])
        li_prev = np.concatenate([[0.0], lots_in[:-1]])
        d_spot = np.concatenate([[0.0], np.diff(s)])
        # 基差逐切片计算（各切片在自己合约内）：B_t = settle_exit − S_t
        b_out = sx_out - s
        b_out_prev = np.concatenate([[0.0], (s_out - s)[:-1]])
        d_b_out = np.concatenate([[0.0], (b_out[1:] - b_out_prev[1:])])
        b_in = sx_in - s
        b_in_prev = np.concatenate([[0.0], (s_in - s)[:-1]])
        d_b_in = np.concatenate([[0.0], (b_in[1:] - b_in_prev[1:])])
        out["pnl_spot"] = lots_prev * d_spot * multiplier
        out["pnl_basis"] = (lo_prev * d_b_out * multiplier
                            + li_prev * d_b_in * multiplier)
        resid = np.abs(out["pnl_spot"] + out["pnl_basis"] - out["pnl"]).max()
        if resid > 1e-6 * max(1.0, notional):
            raise ValueError(f"损益分解不自洽，最大残差 {resid:.6g}")

    return HedgeResult(daily=out, notional=notional, multiplier=multiplier,
                       notional_mode=notional_mode,
                       sleeves=(None if single else tuple(cols)))


def constant_delta(dates, level: float = 1.0) -> pd.Series:
    """常数敞口的 delta 序列。用于基准对照与自检（如满仓多头 = 纯持有期货）。"""
    return pd.Series(float(level), index=pd.DatetimeIndex(dates), name="delta")


if __name__ == "__main__":
    try:
        from .data_loader import load_market_data
        from .futures_chain import FuturesChain
    except ImportError:
        from data_loader import load_market_data
        from futures_chain import FuturesChain

    md = load_market_data()
    chain = FuturesChain(md)
    sched = chain.build_schedule("当月")

    # ---- 自检1：满仓多头 vs 手工计算的期货持有收益（免手续费，纯记账口径） ----
    win = md.trading_days[(md.trading_days >= "2021-01-01") &
                          (md.trading_days <= "2021-12-31")]
    res = run_hedge(constant_delta(win, 1.0), sched, spot=md.index["close"],
                    round_lots=False, close_at_end=False, fee_rate=0.0)
    d = res.daily
    manual = ((d["settle_exit"] - d["settle"].shift(1)) / d["settle"].shift(1)).fillna(0.0)
    print("[自检1] 满仓多头（不取整、不平仓）")
    print(f"  引擎净值 {d['nav'].iloc[-1]:.6f}   手工复利 {(1+manual).prod():.6f}"
          f"   （名义固定 => 两者应接近但不相等）")
    print(f"  损益分解: 方向性 {d['pnl_spot'].sum()/1e8:+.4f} + "
          f"基差 {d['pnl_basis'].sum()/1e8:+.4f} = {d['pnl'].sum()/1e8:+.4f}（净值口径）")

    # ---- 自检2：换月日的记账 ------------------------------------------------
    roll_days = d.index[d["is_roll"]]
    r = roll_days[3]
    i = d.index.get_loc(r)
    print(f"\n[自检2] 换月日 {r.date()} 前后（旧 {d['code'].iloc[i-1]} -> 新 {d['code'].iloc[i]}）")
    print(d.iloc[i - 1: i + 2][["code", "settle", "settle_exit", "lots", "pnl"]].to_string())
    print(f"  换月日损益 = {d['lots'].iloc[i-1]:.4f} × "
          f"({d['settle_exit'].iloc[i]:.1f} − {d['settle'].iloc[i-1]:.1f}) × 200 = "
          f"{d['pnl'].iloc[i]:,.0f} 元   （旧合约内部价差，未跨合约）")
    gap = d["settle"].iloc[i] - d["settle_exit"].iloc[i]
    print(f"  换月价差（新−旧）= {gap:+.1f} 点，未计入损益 —— 若误用跨合约价差，"
          f"当日将虚增 {d['lots'].iloc[i-1]*gap*200:,.0f} 元")

    # ---- 自检3：零敞口 / 取整影响 -------------------------------------------
    zero = run_hedge(constant_delta(win, 0.0), sched, round_lots=False)
    print(f"\n[自检3] 零敞口累计损益 = {zero.daily['pnl'].sum():.2e}（应为0）")
    for notional in (2e7, 1e8, 5e8):
        a = run_hedge(constant_delta(win, 0.82), sched, notional=notional,
                      round_lots=True, close_at_end=False, fee_rate=0.0).total_return
        b = run_hedge(constant_delta(win, 0.82), sched, notional=notional,
                      round_lots=False, close_at_end=False, fee_rate=0.0).total_return
        print(f"  名义 {notional/1e8:.2f}亿: 取整 {a:+.4%} vs 不取整 {b:+.4%}"
              f"   跟踪误差 {abs(a-b)*1e4:5.1f} bp")

    # ---- 自检4：汇总 --------------------------------------------------------
    print("\n[自检4] 满仓多头 2021 年汇总（免手续费口径）")
    print(res.summary().to_string())

    # ---- 自检5：手续费 ------------------------------------------------------
    # 免手续费与万五两次运行，逐日盯市损益应完全一致，净值差应恰等于累计手续费。
    fee_res = run_hedge(constant_delta(win, 1.0), sched, round_lots=False,
                        close_at_end=False, fee_rate=DEFAULT_FEE_RATE)
    d0, d1 = res.daily, fee_res.daily
    same_pnl = np.allclose(d0["pnl"], d1["pnl"])
    gap = (d0["nav"].iloc[-1] - d1["nav"].iloc[-1]) * fee_res.notional
    print(f"\n[自检5] 单边万五手续费（满仓多头 2021）")
    print(f"  盯市损益与免手续费一致: {same_pnl}")
    print(f"  净值差 {gap:,.0f} 元 vs 累计手续费 {d1['fee'].sum():,.0f} 元"
          f"（应相等，差 {abs(gap - d1['fee'].sum()):.2e}）")
    print(f"  年手续费拖累 = {d1['fee'].sum() / fee_res.notional:.4%}"
          f"（满仓多头仅换月产生成交）")

    # ---- 自检6：复利口径 ----------------------------------------------------
    # 满仓多头（delta 恒为 1、不取整、免手续费）下，复利模式的引擎净值应与
    # 手工连乘 **精确相等** —— 这是 fixed 模式做不到的（自检1 中两者只是接近）。
    comp = run_hedge(constant_delta(win, 1.0), sched, spot=md.index["close"],
                     round_lots=False, close_at_end=False, fee_rate=0.0,
                     notional_mode="daily")
    dc = comp.daily
    manual_prod = float((1 + manual).prod())
    print(f"\n[自检6] 复利口径（notional_mode='daily'，满仓多头 2021）")
    print(f"  引擎净值 {dc['nav'].iloc[-1]:.10f}   手工连乘 {manual_prod:.10f}"
          f"   差 {abs(dc['nav'].iloc[-1] - manual_prod):.2e}（应为机器精度）")
    print(f"  对照 fixed 口径 {d['nav'].iloc[-1]:.10f}"
          f"   算术-复利差 {d['nav'].iloc[-1] - manual_prod:+.4%}")
    print(f"  名义基数 首日 {dc['notional_t'].iloc[0]/1e8:.4f}亿 -> "
          f"末日 {dc['notional_t'].iloc[-1]/1e8:.4f}亿（随净值缩放）")
    # 名义规模无关性在复利下同样成立
    for notional in (2e7, 5e8):
        alt = run_hedge(constant_delta(win, 1.0), sched, notional=notional,
                        round_lots=False, close_at_end=False, fee_rate=0.0,
                        notional_mode="daily").total_return
        print(f"  名义 {notional/1e8:.2f}亿 总收益 {alt:+.6%}"
              f"   与1亿之差 {abs(alt - comp.total_return)*1e4:.2e}bp")

    # ---- 自检7：每腿口径 ----------------------------------------------------
    kw = dict(round_lots=False, close_at_end=False, fee_rate=0.0)
    q = win[win.to_series().groupby(win.to_period('Q')).transform('min') == win]
    print(f"\n[自检7] 每腿口径（notional_mode='leg'）")

    # (a) 重置点只有首日 => 整段只有一个窗口，精确退化为 fixed
    leg_once = run_hedge(constant_delta(win, 1.0), sched,
                         notional_mode="leg", reset_dates=win[:1], **kw)
    print(f"  仅首日重置 {leg_once.daily['nav'].iloc[-1]:.10f} vs fixed "
          f"{d['nav'].iloc[-1]:.10f}   差 "
          f"{abs(leg_once.daily['nav'].iloc[-1] - d['nav'].iloc[-1]):.2e}"
          f"（应为 0：单窗口即纯算术累计）")

    # (b) 分季度重置：验证"窗口内算术、窗口间连乘"精确还原净值
    lg = run_hedge(constant_delta(win, 1.0), sched, notional_mode="leg",
                   reset_dates=q, **kw)
    dl = lg.daily
    y = (dl["pnl"] - dl["fee"]) / lg.pnl_base          # 各自基数上的收益
    seg = pd.Series(dl.index, index=dl.index).where(dl.index.isin(q)).ffill()
    factors = 1.0 + y.groupby(seg).sum()
    print(f"  分季重置（{len(q)} 窗口）窗口收益连乘 {factors.prod():.10f}"
          f" vs 引擎净值 {dl['nav'].iloc[-1]:.10f}"
          f"   差 {abs(factors.prod() - dl['nav'].iloc[-1]):.2e}（应为 0）")

    # (c) 重置日前若空仓，则"前一日收盘净值"== "当日盯市净值"，两种基准等价
    #     —— 这正是回测的实际情形（腿间有 reissue_lag_days 个空仓交易日）。
    #     构造：每个季度首日之前 3 个交易日敞口置 0。
    gap_delta = constant_delta(win, 1.0)
    for t in q[1:]:
        i = win.get_loc(t)
        gap_delta.iloc[max(0, i - 3):i] = 0.0
    lg2 = run_hedge(gap_delta, sched, notional_mode="leg", reset_dates=q, **kw)
    d2 = lg2.daily
    nav_prev2 = d2["nav"].shift(1).fillna(1.0)
    nav_mtm2 = nav_prev2 + d2["pnl"] / lg2.notional
    at_reset = d2.index.isin(q[1:])
    print(f"  重置日前空仓时 |前一日收盘 − 当日盯市| 最大 "
          f"{float((nav_prev2 - nav_mtm2)[at_reset].abs().max()):.2e}"
          f"（应为 0：空仓期无损益，两种基准重合）")

    # (d) leg 介于 fixed 与 daily 之间（重置越密越接近 daily，但不会相等：
    #     leg 取前一日收盘、daily 取当日盯市，差一天的损益）
    leg_all = run_hedge(constant_delta(win, 1.0), sched,
                        notional_mode="leg", reset_dates=win, **kw)
    print(f"  逐日重置 {leg_all.daily['nav'].iloc[-1]:.10f} vs daily "
          f"{dc['nav'].iloc[-1]:.10f}（接近但不相等，基数差一天）；"
          f"fixed {d['nav'].iloc[-1]:.10f} < 分季 {dl['nav'].iloc[-1]:.10f}")

    # ---- 自检8：多套筒 ------------------------------------------------------
    print("\n[自检8] 多套筒")
    common = ["delta", "notional_t", "lots", "pnl", "fee", "nav", "traded_lots"]

    # (a) 单列 DataFrame（含 leg 重置）应与 Series 输入完全一致
    m1 = run_hedge(gap_delta.to_frame("s0"), sched, notional_mode="leg",
                   reset_dates={"s0": q}, **kw)
    diff_a = max(float((m1.daily[c] - lg2.daily[c]).abs().max()) for c in common)
    print(f"  (a) 单列 DataFrame vs Series（leg+空仓）最大列差 {diff_a:.2e}（应为 0）")

    # (b) fixed 口径：两套筒等权 ≡ 平均后的单套筒（线性可加）
    d1, d2 = constant_delta(win, 1.0), constant_delta(win, 0.4)
    mm = run_hedge(pd.DataFrame({"a": d1, "b": d2}), sched, **kw)
    ms = run_hedge((d1 + d2) / 2.0, sched, **kw)
    diff_b = float((mm.daily["nav"] - ms.daily["nav"]).abs().max())
    print(f"  (b) 两套筒(1.0, 0.4) vs 单套筒0.7 净值最大差 {diff_b:.2e}（应为机器精度）")

    # (c) 净额计费：两套筒反相调仓（0.8/1.0 与 1.0/0.8 交替，聚合恒为 0.9），
    #     合并账户除首日与换月外零成交；分账户各自天天成交
    alt = np.where(np.arange(len(win)) % 2 == 0, 0.8, 1.0)
    da = pd.Series(alt, index=win)
    db = pd.Series(1.8 - alt, index=win)
    fkw = dict(round_lots=False, close_at_end=False, fee_rate=DEFAULT_FEE_RATE)
    f_net = run_hedge(pd.DataFrame({"a": da, "b": db}), sched, **fkw).daily["fee"].sum()
    f_sep = (run_hedge(da, sched, notional=0.5e8, **fkw).daily["fee"].sum()
             + run_hedge(db, sched, notional=0.5e8, **fkw).daily["fee"].sum())
    print(f"  (c) 反相套筒手续费: 合并净额 {f_net:,.0f} 元 < 分账户 {f_sep:,.0f} 元"
          f"（节省 {1 - f_net / f_sep:.1%}）")

    # (d) 多套筒 leg：套筒分列的 pnl/fee 之和恒等于账户总额
    resets_ab = {"a": q, "b": q[::2]}
    ml = run_hedge(pd.DataFrame({"a": gap_delta, "b": constant_delta(win, 0.5)}),
                   sched, notional_mode="leg", reset_dates=resets_ab, **kw)
    dml = ml.daily
    r1 = float((dml["pnl_a"] + dml["pnl_b"] - dml["pnl"]).abs().max())
    r2 = float((dml["fee_a"] + dml["fee_b"] - dml["fee"]).abs().max())
    r3 = float((dml["base_a"] + dml["base_b"] - dml["notional_t"]).abs().max())
    print(f"  (d) 套筒列加总恒等: pnl {r1:.2e} / fee {r2:.2e} / base {r3:.2e}"
          f"（均应为机器精度）")

    # ---- 自检9：多天展期 ----------------------------------------------------
    print("\n[自检9] 多天展期（roll_window_days=5，满仓多头 2021~2022）")
    sched5 = chain.build_schedule("当月", roll_window_days=5)
    h1 = run_hedge(constant_delta(win, 1.0), sched, spot=md.index["close"],
                   close_at_end=False, fee_rate=DEFAULT_FEE_RATE)
    h5 = run_hedge(constant_delta(win, 1.0), sched5, spot=md.index["close"],
                   close_at_end=False, fee_rate=DEFAULT_FEE_RATE)
    d1_, d5_ = h1.daily, h5.daily
    mixed = d5_[(d5_["lots_out"].abs() > 1e-9) & (d5_["lots_in"].abs() > 1e-9)]
    print(f"  两合约并存的交易日: {len(mixed)} 天"
          f"（窗口日应 ≈ 换月次数 {int(d5_['is_roll'].sum())} × 4）")
    print(f"  窗口内切片权重和恒为 1: "
          f"{bool(np.allclose(d5_['lots_out'] + d5_['lots_in'], d5_['lots']))}")
    print(f"  总成交手数: 单日换月 {d1_['traded_lots'].sum():,.0f} vs "
          f"多天展期 {d5_['traded_lots'].sum():,.0f}"
          f"（应几乎相同——展期总量只是被分摊）")
    print(f"  累计手续费: {d1_['fee'].sum():,.0f} vs {d5_['fee'].sum():,.0f} 元")
    print(f"  期末净值: {d1_['nav'].iloc[-1]:.6f} vs {d5_['nav'].iloc[-1]:.6f}"
          f"   差 {(d5_['nav'].iloc[-1] - d1_['nav'].iloc[-1]) * 1e4:+.1f}bp"
          f"（基差时点效应，应为小量）")
