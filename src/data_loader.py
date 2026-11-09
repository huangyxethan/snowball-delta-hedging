# -*- coding: utf-8 -*-
"""
data_loader.py —— 第0层：原始数据读取与清洗

职责（也是边界）：
    1. 读取 data/raw/ 下的三份 Excel（中证500指数、IC全合约日线、Shibor 1年；
       字段约定见 README「Data」一节，scripts/make_synthetic_data.py 可生成同
       格式的合成数据）
    2. 清洗：去脚注行、按日期升序、单位换算
    3. 校验：数据必须满足的不变量（每日恰好4个IC合约、结算价无缺失、日历对齐）
    4. 打包成一个 MarketData 对象供后续所有层使用

本模块之外的任何代码都不应该直接读 raw 文件。
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 项目根目录 = 本文件所在目录(src/)的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

INDEX_FILE = RAW_DIR / "000905.SH.xlsx"
IC_FILE = RAW_DIR / "IC_daily_price.xlsx"
SHIBOR_FILE = RAW_DIR / "shibor.xlsx"


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketData:
    """清洗后的全部市场数据。

    Attributes
    ----------
    index : pd.DataFrame
        中证500指数日线，DatetimeIndex（升序），列: open/high/low/close。
        起点早于 IC 上市日约两年，专门留给历史波动率预热，不要截断。
    ic : pd.DataFrame
        IC 全合约日线（长表），列: code/date/open/high/low/close/
        presettle/settle/volume/amount/oi，按 (date, code) 升序。
    shibor : pd.Series
        Shibor 1年（小数，如 0.0148），DatetimeIndex（升序）。
    trading_days : pd.DatetimeIndex
        指数与 IC 共同的交易日历（升序），回测主循环只在这些日期上走。
    """

    index: pd.DataFrame
    ic: pd.DataFrame
    shibor: pd.Series
    trading_days: pd.DatetimeIndex

    # -- 常用取数快捷方式 ---------------------------------------------------

    def spot(self, date) -> float:
        """指定日期的指数收盘价（用于 delta 计算与敲入敲出判定）。"""
        return float(self.index.loc[pd.Timestamp(date), "close"])

    def rf(self, date) -> float:
        """指定日期的无风险利率 r（Shibor 1年，小数）。"""
        return float(self.shibor.loc[pd.Timestamp(date)])

    def ic_on(self, date) -> pd.DataFrame:
        """指定日期的 IC 截面（恰好4行：当月/次月/当季/下季）。"""
        return self.ic[self.ic["date"] == pd.Timestamp(date)]


# ---------------------------------------------------------------------------
# 单文件读取
# ---------------------------------------------------------------------------

def load_index(path: Path = INDEX_FILE) -> pd.DataFrame:
    """读中证500指数日线 -> DatetimeIndex 升序的 DataFrame。"""
    df = pd.read_excel(path)
    # 数据终端导出的文件末尾常带"数据来源"脚注行，time 列为 NaT，借此剔除
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = (
        df.rename(columns={"time": "date"})
        .set_index("date")[["open", "high", "low", "close"]]
        .astype(float)
        .sort_index()  # 原始文件是最新日期在前，统一转升序
    )
    return df


def load_shibor(path: Path = SHIBOR_FILE) -> pd.Series:
    """读 Shibor 1年 -> DatetimeIndex 升序的 Series（已除以100转小数）。"""
    df = pd.read_excel(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    s = (
        df.rename(columns={"time": "date"})
        .set_index("date")["value"]
        .astype(float)
        .sort_index()
        / 100.0  # 原始单位是百分数（1.48 表示 1.48%）
    )
    s.name = "shibor_1y"
    return s


def load_ic(path: Path = IC_FILE) -> pd.DataFrame:
    """读 IC 全合约日线 -> 按 (date, code) 升序的长表。"""
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day"]).rename(columns={"day": "date"})
    # code 形如 "IC2109.CFE"，统一去掉交易所后缀，后续按 "IC2109" 引用
    df["code"] = df["code"].str.replace(".CFE", "", regex=False)
    num_cols = ["open", "high", "low", "close", "change",
                "volume", "amount", "presettle", "settle", "oi"]
    df[num_cols] = df[num_cols].astype(float)
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 校验：数据必须满足的不变量
# ---------------------------------------------------------------------------

def _validate(index: pd.DataFrame, ic: pd.DataFrame, shibor: pd.Series,
              trading_days: pd.DatetimeIndex) -> None:
    """任何一条不满足都说明数据被动过/换过，直接报错而不是带病运行。"""
    # 1) 每个交易日恰好4个IC合约（当月/次月/当季/下季）
    per_day = ic.groupby("date")["code"].nunique()
    bad = per_day[per_day != 4]
    if len(bad):
        raise ValueError(f"以下日期IC合约数不是4个: {bad.head().to_dict()}")

    # 2) 结算价、持仓量无缺失（结算价用于损益记账，缺失不可容忍）
    for col in ["settle", "presettle", "close", "oi"]:
        n = ic[col].isna().sum()
        if n:
            raise ValueError(f"IC数据 {col} 列存在 {n} 个缺失值")

    # 3) 无重复记录
    if ic.duplicated(subset=["date", "code"]).any():
        raise ValueError("IC数据存在重复的 (date, code) 记录")
    if not index.index.is_unique:
        raise ValueError("指数数据存在重复日期")

    # 4) 共同日历上三份数据都要有值
    if not trading_days.isin(index.index).all():
        raise ValueError("存在指数缺失的共同交易日")
    missing_sh = ~trading_days.isin(shibor.index)
    if missing_sh.any():
        raise ValueError(f"存在Shibor缺失的共同交易日: {trading_days[missing_sh][:5]}")

    # 5) 指数收盘价为正
    if (index["close"] <= 0).any():
        raise ValueError("指数收盘价存在非正值")


# ---------------------------------------------------------------------------
# 对外主入口
# ---------------------------------------------------------------------------

def load_market_data() -> MarketData:
    """一站式加载：读取三份数据、构建共同日历、校验、打包。

    用法（notebook 中）::

        from src.data_loader import load_market_data
        md = load_market_data()
        md.spot("2021-06-15")     # 指数收盘价
        md.rf("2021-06-15")       # 无风险利率
        md.ic_on("2021-06-15")    # 当日IC截面（4行）
    """
    index = load_index()
    ic = load_ic()
    shibor = load_shibor()

    # 共同交易日历 = 指数与IC都有数据的日期（IC上市晚、末端早，故为瓶颈）
    trading_days = index.index.intersection(ic["date"].unique()).sort_values()

    _validate(index, ic, shibor, trading_days)
    return MarketData(index=index, ic=ic, shibor=shibor, trading_days=trading_days)


if __name__ == "__main__":
    # 直接运行本文件时做一次冒烟测试
    md = load_market_data()
    print(f"指数:   {md.index.index[0].date()} -> {md.index.index[-1].date()}, {len(md.index)} 天")
    print(f"IC:     {md.ic['date'].min().date()} -> {md.ic['date'].max().date()}, "
          f"{md.ic['code'].nunique()} 个合约, {len(md.ic)} 行")
    print(f"Shibor: {md.shibor.index[0].date()} -> {md.shibor.index[-1].date()}, 最新 {md.shibor.iloc[-1]:.4%}")
    print(f"共同交易日: {len(md.trading_days)} 天 "
          f"({md.trading_days[0].date()} -> {md.trading_days[-1].date()})")
    d = md.trading_days[len(md.trading_days) // 2]
    print(f"\n抽查 {d.date()}: spot={md.spot(d):.2f}, r={md.rf(d):.4%}")
    print(md.ic_on(d)[["code", "close", "settle", "oi"]].to_string(index=False))
