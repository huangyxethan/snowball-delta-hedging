# -*- coding: utf-8 -*-
"""策略对比：挑选若干已入档策略，汇总关键指标 Excel + 时序对比图。

    python scripts/compare_exec.py twap_1h twap_power_p1 ...
    python scripts/compare_exec.py --dir output/exec_runs/<cfg_hash> ...

产出（固定名，重跑覆盖）：
    <cfg_dir>/comparisons/comparison.xlsx    每策略一行的指标汇总
    <cfg_dir>/comparisons/comparison.png     五联时序图（纵排）：
        净值走势 / 相对指数累计超额 / 数量项累计 / 价格项累计 / 费差项累计

指标口径与档案一致（年化 = 合计 / (天数/252)）；"单分钟峰值占当日成交(%)"
为逐日 [峰值分钟成交 / 当日总成交] 的全周期中位数（同一分钟跨合约求和；
尺度无关，刻画执行的时间集中度），自 orders/<策略名>.parquet 计算，
无订单流档案的策略该列为 NaN（用新框架重跑一次即可生成）。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                   "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

TRADING_DAYS = 252.0
OUT_ROOT = ROOT / "output" / "exec_runs"


def _default_cfg_dir() -> Path:
    """output/exec_runs/ 下唯一的配置目录；多于一个时须用 --dir 指定。"""
    dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir()]
    if len(dirs) != 1:
        raise SystemExit(f"exec_runs 下有 {len(dirs)} 个配置目录，"
                         f"请用 --dir 指定: {[d.name for d in dirs]}")
    return dirs[0]


def _minute_peak_share(cfg_dir: Path, name: str) -> float:
    """单分钟峰值占当日成交的比例（%，逐日计算后取全周期中位数）。

    峰值分钟 = 当日 Σ|lots| 最大的分钟（跨合约求和）；比例的分子分母
    均为策略自身成交量，与名义本金无关。等权一小时 TWAP 恒为 1/60。
    """
    path = cfg_dir / "orders" / f"{name}.parquet"
    if not path.exists():
        print(f"  [警告] {name} 无订单流档案（orders/），"
              f"单分钟峰值占比为 NaN——用新框架重跑该策略即可生成")
        return np.nan
    o = pd.read_parquet(path)
    by_min = o.assign(q=o["lots"].abs()).groupby(["date", "time"])["q"].sum()
    daily_peak = by_min.groupby("date").max()
    daily_tot = by_min.groupby("date").sum()
    share = (daily_peak / daily_tot).dropna()
    return float(share.median() * 100)


def compare(names: list, cfg_dir: Path) -> tuple:
    """读档汇总。Returns (汇总表, {策略名: 逐日 cache})。"""
    rows, caches = [], {}
    for name in names:
        path = cfg_dir / "caches" / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"缓存不存在: {path}（策略未入档？）")
        d = pd.read_parquet(path)
        caches[name] = d
        years = len(d) / TRADING_DAYS
        te = d["te_qty"] + d["te_price"] + d["te_fee"]
        te_cum = float(te.sum())
        te_ann = float(te.std()) * np.sqrt(TRADING_DAYS) * 100
        ann = te_cum / years * 100
        rows.append({
            "策略": name,
            "期末净值": float(d["nav"].iloc[-1]),
            "累计净值差": te_cum,
            "年化差(pp)": ann,
            "年化跟踪误差(pp)": te_ann,
            "信息比率": ann / te_ann if te_ann else np.nan,
            "数量项年化(pp)": float(d["te_qty"].sum()) / years * 100,
            "价格项年化(pp)": float(d["te_price"].sum()) / years * 100,
            "费差项年化(pp)": float(d["te_fee"].sum()) / years * 100,
            "单分钟峰值占当日成交(%)": _minute_peak_share(cfg_dir, name),
        })
    return pd.DataFrame(rows).set_index("策略"), caches


def plot(caches: dict, path: Path, period_str: str) -> None:
    """五联时序图（共享横轴，纵排）。分项为累计口径（百分点）。"""
    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})
    first = next(iter(caches.values()))
    axes[0].plot(first.index, first["nav_index"], "k-", lw=1.2, label="指数")
    for name, d in caches.items():
        axes[0].plot(d.index, d["nav"], lw=1.0, label=name)
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"净值走势{period_str}")

    panels = [(None, "相对指数累计超额（百分点）"),
              ("te_qty", "数量项累计（百分点）"),
              ("te_price", "价格项累计（百分点）"),
              ("te_fee", "费差项累计（百分点）")]
    for ax, (col, title) in zip(axes[1:], panels):
        for name, d in caches.items():
            y = (d["te_qty"] + d["te_price"] + d["te_fee"]
                 if col is None else d[col]).cumsum() * 100
            ax.plot(d.index, y, lw=1.0, label=name)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="已入档执行策略的对比汇总")
    ap.add_argument("names", nargs="+", help="档案策略名（caches/ 下的文件名）")
    ap.add_argument("--dir", default=None,
                    help="配置目录（默认自动取 exec_runs 下唯一目录）")
    args = ap.parse_args()

    cfg_dir = Path(args.dir) if args.dir else _default_cfg_dir()
    out_dir = cfg_dir / "comparisons"
    out_dir.mkdir(exist_ok=True)

    summ, caches = compare(args.names, cfg_dir)
    first = next(iter(caches.values()))
    period_str = (f"（{first.index[0].date()} ~ {first.index[-1].date()}）")

    summ.to_excel(out_dir / "comparison.xlsx")
    plot(caches, out_dir / "comparison.png", period_str)

    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(f"\n产出: {out_dir / 'comparison.xlsx'}\n      "
          f"{out_dir / 'comparison.png'}")
