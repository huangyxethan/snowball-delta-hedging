# -*- coding: utf-8 -*-
"""
batch.py —— 第5层附件：批量执行回测的输出与档案管理

notebook（execution_report.ipynb）承担汇报叙事；批量策略迭代走本模块：
一次进程内跑完整个策略清单（run_execution 本就多策略共享曲线与行情），
每个策略落三件套，每次运行落一个自描述目录，跨运行维护一张累积总表。

目录结构（目录 = 指数配置，策略在其下积累）
--------------------------------------------
    output/exec_runs/
        leaderboard.csv                  跨运行累积总表：每次运行每策略追加
                                         一行（run 列 = 时间戳_tag），是完整
                                         历史；不同 cfg_hash 的行属于不同
                                         参照系，不可直接比较
        <cfg_hash>/                      一个指数配置一个目录，同配置的
                                         策略无论分几次运行都落在这里
            manifest.json                配置全文 + label + 最近一次运行的
                                         策略清单、区间与耗时
            summary.csv / summary.txt    该配置下**全部**策略的最新结果
                                         对比表（每次运行后由 leaderboard
                                         重建，同名策略取最近一行）
            plots/<策略名>.png           三联图：净值对比 / 累计净值差 /
                                         净值差三项分解
            reports/<策略名>.txt         单策略摘要（头部信息 + 十项统计）
            caches/<策略名>.parquet      逐日明细（字段见 _write_daily）
            orders/<策略名>.parquet      分钟订单流：逐笔成交的时刻、合约、
                                         手数、成交价与当日结算价
                                         （字段见 _write_orders）
            sweeps/<时间戳>_<tag>.csv    参数扫描汇总表（run_sweep 产出：
                                         行=策略、列=十项统计；不落三件套、
                                         不进 leaderboard——扫描属探索，
                                         选中的参数以正式名字经 run_batch
                                         入档）

    同名策略重跑即覆盖全部产物（保留最新版，历史结果在 leaderboard 中
    可查）；参数变体请用不同名字（如 twap_hour_pace_1450）。策略名约定
    用英文（进文件名与 leaderboard），其余文本为中文。

统计口径
--------
    年化差 / 跟踪误差 / 三项分解：与 ExecutionResult.summary() 同源，
        te 逐日差 = te_qty + te_price + te_fee（构造上精确）；
    信息比率 = 年化差 / 年化跟踪误差；
    夏普比率 = 日收益均值 / 日收益标准差 x sqrt(252)，无风险利率取 0，
        日收益 = nav 一阶差分（净值为固定名义上的算术序列）。

指数配置识别
------------
    cfg_hash = BacktestConfig repr 的 sha1 前 8 位。frozen dataclass 的
    repr 含全部字段名与取值，任何字段变动即变号；manifest 存配置全文，
    hash 只作索引。

用法
----
    from batch import run_batch
    res, run_dir = run_batch(strategies, cfg_index, tag="hour-pace")

批量入口见 scripts/batch_exec.py（策略清单写在脚本内，改完即跑）。
"""

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 脚本运行不弹窗
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                   "WenQuanYi Micro Hei"]   # Windows / Linux 回退链
plt.rcParams["axes.unicode_minus"] = False

try:
    from .backtest import BacktestConfig, run_backtest
    from .execution import ExecutionConfig, run_execution
except ImportError:
    from backtest import BacktestConfig, run_backtest
    from execution import ExecutionConfig, run_execution

OUT_ROOT = Path(__file__).resolve().parent.parent / "output" / "exec_runs"

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 标识
# ---------------------------------------------------------------------------

def cfg_hash(cfg: BacktestConfig) -> str:
    """指数配置的短标识：repr 含全部字段，任何字段变动即变号。

    后加的字段取默认值时从 repr 中剔除，使既有档案的哈希（如 fe4eaa78）
    在增加字段后仍可复现。"""
    r = repr(cfg)
    if cfg.ko_obs_weekday == 4:
        r = r.replace(", ko_obs_weekday=4", "")
    return hashlib.sha1(r.encode()).hexdigest()[:8]


def _strategy_params(s) -> dict:
    """策略实例的构造参数（公开属性即全部参数，manifest 记录用；
    下划线开头的运行期缓存不入档）。"""
    return {k: v for k, v in vars(s).items() if not k.startswith("_")}


def _safe_name(name: str) -> str:
    """策略名转目录名（去除路径非法字符）。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def _sharpe(nav: pd.Series) -> float:
    r = nav.diff().dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def strategy_stats(res, name: str) -> dict:
    """单策略的十项统计（report.txt 与 leaderboard 的行）。"""
    d = res.daily[name]
    years = len(res.dates) / TRADING_DAYS
    te = d["te_qty"] + d["te_price"] + d["te_fee"]
    notional_traded = (d["traded_lots"] * d["settle_px"]).sum()
    slip_bp = (d["slip_cost"].sum()
               / (notional_traded * res.bt.hedge.multiplier) * 1e4
               if notional_traded else np.nan)
    ann_diff = te.sum() / years * 100
    ann_te = te.std() * np.sqrt(TRADING_DAYS) * 100
    return {
        "年化差(pp)": ann_diff,
        "年化跟踪误差(pp)": ann_te,
        "信息比率": ann_diff / ann_te if ann_te else np.nan,
        "数量项年化(pp)": d["te_qty"].sum() / years * 100,
        "价格项年化(pp)": d["te_price"].sum() / years * 100,
        "费差项年化(pp)": d["te_fee"].sum() / years * 100,
        "加权滑点(bp)": slip_bp,
        "日均成交(手)": float(d["traded_lots"].mean()),
        "日均持仓缺口(手)": float((d["lots_eod"] - d["lots_index"])
                                  .abs().mean()),
        "夏普比率": _sharpe(d["nav"]),
    }


def batch_summary(res) -> pd.DataFrame:
    """全部策略的统计表（每策略一行）。"""
    return pd.DataFrame({n: strategy_stats(res, n)
                         for n in res.daily}).T


# ---------------------------------------------------------------------------
# 三件套
# ---------------------------------------------------------------------------

def _write_plot(res, name: str, path: Path, title_extra: str = "") -> None:
    """三联图：净值对比 / 相对指数累计净值差 / 净值差三项分解。"""
    d = res.daily[name]
    nav_idx = res.nav_index
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    axes[0].plot(nav_idx.index, nav_idx, "k-", lw=1.2, label="指数")
    axes[0].plot(d.index, d["nav"], lw=1.0, label=name)
    axes[0].legend()
    axes[0].set_title(f"净值：指数与 {name}{title_extra}")

    te_cum = (d["te_qty"] + d["te_price"] + d["te_fee"]).cumsum() * 100
    axes[1].plot(d.index, te_cum, lw=1.0)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("相对指数的累计净值差（百分点）")

    for col, lb in [("te_qty", "数量项"), ("te_price", "价格项"),
                    ("te_fee", "费差项")]:
        axes[2].plot(d.index, d[col].cumsum() * 100, lw=1.0, label=lb)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].legend()
    axes[2].set_title("累计净值差分解（百分点）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(res, name: str, stats: dict, path: Path,
                  meta: dict) -> None:
    """单策略摘要：头部信息 + 十项统计。"""
    d = res.daily[name]
    params = next(s["params"] for s in meta["last_run"]["strategies"]
                  if s["params"].get("name") == name)
    lines = [
        f"策略        {name}",
        f"策略参数    {json.dumps(params, ensure_ascii=False)}",
        f"回测区间    {res.dates[0].date()} ~ {res.dates[-1].date()}"
        f"（{len(res.dates)} 个交易日，{len(res.bt.legs)} 只雪球）",
        f"指数配置    cfg_hash={meta['cfg_hash']}  "
        f"（配置全文见 manifest.json）",
        f"期末净值    策略 {float(d['nav'].iloc[-1]):.4f}   "
        f"指数 {float(res.nav_index.iloc[-1]):.4f}   "
        f"累计净值差 {float((d['te_qty'] + d['te_price'] + d['te_fee']).sum()):+.4f}",
        f"指数夏普    {_sharpe(res.nav_index):.3f}（对照；无风险利率取 0）",
        f"生成时间    {meta['last_run']['generated']}",
        "-" * 46,
    ]
    for k, v in stats.items():
        lines.append(f"{k:<14}{v:>12.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_daily(res, name: str, path: Path) -> None:
    """逐日明细。字段 = 执行框架的全部逐日输出 + 指数净值 + 累计净值差：

    nav / nav_index          策略与指数净值
    te_qty / te_price / te_fee / te_cum
                             净值差三项分解（当日）与累计净值差
    pnl / fee                当日损益与手续费（元）
    traded_lots / slip_cost  当日成交手数与滑点成本（元，正=跑输结算价）
    lots_eod / lots_index    日终持仓与指数目标持仓（手）
    settle_px                目标合约结算价
    patched / is_roll        分钟数据补丁日 / 换月窗口日标记
    lots_a / lots_b 等       策略 plan_day 返回的诊断列（随策略而异）
    """
    d = res.daily[name].copy()
    d["nav_index"] = res.nav_index
    d["te_cum"] = (d["te_qty"] + d["te_price"] + d["te_fee"]).cumsum()
    d.to_parquet(path)


def _write_orders(res, name: str, path: Path) -> None:
    """分钟订单流（逐笔成交明细）。

    列：date / time / code / lots / fill_px / settle_px
    time 为分钟 bar 结束标记（如 143000），Int64，<NA> = 按结算价成交；
    lots 带方向；settle_px 为该合约当日结算价（滑点、手续费均可由此重算）。
    """
    res.orders[name].to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_batch(strategies: list, cfg_index: BacktestConfig = None,
              exec_cfg: ExecutionConfig = None, tag: str = None,
              out_root=None, start=None, end=None, verbose: bool = True):
    """跑一批执行策略并落盘三件套。

    Parameters
    ----------
    strategies : ExecutionStrategy 实例列表（一次遍历共享曲线与行情）
    cfg_index  : 指数配置（None = BacktestConfig 默认值）
    exec_cfg   : 执行配置（None = ExecutionConfig 默认值）
    tag        : 运行目录后缀（默认 "run"）
    out_root   : 档案根目录（默认 output/exec_runs/）
    start/end  : 子区间运行（诊断用；净值从 1 起算，与指数水平不可比，
                 te 分解仍精确——见 run_execution docstring）

    Returns
    -------
    (ExecutionResult, Path)  执行结果与该配置的档案目录
    """
    cfg_index = cfg_index or BacktestConfig()
    root = Path(out_root) if out_root else OUT_ROOT
    ch = cfg_hash(cfg_index)
    cfg_dir = root / ch
    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{tag or 'run'}"

    tic = time.perf_counter()
    bt = run_backtest(cfg_index)
    t_bt = time.perf_counter() - tic
    tic = time.perf_counter()
    res = run_execution(bt, strategies, exec_cfg, start=start, end=end)
    t_exec = time.perf_counter() - tic

    meta = {
        "cfg_hash": ch,
        "label": cfg_index.label(),
        "cfg": asdict(cfg_index),
        "last_run": {
            "run_id": run_id,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exec_cfg": asdict(res.cfg),
            "strategies": [{"class": type(s).__name__,
                            "params": _strategy_params(s)}
                           for s in strategies],
            "period": [str(res.dates[0].date()), str(res.dates[-1].date())],
            "n_days": int(len(res.dates)),
            "n_legs": int(len(bt.legs)),
            "nav_index_end": float(res.nav_index.iloc[-1]),
            "backtest_seconds": round(t_bt, 1),
            "execution_seconds": round(t_exec, 1),
        },
    }
    (cfg_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    period = meta["last_run"]["period"]
    period_str = f"（{period[0]} ~ {period[1]}）"
    for sub in ("plots", "reports", "caches", "orders"):
        (cfg_dir / sub).mkdir(exist_ok=True)
    rows = []
    for s in strategies:
        name = s.name
        fname = _safe_name(name)
        stats = strategy_stats(res, name)
        _write_plot(res, name, cfg_dir / "plots" / f"{fname}.png",
                    title_extra=period_str)
        _write_report(res, name, stats,
                      cfg_dir / "reports" / f"{fname}.txt", meta)
        _write_daily(res, name, cfg_dir / "caches" / f"{fname}.parquet")
        _write_orders(res, name, cfg_dir / "orders" / f"{fname}.parquet")
        rows.append({"run": run_id, "cfg_hash": ch,
                     "策略": name, "class": type(s).__name__,
                     "params": json.dumps(_strategy_params(s),
                                          ensure_ascii=False),
                     "起始": period[0], "结束": period[1],
                     **stats})

    # 跨运行累积总表（完整历史；cfg_hash 不同的行不可直接比较）
    lb_path = root / "leaderboard.csv"
    lb = pd.DataFrame(rows)
    if lb_path.exists():
        lb = pd.concat([pd.read_csv(lb_path, encoding="utf-8-sig"), lb],
                       ignore_index=True)
    lb.to_csv(lb_path, index=False, encoding="utf-8-sig")

    # 该配置下全部策略的最新结果对比表（同名策略取最近一行）
    sub = (lb[lb["cfg_hash"] == ch]
           .drop_duplicates("策略", keep="last")
           .set_index("策略")
           .drop(columns=["cfg_hash", "class", "params", "起始", "结束"]))
    sub.to_csv(cfg_dir / "summary.csv", encoding="utf-8-sig")
    summary_txt = sub.round(3).to_string()
    (cfg_dir / "summary.txt").write_text(summary_txt + "\n", encoding="utf-8")

    if verbose:
        print(f"回测 {t_bt:.0f}s + 执行 {t_exec:.0f}s，"
              f"{period[0]} ~ {period[1]}，{meta['last_run']['n_legs']} 腿，"
              f"cfg_hash={ch}")
        print(f"本次运行: {[s.name for s in strategies]}")
        print(f"该配置下全部策略（最新结果）:\n{summary_txt}")
        print(f"档案目录: {cfg_dir}")
    return res, cfg_dir


def run_sweep(strategies: list, cfg_index: BacktestConfig = None,
              exec_cfg: ExecutionConfig = None, tag: str = None,
              out_root=None, start=None, end=None, verbose: bool = True):
    """参数扫描：只产出一张汇总表（行=策略、列=十项统计）。

    与 run_batch 的区别：不落三件套、不写 manifest、不进 leaderboard；
    表写入 <cfg_dir>/sweeps/<时间戳>_<tag>.csv 与同名 .txt。扫描属探索，
    选中的参数以正式名字经 run_batch 入档。

    Returns
    -------
    (ExecutionResult, Path)  执行结果与汇总表 csv 路径
    """
    cfg_index = cfg_index or BacktestConfig()
    root = Path(out_root) if out_root else OUT_ROOT
    sweep_dir = root / cfg_hash(cfg_index) / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{tag or 'sweep'}"

    tic = time.perf_counter()
    bt = run_backtest(cfg_index)
    t_bt = time.perf_counter() - tic
    tic = time.perf_counter()
    res = run_execution(bt, strategies, exec_cfg, start=start, end=end)
    t_exec = time.perf_counter() - tic

    summary = batch_summary(res)
    csv_path = sweep_dir / f"{run_id}.csv"
    summary.to_csv(csv_path, encoding="utf-8-sig")
    summary_txt = summary.round(3).to_string()
    (sweep_dir / f"{run_id}.txt").write_text(summary_txt + "\n",
                                             encoding="utf-8")
    if verbose:
        print(f"回测 {t_bt:.0f}s + 执行 {t_exec:.0f}s，"
              f"{res.dates[0].date()} ~ {res.dates[-1].date()}，"
              f"{len(strategies)} 个策略，cfg_hash={cfg_hash(cfg_index)}")
        print(summary_txt)
        print(f"汇总表: {csv_path}")
    return res, csv_path


# ---------------------------------------------------------------------------
# 汇总 Excel（shortlist：入选策略的对比表）
# ---------------------------------------------------------------------------

def _parse_report(path: Path) -> dict:
    """把 reports/<策略名>.txt 读回一行汇总（键与文件行一一对应）。"""
    head, stats, in_stats = {}, {}, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and set(line.strip()) == {"-"}:
            in_stats = True
            continue
        if not line.strip():
            continue
        if in_stats:
            k, v = line.rsplit(maxsplit=1)
            stats[k.strip()] = float(v)
        else:
            k, v = line.split(maxsplit=1)
            head[k] = v.strip()
    nav = re.search(r"策略 ([\d.]+)\s+指数 ([\d.]+)\s+累计净值差 ([+\-\d.]+)",
                    head.get("期末净值", ""))
    span = re.search(r"(\d{4}-\d{2}-\d{2}) ~ (\d{4}-\d{2}-\d{2})",
                     head.get("回测区间", ""))
    idx_sharpe = re.match(r"[+\-\d.]+", head.get("指数夏普", ""))
    return {"策略": head.get("策略"),
            "起始": span.group(1) if span else None,
            "结束": span.group(2) if span else None,
            "期末净值": float(nav.group(1)) if nav else None,
            "指数期末净值": float(nav.group(2)) if nav else None,
            "累计净值差": float(nav.group(3)) if nav else None,
            **stats,
            "指数夏普": float(idx_sharpe.group()) if idx_sharpe else None,
            "参数": head.get("策略参数"),
            "生成时间": head.get("生成时间")}


def update_shortlist(cfg_index: BacktestConfig = None, add: list = None,
                     remove: list = None, names: list = None,
                     out_root=None, verbose: bool = True):
    """维护该指数配置下的汇总 Excel：shortlist.xlsx（入选策略对比表）。

    名单持久化于同目录 shortlist.json（有序、去重）；每次调用先按参数
    调整名单，再按名单重读 reports/<策略名>.txt 整表重建 shortlist.xlsx，
    故重跑过的策略自动取最新结果。名单中尚无 report 的策略跳过并提示。

    Parameters
    ----------
    cfg_index : 指数配置（None = 默认值；决定 cfg_hash 目录）
    add       : 追加进名单的策略名列表（已在名单则保持原位置）
    remove    : 从名单移除的策略名列表
    names     : 整体替换名单（与 add/remove 互斥地先行生效）
    out_root  : 档案根目录（默认 output/exec_runs/）

    Returns
    -------
    (pd.DataFrame, Path)  汇总表与 xlsx 路径
    """
    cfg_index = cfg_index or BacktestConfig()
    root = Path(out_root) if out_root else OUT_ROOT
    cfg_dir = root / cfg_hash(cfg_index)
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"配置目录不存在（先跑 run_batch）：{cfg_dir}")
    members_path = cfg_dir / "shortlist.json"
    members = (json.loads(members_path.read_text(encoding="utf-8"))
               if members_path.exists() else [])
    if names is not None:
        members = list(dict.fromkeys(names))
    for n in add or []:
        if n not in members:
            members.append(n)
    members = [n for n in members if n not in set(remove or [])]

    rows = []
    for n in members:
        rp = cfg_dir / "reports" / f"{_safe_name(n)}.txt"
        if not rp.exists():
            if verbose:
                print(f"[shortlist] 跳过 {n}：无 {rp.name}（尚未回测？）")
            continue
        rows.append(_parse_report(rp))
    df = pd.DataFrame(rows)

    members_path.write_text(
        json.dumps(members, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    xlsx_path = cfg_dir / "shortlist.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df.round(3).to_excel(xw, sheet_name="执行策略汇总", index=False)
        ws = xw.sheets["执行策略汇总"]
        ws.freeze_panes = "B2"
        for col in ws.columns:
            width = max(len(str(c.value)) + 2 * sum(ord(ch) > 0x2e80
                        for ch in str(c.value)) for c in col if c.value
                        is not None)
            ws.column_dimensions[col[0].column_letter].width = \
                min(max(width + 2, 10), 60)
    if verbose:
        print(f"[shortlist] 名单 {len(members)} 个、成表 {len(df)} 行 "
              f"-> {xlsx_path}")
    return df, xlsx_path


if __name__ == "__main__":
    # 冒烟测试：默认配置 + 单策略，输出到临时目录
    import tempfile
    try:
        from .strategies import TwapTwoStage
    except ImportError:
        from strategies import TwapTwoStage

    with tempfile.TemporaryDirectory() as tmp:
        res, run_dir = run_batch(
            [TwapTwoStage(name="两段式TWAP")],
            BacktestConfig(start_date="2016-06-01", delta_cap=2.0),
            tag="smoke", out_root=tmp)
        files = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")
                       if p.is_file())
        print("\n产出文件:")
        for f in files:
            print(f"  {f}")
