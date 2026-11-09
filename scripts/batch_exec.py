# -*- coding: utf-8 -*-
"""批量执行回测入口。

用法：在下方「策略清单」处增删策略实例，然后

    python scripts/batch_exec.py
    python scripts/batch_exec.py --tag hour-pace     # 自定义运行目录后缀
    python scripts/batch_exec.py --only twap_1h      # 只跑清单中的部分策略
    python scripts/batch_exec.py --shortlist         # 回测后自动加入汇总
                                                     # Excel（shortlist.xlsx）

产出（目录 = 指数配置，同配置的策略跨运行积累在同一目录；
见 batch.py 模块 docstring）：
    output/exec_runs/<cfg_hash>/
        manifest.json / summary.csv / summary.txt
        plots/<策略名>.png                              # 同名重跑覆盖
        reports/<策略名>.txt
        caches/<策略名>.parquet                         # 逐日明细
        orders/<策略名>.parquet                         # 分钟订单流
    output/exec_runs/leaderboard.csv        跨运行累积总表（完整历史）

策略名约定用英文（进文件名与 leaderboard），参数变体后缀区分
（如 twap_hour_pace_1450）。

全部策略共享同一次日频回测与同一批 PDE 曲线（盘存档命中时回测约 25s），
批量应写成一个大清单一次跑，而非多次调用本脚本。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest import BacktestConfig
from batch import run_batch, update_shortlist
from execution import ExecutionConfig
from strategies import (TwapTwoStage, TwapHourPace, TwapPowerWeight,
                        TwapVolProfile, TwapLastHour, TwapPreWindow,
                        MarketOnSignal)

# ---------------------------------------------------------------------------
# 指数配置（改动任何字段会得到新的 cfg_hash，leaderboard 中不同 hash 的行
# 不可直接比较）。区间与 scripts/make_synthetic_data.py 生成的合成分钟数据
# 对齐；换成真实数据时按数据区间调整。
# ---------------------------------------------------------------------------
CFG_INDEX = BacktestConfig(
    start_date="2022-01-03",
    end_date="2023-12-29",
    delta_cap=2.0,
    # 其余字段与 BacktestConfig 默认值一致（周网格 4 槽位、PDE、
    # 当月合约、5 日换月、leg 名义口径、曲线盘存档开启）
)

CFG_EXEC = ExecutionConfig(
    fill_mode="bar_vwap",        # 成交价 = 所在分钟的成交量加权价
    fee_rate=None,               # 沿用指数的手续费率
)

# ---------------------------------------------------------------------------
# 策略清单（时刻为分钟 bar 的结束标记，如 135500 = 13:55）。
# 清单长度即批量粒度：日常迭代放一个，参数扫描放一族。
# ---------------------------------------------------------------------------
STRATEGIES = [
    # 对照：13:55 信号、14:00 一次性集中成交
    MarketOnSignal(name="market_on_signal", signal_time=135500,
                   trade_time=140000, delta_cap=2.0),
    # 对照：执行窗与结算价计价窗重合的一小时 TWAP
    TwapLastHour(name="twap_1h", signal_time=135500, delta_cap=2.0),
    # 基准：两段式 TWAP（13:55 信号 + 14:55 收尾修正）
    TwapTwoStage(name="twap_2stage", signal_time=135500, final_time=145500,
                 recalc=True, delta_cap=2.0),
    # 切片权重族：等速 / 幂权重（power=0 等权，>0 后置，<0 前置）
    TwapHourPace(name="twap_hour_pace", signal_time=135500,
                 final_time=145500, delta_cap=2.0),
    TwapPowerWeight(name="twap_power_p1", signal_time=135500,
                    final_time=145500, power=1.0, delta_cap=2.0),
    TwapPowerWeight(name="twap_power_p3", signal_time=135500,
                    final_time=145500, power=3.0, delta_cap=2.0),
    # 切片权重族：以市场分钟成交量分布为基底
    TwapVolProfile(name="twap_volprof_p0", power=0.0, window=20,
                   delta_cap=2.0),
    # 对照：成交落在结算价计价窗之外
    TwapPreWindow(name="twap_pre_window", signal_time=113000,
                  mid_time=135500, final_time=145500, delta_cap=2.0),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="批量执行回测")
    ap.add_argument("--tag", default="run", help="运行目录后缀")
    ap.add_argument("--start", default="2023-02-06",
                    help="执行回测起点；默认与合成分钟数据窗口对齐，真实数据下可设 None 跑全区间")
    ap.add_argument("--end", default="2023-06-30",
                    help="执行回测终点；默认与合成分钟数据窗口对齐")
    ap.add_argument("--only", default=None,
                    help="只跑清单中指定名字的策略（逗号分隔）")
    ap.add_argument("--shortlist", action="store_true",
                    help="回测后把本次策略加入汇总 Excel（shortlist.xlsx）"
                         "并重建")
    args = ap.parse_args()

    strategies = STRATEGIES
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        strategies = [s for s in STRATEGIES if s.name in names]
        missing = names - {s.name for s in strategies}
        if missing:
            ap.error(f"清单中没有这些策略：{'、'.join(sorted(missing))}")

    run_batch(strategies, CFG_INDEX, CFG_EXEC, tag=args.tag,
              start=args.start, end=args.end)
    if args.shortlist:
        update_shortlist(CFG_INDEX, add=[s.name for s in strategies])
