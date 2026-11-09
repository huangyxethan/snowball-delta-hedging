# -*- coding: utf-8 -*-
"""参数扫描入口：一族策略一次跑，只产出一张汇总表。

用法：编辑下方策略族生成逻辑，然后

    python scripts/sweep_exec.py --tag power

产出：output/exec_runs/<cfg_hash>/sweeps/<时间戳>_<tag>.csv 与 .txt
（行=策略、列=十项统计；不落三件套、不进 leaderboard——选中的参数
以正式名字经 batch_exec.py 入档）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from batch import run_sweep
from batch_exec import CFG_INDEX, CFG_EXEC
from strategies import TwapPowerWeight

# ---------------------------------------------------------------------------
# 示例策略族：幂权重两段式 TWAP 的幂指数 power 扫描（前置—均匀—后置谱系）。
# power<0 前置、0 等速（=TwapHourPace）、>0 后置。
# ---------------------------------------------------------------------------
POWERS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
STRATEGIES = [
    TwapPowerWeight(name=f"twap_power_p{p:g}", signal_time=135500,
                    final_time=145500, power=p, delta_cap=2.0)
    for p in POWERS
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="执行策略参数扫描")
    ap.add_argument("--tag", default="sweep", help="汇总表文件名后缀")
    ap.add_argument("--start", default="2023-02-06",
                    help="执行回测起点；默认与合成分钟数据窗口对齐，真实数据下可设 None 跑全区间")
    ap.add_argument("--end", default="2023-06-30",
                    help="执行回测终点；默认与合成分钟数据窗口对齐")
    args = ap.parse_args()

    run_sweep(STRATEGIES, CFG_INDEX, CFG_EXEC, tag=args.tag,
              start=args.start, end=args.end)
