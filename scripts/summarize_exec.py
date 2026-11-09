# -*- coding: utf-8 -*-
"""执行策略汇总 Excel（shortlist.xlsx）维护入口。

汇总表是"入选策略"的对比 Excel，与 summary.csv（该配置下全部策略）
不同，只含手工挑进名单的策略。名单存在配置目录的 shortlist.json，
本脚本调整名单并整表重建：

    python scripts/summarize_exec.py                          # 按现名单重建
    python scripts/summarize_exec.py --set twap_1h,twap_hour_pace
    python scripts/summarize_exec.py --add twap_power_p1      # 追加
    python scripts/summarize_exec.py --remove twap_1h         # 移除

产出 output/exec_runs/<cfg_hash>/shortlist.xlsx。每行来自
reports/<策略名>.txt（重跑过的策略自动取最新结果）；回测时自动入选
可用 batch_exec.py 的 --shortlist 开关。

指数配置须与 batch_exec.py 的 CFG_INDEX 一致（决定 cfg_hash 目录）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from batch import update_shortlist
from batch_exec import CFG_INDEX


def _names(s):
    return [n.strip() for n in s.split(",") if n.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="执行策略汇总 Excel 维护")
    ap.add_argument("--set", dest="set_", default=None,
                    help="整体替换名单（逗号分隔）")
    ap.add_argument("--add", default=None, help="追加策略（逗号分隔）")
    ap.add_argument("--remove", default=None, help="移除策略（逗号分隔）")
    args = ap.parse_args()

    df, path = update_shortlist(
        CFG_INDEX,
        names=_names(args.set_) if args.set_ else None,
        add=_names(args.add) if args.add else None,
        remove=_names(args.remove) if args.remove else None)
    if not df.empty:
        cols = ["策略", "年化差(pp)", "年化跟踪误差(pp)", "信息比率",
                "数量项年化(pp)", "价格项年化(pp)", "费差项年化(pp)"]
        print(df[cols].round(3).to_string(index=False))
