# -*- coding: utf-8 -*-
"""strategies —— 执行策略注册处（一个策略一个文件）。

框架（策略基类 ExecutionStrategy、DayContext、记账引擎、run_execution、
验收工具）在 execution.py，稳定不动；本包只放可交易的执行策略。

公开版收录基础策略族：集中成交与一小时 TWAP 两个对照、两段式基准、
等速 / 幂权重 / 成交量分布三种切片权重，以及计价窗前执行的对照。

新增策略的步骤：
    1. 新建 <strategy_name>.py，定义 ExecutionStrategy 子类
       （模板见任一现有文件；docstring 写明机制与参数）；
    2. 在下方导入并加入 __all__；
    3. 在 scripts/batch_exec.py 的清单中实例化即可回测。

策略名约定用英文（进档案文件名与 leaderboard），参数变体用后缀区分
（如 twap_hour_pace_1450）。
"""

from .twap_two_stage import TwapTwoStage
from .twap_hour_pace import TwapHourPace
from .twap_power_weight import TwapPowerWeight
from .twap_vol_profile import TwapVolProfile
from .twap_last_hour import TwapLastHour
from .twap_pre_window import TwapPreWindow
from .market_on_signal import MarketOnSignal

__all__ = ["TwapTwoStage", "TwapHourPace", "TwapPowerWeight", "TwapVolProfile",
           "TwapLastHour", "TwapPreWindow", "MarketOnSignal"]
