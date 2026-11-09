# -*- coding: utf-8 -*-
"""
curve_cache.py —— 第2层附件：PDE 曲线盘存档

动机：默认条款的全周期回测须做上万次 PDE 求解（4 槽位 x 约 2500 个交易日，
另加每腿发行日反解票息的两次），耗时约 10 分钟；配置不变时这些求解的
输入逐次相同。本模块把每次求解的整条输出曲线（PDECurves 的六个数组）
按**定价输入的哈希**落盘，同键的后续求解直接读盘、逐位还原原对象——
回测与执行层因此可在默认配置下免去全部 PDE 重算，只按不同价格查 delta。

键的构成（= solve_curves 的全部实际输入）
------------------------------------------
    条款      s0 / ki_pct / ko_pct / 发行日 / 到期日 / 全部敲出、敲入观察日
    票息      解析后的 c（反解票息时以 0、1 两点各存一条）
    估值日    value_date（None 解析为发行日，与 solve_curves 同约定）
    市场参数  sigma / repo / r —— solve_curves 不消费 snap.s：曲线本就是
              整条 S 轴上的函数，这正是"任意价格查 delta"可行的原因
    数值参数  n_space / halfwidth / theta / rannacher / ki_node，以及
              pde_engine.py 源文件的哈希（RANNACHER_SUBSTEPS 等模块常数
              与算法本身的任何改动都随之入键）

由此得到两个性质：

* **不需要单独的数据指纹。** 行情数据更新后 sigma/repo 序列改变，键随之
  改变，旧存档自动退化为未命中并重算；不存在吃到旧曲线的路径。
* **对冲侧改动不失效。** 合约选择、换月、名义口径、敞口上限、手续费均在
  曲线下游，不进入键——对冲规则变体（免调仓带、软上限等）复用同一批
  存档快速重跑。代价是 pde_engine.py 的任何改动（含注释）都会使全部
  存档失效并触发一次全量重算——以 10 分钟的自愈换取"存档与求解器
  永不脱节"的保证。

浮点入键取 IEEE-754 位模式（np.float64(...).tobytes()），不经十进制
往返，等值判定与计算所用值严格一致。

存储
----
data/pde_cache/<键前2位>/<sha1>.npz，每次求解一个文件（float64 原位存储，
约 30KB；全周期默认配置约 1 万个文件、300MB）。写入先落同目录临时文件再
原子改名（os.replace），中断不会留下损坏的存档；读到损坏文件按未命中
处理并删除。命中的曲线与首次计算逐位一致（npz 往返无精度损失）。

用法
----
backtest.BacktestConfig(curve_cache=True)（默认开启）与执行层 CurveBook
自动经由本模块求解；也可直接调用::

    from curve_cache import solve_curves_cached, solve_coupon_cached
    cv = solve_curves_cached(terms, snap, value_date=d, n_space=601)

首次运行边算边存；第二次起同键求解直接读盘。cache_info() 报告本进程
命中数与磁盘占用，purge() 清空存档。
"""

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import pde_engine
    from .pde_engine import (PDECurves, solve_curves,
                             N_SPACE, X_HALFWIDTH, THETA, RANNACHER, KI_NODE)
except ImportError:
    import pde_engine
    from pde_engine import (PDECurves, solve_curves,
                            N_SPACE, X_HALFWIDTH, THETA, RANNACHER, KI_NODE)

# 存档根目录：<项目根>/data/pde_cache
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "pde_cache"

_ARRAYS = ("s", "x", "v_out", "v_in", "d_out", "d_in")

# 本进程的命中统计（cache_info() 报告）
_hits = 0
_misses = 0


# ---------------------------------------------------------------------------
# 键
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _engine_tag() -> bytes:
    """pde_engine.py 源文件哈希：求解器与其模块常数的任何改动都使键改变。"""
    return hashlib.sha1(Path(pde_engine.__file__).read_bytes()).digest()


def _f64(x) -> bytes:
    """浮点按 IEEE-754 位模式入键（不经十进制往返）。"""
    return np.float64(x).tobytes()


def _i64(x) -> bytes:
    return np.int64(x).tobytes()


def _dates(seq) -> bytes:
    """日期序列按 int64 纳秒时间戳数组入键。

    显式归一到 ns：pandas 3.0 起日期解析默认微秒精度，asi8 随精度变值；
    不归一则同一日期在新旧 pandas 下产生不同的键，存档整体失效。
    """
    return pd.DatetimeIndex(seq).as_unit("ns").asi8.tobytes()


def _key(terms, snap, coupon, value_date, n_space, halfwidth, theta,
         rannacher, ki_node) -> str:
    """定价输入的 sha1。与 solve_curves 的入参解析约定严格一致。"""
    c = terms.coupon if coupon is None else coupon
    if c is None:
        raise ValueError("需要票息：传入 coupon 或先 with_coupon")
    v = terms.issue_date if value_date is None else pd.Timestamp(value_date)

    h = hashlib.sha1()
    h.update(_engine_tag())
    # 条款
    h.update(_f64(terms.s0))
    h.update(_f64(terms.ki_pct))
    h.update(_f64(terms.ko_pct))
    h.update(_i64(terms.issue_date.value))
    h.update(_i64(terms.maturity_date.value))
    h.update(_dates(terms.ko_obs_dates))
    h.update(_dates(terms.ki_obs_dates))
    # 票息与估值日
    h.update(_f64(c))
    h.update(_i64(v.value))
    # 市场参数（snap.s 不入键：不参与求解）
    h.update(_f64(snap.sigma))
    h.update(_f64(snap.repo))
    h.update(_f64(snap.r))
    # 数值参数
    h.update(_i64(n_space))
    h.update(_f64(halfwidth))
    h.update(_f64(theta))
    h.update(_i64(rannacher))
    h.update(str(ki_node).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def _load(path: Path):
    """读一条存档；损坏（截断、格式错）按未命中处理并删除。"""
    try:
        with np.load(path, allow_pickle=False) as z:
            return PDECurves(**{a: z[a] for a in _ARRAYS})
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _store(path: Path, cv: PDECurves) -> None:
    """原子写入：同目录临时文件 + os.replace，中断不留半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **{a: getattr(cv, a) for a in _ARRAYS})
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 对外接口：与 pde_engine 同形的两个动作
# ---------------------------------------------------------------------------

def solve_curves_cached(terms, snap, coupon=None, value_date=None,
                        n_space: int = N_SPACE, halfwidth: float = X_HALFWIDTH,
                        theta: float = THETA, rannacher: int = RANNACHER,
                        ki_node: str = KI_NODE) -> PDECurves:
    """solve_curves 的盘存档版本：同签名、同结果（逐位一致）。"""
    global _hits, _misses
    k = _key(terms, snap, coupon, value_date, n_space, halfwidth, theta,
             rannacher, ki_node)
    path = CACHE_DIR / k[:2] / f"{k}.npz"
    if path.exists():
        cv = _load(path)
        if cv is not None:
            _hits += 1
            return cv
    _misses += 1
    cv = solve_curves(terms, snap, coupon, value_date, n_space=n_space,
                      halfwidth=halfwidth, theta=theta, rannacher=rannacher,
                      ki_node=ki_node)
    _store(path, cv)
    return cv


def solve_coupon_cached(terms, snap, **kw) -> float:
    """pde_solve_coupon 的盘存档版本：0、1 两点各走一次存档求解。

    公式与 pde_engine.pde_solve_coupon 逐字相同（解对票息严格线性）。
    """
    if abs(snap.s / terms.s0 - 1) > 1e-9:
        raise ValueError("solve_coupon 需要发行日快照（snap.s 应等于 terms.s0）")
    p0 = solve_curves_cached(terms, snap, 0.0, None, **kw).value(snap.s)
    p1 = solve_curves_cached(terms, snap, 1.0, None, **kw).value(snap.s)
    return (1.0 - p0) / (p1 - p0)


# ---------------------------------------------------------------------------
# 管理
# ---------------------------------------------------------------------------

def cache_info() -> dict:
    """本进程命中统计 + 磁盘占用。"""
    files = list(CACHE_DIR.rglob("*.npz")) if CACHE_DIR.exists() else []
    return {
        "hits": _hits,
        "misses": _misses,
        "files": len(files),
        "size_mb": round(sum(f.stat().st_size for f in files) / 2 ** 20, 1),
        "dir": str(CACHE_DIR),
    }


def purge() -> None:
    """清空全部存档（下次运行全量重算）。"""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


if __name__ == "__main__":
    import time

    try:
        from .data_loader import load_market_data
        from .futures_chain import FuturesChain
        from .market_params import build_market_params
        from .snowball_contract import make_snowball_weekly, next_issue_monday
    except ImportError:
        from data_loader import load_market_data
        from futures_chain import FuturesChain
        from market_params import build_market_params
        from snowball_contract import make_snowball_weekly, next_issue_monday

    md = load_market_data()
    params = build_market_params(md, FuturesChain(md))
    issue = next_issue_monday(md.trading_days, "2021-06-15")
    snap0 = params.snapshot(issue)
    terms = make_snowball_weekly(md.trading_days, issue, s0=snap0.s)

    # 反解票息（4 次求解：miss 后 hit 同键校验）
    tic = time.perf_counter()
    c1 = solve_coupon_cached(terms, snap0, n_space=601)
    t1 = time.perf_counter() - tic
    tic = time.perf_counter()
    c2 = solve_coupon_cached(terms, snap0, n_space=601)
    t2 = time.perf_counter() - tic
    print(f"票息反解: 首次 {t1:.3f}s -> 存档命中 {t2:.3f}s，"
          f"两次结果位级一致: {c1 == c2}")

    # 存续期曲线：命中结果与现解逐位一致
    terms = terms.with_coupon(c1)
    dv = "2021-09-14"
    snap = params.snapshot_prev(dv)
    cv_miss = solve_curves_cached(terms, snap, value_date=dv, n_space=601)
    cv_hit = solve_curves_cached(terms, snap, value_date=dv, n_space=601)
    cv_ref = solve_curves(terms, snap, value_date=dv, n_space=601)
    same = all(np.array_equal(getattr(cv_hit, a), getattr(cv_ref, a))
               for a in _ARRAYS)
    print(f"存续期曲线: 命中 vs 现解 六数组逐位一致: {same}")
    print(f"任意价位查询一致: "
          f"{cv_hit.delta(snap.s * 0.93, True) == cv_ref.delta(snap.s * 0.93, True)}")
    print(cache_info())
