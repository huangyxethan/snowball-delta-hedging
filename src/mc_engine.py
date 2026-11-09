# -*- coding: utf-8 -*-
"""
mc_engine.py —— 第2层：GBM 路径生成器（MC 引擎的随机部分）

刻意保持"无知"：本模块不知道雪球是什么，只负责生成风险中性测度下的
几何布朗运动日频路径。条款判定在 payoff.py，组装在 pricer.py。

两个关键设计：

1. 相对路径（初值=1）
   GBM 对初值是乘法齐次的：S_t = S0 × exp(...)，随机部分与 S0 无关。
   因此本引擎返回初值为 1 的相对路径，调用方乘以 S0 得到价格路径。
   好处：算 delta 时对 S0±h 的两次估值可以复用同一个路径数组
   （(S0+h)×rel 与 (S0−h)×rel），公共随机数天然实现、且零重复模拟。

2. 时间口径（混合惯例，故意为之）
   - 扩散：每个交易日步长 dt = 1/252。因为波动率是"日收益标准差×sqrt(252)"
     估出来的，sigma/sqrt(252) 才是经验上正确的单日标准差；
   - 票息计提与贴现：用实际日期的 自然日/365（payoff 层从条款取 tau）。
   两套口径各自对应自己的经验来源，这是业界标准做法。

方差缩减：对偶变量（antithetic variates），Z 与 −Z 成对出现，
路径均值的抽样误差显著降低，且不引入偏差。
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def simulate_relative_paths(mu: float, sigma: float, n_steps: int,
                            n_paths: int, seed: int,
                            antithetic: bool = True) -> np.ndarray:
    """生成初值为 1 的 GBM 相对路径。

    Parameters
    ----------
    mu : 风险中性漂移（= repo = r − q，年化）
    sigma : 年化波动率
    n_steps : 步数（= 模拟网格长度 − 1，每步一个交易日）
    n_paths : 路径条数（antithetic=True 时须为偶数）
    seed : 随机种子。同 (seed, n_steps, n_paths) 必产出同一数组——
           这是 delta 公共随机数与结果可复现的基础。
    antithetic : 对偶变量开关。前一半路径用 Z，后一半用 −Z。

    Returns
    -------
    ndarray, shape (n_paths, n_steps + 1)，第0列恒为 1.0。
    """
    if antithetic and n_paths % 2:
        raise ValueError("antithetic=True 时 n_paths 必须为偶数")

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    rng = np.random.default_rng(seed)

    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_steps))
    if antithetic:
        z = np.vstack([z, -z])

    # 精确离散化：log S_{k+1} − log S_k = (mu − sigma²/2)dt + sigma·sqrt(dt)·Z
    log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_inc, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = 1.0
    np.exp(log_paths, out=paths[:, 1:])
    return paths


if __name__ == "__main__":
    # 冒烟测试：验证生成的路径在统计上符合 GBM 理论值
    mu, sigma, n_steps, n_paths = -0.10, 0.20, 242, 200_000
    paths = simulate_relative_paths(mu, sigma, n_steps, n_paths, seed=42)

    t_diff = n_steps / TRADING_DAYS_PER_YEAR      # 扩散口径的总时长
    st = paths[:, -1]

    print(f"形状: {paths.shape}，第0列: {paths[0, 0]}")
    print(f"鞅性检验  E[S_T] = {st.mean():.6f}  vs 理论 e^(mu*t) = {np.exp(mu * t_diff):.6f}")

    log_ret = np.diff(np.log(paths), axis=1)
    print(f"单步波动  实测 {log_ret.std() * np.sqrt(252):.6f}  vs 理论 sigma = {sigma:.6f}")
    print(f"终值对数均值 实测 {np.log(st).mean():.6f}  vs 理论 (mu-sigma^2/2)t = "
          f"{(mu - 0.5 * sigma**2) * t_diff:.6f}")

    # 对偶性：随机项互为相反数 <=> 成对路径的对数增量之和恒等于 2×漂移项
    half = n_paths // 2
    inc = np.diff(np.log(paths), axis=1)
    drift = (mu - 0.5 * sigma ** 2) / TRADING_DAYS_PER_YEAR
    anti_ok = np.allclose(inc[:half] + inc[half:], 2 * drift)
    print(f"对偶变量结构正确: {anti_ok}")

    # 可复现性：同种子同结果
    paths2 = simulate_relative_paths(mu, sigma, n_steps, n_paths, seed=42)
    print(f"同种子可复现: {np.array_equal(paths, paths2)}")
