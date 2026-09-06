"""评测统计工具：bootstrap 置信区间与配对显著性检验（仅依赖 numpy）。"""

import numpy as np


def _clean(values) -> np.ndarray:
    return np.asarray([float(v) for v in values if v is not None], dtype=float)


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """均值的 bootstrap 置信区间。返回 (mean, lo, hi)；空数据返回 (nan, nan, nan)。"""
    v = _clean(values)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(v.mean()), lo, hi


def paired_bootstrap_diff(values_a, values_b, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """配对差值检验：A−B 的均值差、95% CI 与双侧 bootstrap p 值。

    values_a / values_b 必须按题目一一对应（None 配对剔除）。
    """
    pairs = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    if not pairs:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    d = np.asarray([a - b for a, b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    p_le = float((boot <= 0).mean())
    p_ge = float((boot >= 0).mean())
    p = max(2.0 * min(p_le, p_ge), 1.0 / n_boot)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "p": min(1.0, p), "n": int(d.size)}


def fmt_ci(mean: float, lo: float, hi: float, pct: bool = True) -> str:
    if mean != mean:  ***REMOVED*** NaN
        return "-"
    f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(mean)} [{f(lo)}, {f(hi)}]"
