"""统计工具测试：bootstrap 置信区间与配对显著性。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.stats import bootstrap_ci, fmt_ci, paired_bootstrap_diff  ***REMOVED*** noqa: E402


def test_bootstrap_ci_contains_mean():
    vals = [1.0] * 70 + [0.0] * 30  ***REMOVED*** 均值 0.7
    mean, lo, hi = bootstrap_ci(vals, n_boot=500)
    assert abs(mean - 0.7) < 1e-9
    assert lo <= mean <= hi
    assert 0.55 < lo and hi < 0.85  ***REMOVED*** 200 样本的 CI 不应过宽


def test_paired_diff_detects_improvement():
    a = [0.0] * 60 + [1.0] * 40   ***REMOVED*** 基线 40%
    b = [0.0] * 30 + [1.0] * 70   ***REMOVED*** 改进后 70%
    r = paired_bootstrap_diff(b, a)  ***REMOVED*** B - A
    assert r["diff"] > 0.25
    assert r["lo"] > 0          ***REMOVED*** 显著为正
    assert r["p"] < 0.05
    assert r["n"] == 100


def test_paired_diff_none_handling():
    a = [0.5, None, 0.8]
    b = [0.7, 0.2, None]
    r = paired_bootstrap_diff(a, b)
    assert r["n"] == 1          ***REMOVED*** 仅第一对同时非 None


def test_empty_values():
    mean, lo, hi = bootstrap_ci([])
    assert mean != mean  ***REMOVED*** NaN
    r = paired_bootstrap_diff([], [])
    assert r["n"] == 0


def test_fmt_ci():
    assert fmt_ci(0.955, 0.93, 0.98) == "95.5% [93.0%, 98.0%]"
    assert fmt_ci(0.912, 0.9, 0.92, pct=False) == "0.912 [0.900, 0.920]"
    assert fmt_ci(float("nan"), 0, 1) == "-"
