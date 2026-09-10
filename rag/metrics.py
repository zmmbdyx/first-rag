"""运行指标：SQLite 落库（延迟/Token/错误率）+ 阈值告警 + 报告聚合。

每次 chat() 调用写入一条记录；`scripts/metrics_report.py` 输出 P50/P95/P99、
错误率、Token 消耗。阈值告警写入 logs/alerts.log（当前为本地单机版，
接 Prometheus/企业微信 webhook 只需替换 alert() 的落地方式）。
"""

import logging
import math
import sqlite3
import statistics
import time
from pathlib import Path

from .config import ROOT

DB_PATH = Path(ROOT / "logs" / "metrics.db")
ALERT_LOG = Path(ROOT / "logs" / "alerts.log")

***REMOVED*** 告警阈值（环境变量可覆盖）
ALERT_LATENCY_P95_MS = 5000
ALERT_ERROR_RATE = 0.1
ALERT_MIN_SAMPLES = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, model TEXT, mode TEXT,
    retrieval_ms REAL, gen_ms REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    error INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ts ON chat_metrics(ts);
"""


def record(model: str, mode: str, retrieval_ms: float, gen_ms: float,
           prompt_tokens: int = 0, completion_tokens: int = 0, error: bool = False) -> None:
    """写入一条问答指标并触发阈值检查。失败不影响主流程。"""
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(DB_PATH)
        try:
            con.executescript(_SCHEMA)
            con.execute(
                "INSERT INTO chat_metrics (ts, model, mode, retrieval_ms, gen_ms,"
                " prompt_tokens, completion_tokens, error) VALUES (?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), model, mode,
                 round(retrieval_ms, 1), round(gen_ms, 1),
                 int(prompt_tokens), int(completion_tokens), int(error)))
            con.commit()
        finally:
            con.close()
        _check_alerts()
    except sqlite3.Error as e:  ***REMOVED*** 并发写锁等 SQLite 异常——指标失败绝不影响主流程
        logging.warning("metrics 落库失败: %s", e)
    except OSError as e:  ***REMOVED*** 磁盘/权限问题
        logging.warning("metrics 落库失败: %s", e)


def _check_alerts() -> None:
    stats = aggregate(last_n=ALERT_MIN_SAMPLES)
    if not stats or stats["n"] < ALERT_MIN_SAMPLES:
        return
    alerts = []
    if stats["p95_gen_ms"] > ALERT_LATENCY_P95_MS:
        alerts.append(f"生成延迟 P95={stats['p95_gen_ms']:.0f}ms 超阈值 {ALERT_LATENCY_P95_MS}ms")
    if stats["error_rate"] > ALERT_ERROR_RATE:
        alerts.append(f"错误率 {stats['error_rate']:.1%} 超阈值 {ALERT_ERROR_RATE:.0%}")
    if alerts:
        from .security import _append_log

        _append_log(ALERT_LOG, {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "alerts": alerts, "window_n": stats["n"]})


def aggregate(last_n: int = 500) -> dict:
    """聚合最近 last_n 条记录的延迟分位数 / 错误率 / Token 消耗。"""
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            "SELECT retrieval_ms, gen_ms, prompt_tokens, completion_tokens, error "
            "FROM chat_metrics ORDER BY id DESC LIMIT ?", (last_n,)).fetchall()
    finally:
        con.close()
    if not rows:
        return {}
    ret = [r[0] for r in rows]
    gen = [r[1] for r in rows]
    gen_sorted = sorted(gen)

    def _pct(q: float) -> float:
        """最近秩法（nearest-rank）：返回第 ceil(q*n) 个观测值。

        修复：原实现 `gen_sorted[int(len*q)]` 取的是"比 q 分位多一位"的元素，
        n=20 时 P95 直接落到最大值（int(19)=19），小样本下 P95/P99 被系统性高估，
        会让 ALERT_LATENCY_P95_MS 阈值告警偏晚甚至失效。
        """
        rank = max(1, math.ceil(q * len(gen_sorted)))
        return gen_sorted[min(rank - 1, len(gen_sorted) - 1)]

    return {
        "n": len(rows),
        "p50_retrieval_ms": round(statistics.median(ret), 1),
        "p50_gen_ms": round(_pct(0.5), 1),
        "p95_gen_ms": round(_pct(0.95), 1),
        "p99_gen_ms": round(_pct(0.99), 1),
        "error_rate": round(sum(r[4] for r in rows) / len(rows), 4),
        "prompt_tokens_total": sum(r[2] for r in rows),
        "completion_tokens_total": sum(r[3] for r in rows),
        "avg_tokens_per_query": round((sum(r[2] + r[3] for r in rows)) / len(rows), 1),
    }


def estimate_tokens(text: str) -> int:
    """端点不回传 usage 时的兜底估算：中文约 1.5 字符/token。"""
    return max(1, int(len(text or "") / 1.5))
