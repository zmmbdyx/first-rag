"""运行指标：SQLite 落库（延迟/Token/错误率）+ 阈值告警 + 报告聚合。

每次 chat() 调用写入一条记录；`scripts/metrics_report.py` 输出 P50/P95/P99、
错误率、Token 消耗。阈值告警写入 logs/alerts.log（当前为本地单机版，
接 Prometheus/企业微信 webhook 只需替换 alert() 的落地方式）。
"""

import hashlib
import logging
import math
import sqlite3
import statistics
import time
from pathlib import Path

from .config import ROOT

DB_PATH = Path(ROOT / "logs" / "metrics.db")
ALERT_LOG = Path(ROOT / "logs" / "alerts.log")

# 告警阈值（环境变量可覆盖）
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

# 每次问答的**明细**（可追溯到具体请求）：用于回答"这条回答为什么错"
# 与"哪个部门在烧 token"两个此前无法回答的问题。
#   request_id    贯穿 API→检索→生成→审计的关联键
#   user_hash     用户/租户标识的哈希（不存明文，兼顾归因与隐私）
#   conversation_id / message_id  关联到具体会话与回答
#   confidence_tier / top_score   检索置信度，用于分析"低置信仍作答"的比例
#   refused       是否被置信度门限拦下（未调用大模型）
#   acl_groups    本次检索使用的权限组（审计越权尝试）
_DETAIL_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_requests (
    request_id TEXT PRIMARY KEY,
    ts TEXT,
    conversation_id TEXT,
    message_id INTEGER,
    user_hash TEXT,
    model TEXT, mode TEXT,
    query_used TEXT,
    retrieval_ms REAL, gen_ms REAL, total_ms REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    n_hits INTEGER,
    top_score REAL,
    confidence_tier TEXT, confidence_basis TEXT,
    refused INTEGER DEFAULT 0,
    acl_groups TEXT,
    cache_hit INTEGER DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_req_ts ON chat_requests(ts);
CREATE INDEX IF NOT EXISTS idx_req_user ON chat_requests(user_hash);
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
    except sqlite3.Error as e: # 并发写锁等 SQLite 异常——指标失败绝不影响主流程
        logging.warning("metrics 落库失败: %s", e)
    except OSError as e: # 磁盘/权限问题
        logging.warning("metrics 落库失败: %s", e)


def hash_identity(value: str | None) -> str:
    """把用户/租户标识哈希后入库：既能按人归因成本，又不留存明文标识。"""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def record_request(request_id: str, **fields) -> None:
    """写入一次问答的明细行（可追溯到具体请求）。失败不影响主流程。"""
    if not request_id:
        return
    cols = {
        "request_id": request_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conversation_id": fields.get("conversation_id", ""),
        "message_id": fields.get("message_id"),
        "user_hash": fields.get("user_hash", ""),
        "model": fields.get("model", ""),
        "mode": fields.get("mode", ""),
        "query_used": fields.get("query_used", ""),
        "retrieval_ms": round(float(fields.get("retrieval_ms") or 0.0), 1),
        "gen_ms": round(float(fields.get("gen_ms") or 0.0), 1),
        "total_ms": round(float(fields.get("total_ms") or 0.0), 1),
        "prompt_tokens": int(fields.get("prompt_tokens") or 0),
        "completion_tokens": int(fields.get("completion_tokens") or 0),
        "n_hits": int(fields.get("n_hits") or 0),
        "top_score": (round(float(fields["top_score"]), 4)
                      if fields.get("top_score") is not None else None),
        "confidence_tier": fields.get("confidence_tier", ""),
        "confidence_basis": fields.get("confidence_basis", ""),
        "refused": int(bool(fields.get("refused"))),
        "acl_groups": ",".join(fields.get("acl_groups") or []),
        "cache_hit": int(bool(fields.get("cache_hit"))),
        "error": str(fields.get("error") or "")[:500],
    }
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(DB_PATH)
        try:
            con.executescript(_DETAIL_SCHEMA)
            con.execute(
                "INSERT OR REPLACE INTO chat_requests ("
                + ",".join(cols) + ") VALUES (" + ",".join("?" * len(cols)) + ")",
                tuple(cols.values()))
            con.commit()
        finally:
            con.close()
    except (sqlite3.Error, OSError) as e:
        logging.warning("metrics 明细落库失败: %s", e)


def get_request(request_id: str) -> dict | None:
    """按 request_id 取回明细，用于排障（"这条回答为什么错"）。"""
    if not request_id or not DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            con.executescript(_DETAIL_SCHEMA)
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM chat_requests WHERE request_id = ?",
                              (request_id,)).fetchone()
            return dict(row) if row else None
        finally:
            con.close()
    except (sqlite3.Error, OSError):
        return None


def cost_by_user(last_n_days: int = 7, limit: int = 50) -> list[dict]:
    """按用户聚合 token 消耗（成本归因）。"""
    if not DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            con.executescript(_DETAIL_SCHEMA)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT user_hash, COUNT(*) AS n,"
                " SUM(prompt_tokens) AS prompt_tokens,"
                " SUM(completion_tokens) AS completion_tokens,"
                " AVG(total_ms) AS avg_ms"
                " FROM chat_requests WHERE ts >= datetime('now', ?)"
                " GROUP BY user_hash ORDER BY (SUM(prompt_tokens)+SUM(completion_tokens)) DESC"
                " LIMIT ?",
                (f"-{int(last_n_days)} days", int(limit))).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
    except (sqlite3.Error, OSError):
        return []


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
