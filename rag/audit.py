"""审计日志：PII 脱敏 + 按天轮转 + 保留期清理。

改造前的三个问题
----------------
1. ``audit.jsonl`` 落盘**完整 prompt、检索结果与答案原文**，无任何脱敏 ——
   员工问的敏感问题与被命中的敏感文档片段被明文长期留存；
2. 无轮转，单文件无限增长；
3. 无保留期，无法响应"删除我的数据"。

本模块提供：
* ``redact()``：对手机号、身份证、邮箱、银行卡、API Key、IP、私钥等做掩码；
* ``audit()``：写入前按开关脱敏；可按 ``AUDIT_STORE_TEXT=0`` 只存长度摘要与哈希；
* ``rotate_if_needed()``：按天切分文件；
* ``purge_expired()``：删除超过保留期的审计文件。

注意：脱敏是**最大限度降低风险**，不是合规级匿名化。真正的合规方案是
"不落原文"（AUDIT_STORE_TEXT=0）+ 独立加密存储 + 访问审计。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from .config import (
    AUDIT_ENABLED,
    AUDIT_REDACT,
    AUDIT_RETENTION_DAYS,
    AUDIT_STORE_TEXT,
    ROOT,
)

LOG_DIR = ROOT / "logs"
AUDIT_LOG = LOG_DIR / "audit.jsonl"

_lock = threading.Lock()

# 脱敏规则：(名称, 正则, 替换模板)。顺序有意义——先处理长模式，避免被短模式截断。
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # 私钥整块
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "[REDACTED:private_key]"),
    # API Key / Token
    ("api_key", re.compile(r"\b(sk|pk|ghp|gho|ghs|ghr|xox[baprs])-[A-Za-z0-9_\-]{16,}"), "[REDACTED:api_key]"),
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]{16,}"), "Bearer [REDACTED:token]"),
    # 身份证：18 位（末位可能为 X）
    ("id_card", re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])"), "[REDACTED:id_card]"),
    # 银行卡：16-19 位连续数字（排除已匹配的身份证长度）
    ("bank_card", re.compile(r"(?<![\d.])\d{16}(?![\d])"), "[REDACTED:bank_card]"),
    # 手机号：中国大陆 11 位
    ("phone", re.compile(r"(?<![\d.])1[3-9]\d{9}(?![\d.])"), "[REDACTED:phone]"),
    # 邮箱
    ("email", re.compile(r"(?<![.\w])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?![.\w])"),
     "[REDACTED:email]"),
    # IP（含内网）
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED:ip]"),
]


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """对文本做 PII 掩码，返回 (脱敏后文本, 各规则命中次数)。"""
    if not text:
        return text, {}
    counts: dict[str, int] = {}
    out = text
    for name, pat, repl in _PATTERNS:
        out, n = pat.subn(repl, out)
        if n:
            counts[name] = counts.get(name, 0) + n
    return out, counts


def _digest(text: str) -> str:
    """脱敏后仍想知道"是不是同一段内容"时用的短哈希。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def redact_event(event: dict) -> tuple[dict, dict[str, int]]:
    """递归脱敏整个事件字典。返回 (新事件, 命中统计)。"""
    total: dict[str, int] = {}

    def walk(node):
        if isinstance(node, str):
            cleaned, counts = redact_text(node)
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
            return cleaned
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(event), total


def _text_or_summary(value: str, key: str) -> object:
    """按 AUDIT_STORE_TEXT 决定存原文还是存摘要。"""
    if AUDIT_STORE_TEXT:
        return value
    return {"len": len(value or ""), "sha256_16": _digest(value or ""), "field": key}


def _current_path() -> Path:
    """按天分文件：audit-YYYYMMDD.jsonl（便于按保留期整文件删除）。"""
    return LOG_DIR / f"audit-{time.strftime('%Y%m%d')}.jsonl"


def rotate_if_needed() -> None:
    """确保日志目录存在（按天分文件后无需显式轮转，这里只做目录准备）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def purge_expired(retention_days: int | None = None) -> int:
    """删除超过保留期的审计文件，返回删除数量。"""
    days = AUDIT_RETENTION_DAYS if retention_days is None else retention_days
    if days <= 0 or not LOG_DIR.exists():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for p in LOG_DIR.glob("audit-*.jsonl*"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def audit(event: dict) -> dict:
    """写入一条审计事件（已脱敏、已按天分文件）。

    返回实际落盘的事件（脱敏后），便于调用方观察被掩码了什么。
    """
    if not AUDIT_ENABLED:
        return {}
    rotate_if_needed()

    payload = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}

    # 大文本字段按开关决定存原文还是摘要
    for key in ("prompt", "context", "answer", "question"):
        if key in payload and isinstance(payload[key], str):
            payload[key] = _text_or_summary(payload[key], key)

    redactions: dict[str, int] = {}
    if AUDIT_REDACT:
        payload, redactions = redact_event(payload)
    if redactions:
        payload["_redactions"] = redactions

    try:
        with _lock:
            with open(_current_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # 审计失败绝不能影响问答主流程
        return {}
    return payload
