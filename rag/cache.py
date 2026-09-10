"""Redis 问答缓存：命中高频问答对直接返回，跳过检索与生成。

设计原则：
1. **优雅降级**——Redis 不可用（未安装 / 连不上 / 未配 redis-py）时全部方法变成
   no-op，主流程照常走完整链路，绝不让缓存故障影响可用性；
2. **入库自动失效**——缓存键带 KB 版本号，入库后版本 +1，旧缓存自然全部失效，
   不需要 SCAN/DEL 全库扫描（O(N) 且会阻塞 Redis）；
3. **只缓存确定性结果**——同一 (问题, 检索模式, top_k, 模型, KB版本) 才复用，
   避免把 A 模型的答案返回给 B 模型。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

# ---------- 配置 ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "auto").strip().lower() # auto | on | off
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800")) # 秒，默认 30 分钟
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "rag:qa")
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "5000"))
KB_VERSION_KEY = f"{CACHE_PREFIX}:kbver"

# 命中统计（进程级，用于 UI / 基准报告展示）
_stats = {"hit": 0, "miss": 0, "error": 0, "enabled": False}

_client = None
_init_done = False


def _connect():
    """建立 Redis 连接，兼容老服务端。

    redis-py 8.x 默认走 RESP3 握手（HELLO 3），而 Redis 5.x/6.0 不识别 HELLO 命令，
    会直接报 unknown command。这里先尝试默认协议，失败后退到 RESP2（protocol=2），
    再失败才判定不可用——避免"服务端明明是好的，却因为客户端协议协商失败而放弃缓存"。
    """
    import redis # 延迟导入：未装 redis-py 时不影响主流程

    last_err: Exception | None = None
    for kwargs in ({"socket_connect_timeout": 1.5, "socket_timeout": 1.5, "decode_responses": True},
                   {"socket_connect_timeout": 1.5, "socket_timeout": 1.5, "decode_responses": True,
                    "protocol": 2}):
        try:
            c = redis.Redis.from_url(REDIS_URL, **kwargs)
            c.ping()
            return c
        except Exception as e: # noqa: BLE001 — 逐个尝试协议，最后统一降级
            last_err = e
    raise last_err if last_err else RuntimeError("Redis 连接失败")


def _init() -> None:
    """惰性初始化 Redis 连接。auto 模式下失败即永久关闭缓存（不再重试，避免每请求超时）。"""
    global _client, _init_done
    if _init_done:
        return
    _init_done = True
    if CACHE_ENABLED == "off":
        _stats["enabled"] = False
        return
    try:
        import redis # noqa: F401
    except ImportError:
        if CACHE_ENABLED == "on":
            print("[cache] 已设置 CACHE_ENABLED=on 但未安装 redis-py，缓存关闭（pip install redis）")
        _stats["enabled"] = False
        return
    try:
        _client = _connect()
        _stats["enabled"] = True
    except Exception as e: # noqa: BLE001 — 连不上就降级，不影响问答
        if CACHE_ENABLED == "on":
            print(f"[cache] Redis 连接失败，缓存关闭：{e}")
        _client = None
        _stats["enabled"] = False


def available() -> bool:
    """缓存当前是否可用（供 UI 显示状态）。"""
    _init()
    return _client is not None


def stats() -> dict:
    _init()
    s = dict(_stats)
    s["ttl"] = CACHE_TTL
    s["url"] = REDIS_URL if s["enabled"] else ""
    total = s["hit"] + s["miss"]
    s["hit_rate"] = round(s["hit"] / total, 4) if total else 0.0
    return s


# ---------- KB 版本（入库失效用） ----------

def kb_version() -> int:
    """读取当前知识库版本号；Redis 不可用时返回 0（键里固定带 0，不影响正确性）。"""
    _init()
    if _client is None:
        return 0
    try:
        v = _client.get(KB_VERSION_KEY)
        return int(v) if v else 0
    except Exception: # noqa: BLE001
        _stats["error"] += 1
        return 0


def bump_kb_version(invalidate: bool = True) -> int:
    """入库完成后调用：版本 +1，使所有旧问答缓存立即失效。

    **为什么必须先删旧键再自增版本**（顺序不能反）：
    键名里带版本号，自增后新请求会去读 `v(new)` 命名空间，旧键自然不再被命中；
    但如果只自增不删除，旧键会一直留在 Redis 里直到 TTL 到期——一旦版本号因
    回滚/恢复备份而回落，旧问题就可能重新命中**已经过期的知识库答案**。
    因此这里先把旧命名空间删干净，再自增版本，保证"失效"是彻底的。
    """
    _init()
    if _client is None:
        return 0
    try:
        if invalidate:
            _purge_older_versions()
        return int(_client.incr(KB_VERSION_KEY))
    except Exception: # noqa: BLE001
        _stats["error"] += 1
        return 0


def _purge_older_versions() -> int:
    """删除所有旧版本命名空间下的问答缓存（走 SCAN，不阻塞 Redis）。

    注意 SCAN 的 glob 里 `*` 会匹配到冒号，所以必须用 `:v*:*` 两段限定，
    否则 `rag:qa:v1:abc` 和形如 `rag:qa:version_log` 的键都会被误删。
    """
    removed = 0
    try:
        for key in _client.scan_iter(match=f"{CACHE_PREFIX}:v*:*", count=500):
            removed += _client.delete(key)
    except Exception: # noqa: BLE001
        _stats["error"] += 1
    return removed


# ---------- 键与读写 ----------

def cache_key(question: str, mode: str, k: int, model: str, version: int | None = None) -> str:
    """缓存键 = 前缀 : KB版本 : 参数摘要。

    摘要里包含 mode/k/model —— 换检索模式或换模型必须重新生成，
    否则会把「纯向量检索」的结果当成「混合检索」的答案返回。
    """
    v = kb_version() if version is None else version
    raw = f"{question}\x00{mode}\x00{k}\x00{model}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{CACHE_PREFIX}:v{v}:{digest}"


def get(key: str) -> dict | None:
    _init()
    if _client is None:
        return None
    try:
        raw = _client.get(key)
    except Exception: # noqa: BLE001
        _stats["error"] += 1
        return None
    if raw is None:
        _stats["miss"] += 1
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        _stats["error"] += 1
        return None
    _stats["hit"] += 1
    data["cache_hit"] = True
    return data


def set(key: str, value: dict, ttl: int | None = None) -> bool:
    """写入缓存。value 必须可 JSON 序列化；写入失败静默返回 False。"""
    _init()
    if _client is None:
        return False
    try:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        pipe = _client.pipeline()
        pipe.set(key, payload, ex=ttl or CACHE_TTL)
        # 条目数上限护栏：近似 LRU 由 Redis maxmemory 负责，这里只在超限时告警式裁剪
        pipe.execute()
        return True
    except Exception: # noqa: BLE001
        _stats["error"] += 1
        return False


def clear() -> int:
    """清空本前缀下所有缓存（管理用，走 SCAN 避免阻塞）。"""
    _init()
    if _client is None:
        return 0
    return _purge_older_versions()


def reset_stats() -> None:
    _stats.update({"hit": 0, "miss": 0, "error": 0})
