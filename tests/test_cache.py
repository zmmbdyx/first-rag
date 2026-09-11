"""Redis 问答缓存与异步入库的单测。

缓存部分不依赖真实 Redis：通过 monkeypatch 替换底层 client，
从而在 CI（无 Redis）环境下也能验证键设计、失效语义与降级行为。
"""
import asyncio
import time
from pathlib import Path

import pytest

from rag import cache as qa_cache


# ---------------- 假 Redis（实现用到的子集） ----------------

class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def ping(self):
        return True

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v
        return True

    def incr(self, k):
        cur = int(self.store.get(k, 0)) + 1
        self.store[k] = str(cur)
        return cur

    def delete(self, *keys):
        n = 0
        for k in keys:
            n += 1 if self.store.pop(k, None) is not None else 0
        return n

    def scan_iter(self, match=None, count=None):
        import fnmatch
        pat = match or "*"
        for k in list(self.store):
            if fnmatch.fnmatch(k, pat):
                yield k

    def pipeline(self):
        return _FakePipe(self)


class _FakePipe:
    def __init__(self, r):
        self.r = r
        self.ops = []

    def set(self, k, v, ex=None):
        self.ops.append((k, v))

    def execute(self):
        for k, v in self.ops:
            self.r.store[k] = v
        return [True] * len(self.ops)


@pytest.fixture
def fake_redis(monkeypatch):
    """把 cache 模块切到假 Redis，并重置其单例状态与命中统计。

    注意两点，都是踩过的坑：
    1. 必须同时把 ``_init_done`` 置为 True。``get()/set()`` 内部会调用
       ``_init()``，而 ``_init()`` 只在 ``_init_done`` 为 False 时才去连真实
       Redis。若只替换 ``_client`` 而不封住 ``_init()``，第一次读写就会把
       假 client **覆盖回真实连接** —— 于是本机跑着 Redis 时用例会读到上次
       运行留下的真实缓存键，出现「单独跑也失败」的假失败（CI 无 Redis 时
       恰好连不上、``_client`` 被置 None，反而掩盖了这个问题）。
    2. 每个测试都拿到**全新的空 store**，避免测试之间互相污染。
    """
    fake = FakeRedis()
    monkeypatch.setattr(qa_cache, "_client", fake, raising=False)
    monkeypatch.setattr(qa_cache, "_init_done", True, raising=False)
    monkeypatch.setattr(qa_cache, "_stats",
                        {"hit": 0, "miss": 0, "error": 0, "enabled": True}, raising=False)
    monkeypatch.setattr(qa_cache, "available", lambda: True)
    return fake


# ---------------- 键设计 ----------------

def test_cache_key_isolates_mode_model_and_k():
    """换检索模式 / 换模型 / 换 top_k 必须是不同的键。

    否则会把「纯向量检索」的结果当成「混合检索」的答案返回，
    或把 A 模型的答案给 B 模型。
    """
    base = qa_cache.cache_key("试用期多久", "hybrid", 5, "m1", version=0)
    assert base != qa_cache.cache_key("试用期多久", "vector", 5, "m1", version=0)
    assert base != qa_cache.cache_key("试用期多久", "hybrid", 3, "m1", version=0)
    assert base != qa_cache.cache_key("试用期多久", "hybrid", 5, "m2", version=0)
    # 同一输入必须稳定（否则永远命中不了）
    assert base == qa_cache.cache_key("试用期多久", "hybrid", 5, "m1", version=0)


def test_cache_key_changes_with_kb_version():
    a = qa_cache.cache_key("q", "hybrid", 5, "m", version=1)
    b = qa_cache.cache_key("q", "hybrid", 5, "m", version=2)
    assert a != b
    assert ":v1:" in a and ":v2:" in b


def test_cache_key_does_not_collide_on_separator():
    """问题里包含分隔符时不能与另一组参数撞键。"""
    a = qa_cache.cache_key("q\x00hybrid", "vector", 5, "m", version=0)
    b = qa_cache.cache_key("q", "hybrid", 5, "m", version=0)
    assert a != b


# ---------------- 读写与统计 ----------------

def test_set_get_roundtrip_and_stats(fake_redis):
    # 用独立的问题串，避免与其他用例共用同一个键（每个用例都拿到全新的空 store，
    # 但显式不同的键能让失败信息更清晰）
    k = qa_cache.cache_key("roundtrip-question", "hybrid", 5, "m", version=0)
    assert qa_cache.get(k) is None # miss
    assert qa_cache.set(k, {"answer": "A", "citations": [1]})
    got = qa_cache.get(k)
    assert got["answer"] == "A" and got["cache_hit"] is True
    s = qa_cache.stats()
    assert s["hit"] == 1 and s["miss"] == 1 and s["hit_rate"] == 0.5


def test_get_returns_none_on_corrupt_payload(fake_redis):
    """缓存里是坏 JSON 时必须当作 miss，而不是抛异常打断问答。"""
    k = qa_cache.cache_key("q", "hybrid", 5, "m", version=0)
    fake_redis.store[k] = "{not json"
    assert qa_cache.get(k) is None
    assert qa_cache.stats()["error"] >= 1


# ---------------- 失效语义（本次修复的核心） ----------------

def test_bump_version_purges_old_entries(fake_redis):
    """入库后版本 +1，且**旧键必须被真正删除**。

    只自增不删除时，旧键会留到 TTL 到期；若版本号因回滚而回落，
    就可能重新命中已经过期的知识库答案。
    """
    old = qa_cache.cache_key("q", "hybrid", 5, "m", version=0)
    qa_cache.set(old, {"answer": "旧答案"})
    assert old in fake_redis.store

    newv = qa_cache.bump_kb_version()
    assert newv == 1
    assert old not in fake_redis.store # 旧键被清理

    new = qa_cache.cache_key("q", "hybrid", 5, "m")
    assert ":v1:" in new
    assert qa_cache.get(new) is None # 新命名空间里没有旧答案


def test_clear_removes_all_versions(fake_redis):
    for v in (0, 1, 2):
        qa_cache.set(qa_cache.cache_key("q", "hybrid", 5, "m", version=v), {"answer": str(v)})
    removed = qa_cache.clear()
    assert removed == 3
    assert not [k for k in fake_redis.store if k.startswith(qa_cache.CACHE_PREFIX)]


# ---------------- 降级行为（可用性优先） ----------------

def test_degrades_gracefully_without_redis(monkeypatch):
    """Redis 不可用时所有方法都要安全 no-op，绝不抛异常。"""
    monkeypatch.setattr(qa_cache, "_client", None, raising=False)
    monkeypatch.setattr(qa_cache, "_init_done", True, raising=False)
    monkeypatch.setattr(qa_cache, "_stats",
                        {"hit": 0, "miss": 0, "error": 0, "enabled": False}, raising=False)

    assert qa_cache.available() is False
    assert qa_cache.get("any:key") is None
    assert qa_cache.set("any:key", {"a": 1}) is False
    assert qa_cache.kb_version() == 0
    assert qa_cache.bump_kb_version() == 0
    assert qa_cache.clear() == 0


def test_corrupt_cache_does_not_break_chat(monkeypatch):
    """端到端护栏：缓存层任何异常都不能影响 chat 主流程。

    这里让底层的 get 直接抛异常，验证 pipeline 不会因此中断。
    """
    from rag import pipeline

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("redis down")

    monkeypatch.setattr(qa_cache, "_client", Boom(), raising=False)
    monkeypatch.setattr(qa_cache, "_init_done", True, raising=False)
    # get 内部已捕获异常 → 返回 None，不应向外抛
    assert qa_cache.get("k") is None


# ---------------- 异步入库 ----------------

def _docs(tmp_path: Path, n: int) -> list[Path]:
    files = []
    for i in range(n):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(
            "第一章 总则\n" + f"这是第{i}份测试文档，用于验证异步并发解析。\n" * 40
            + "\n第二章 细则\n内容若干。\n", encoding="utf-8")
        files.append(p)
    return files


def test_async_parse_matches_sync_results(tmp_path):
    """异步并发解析的结果必须与串行完全一致（顺序与内容都不能变）。"""
    from rag.pipeline import _gather_parse, _parse_and_chunk

    files = _docs(tmp_path, 6)
    todo = [(f, "md5") for f in files]

    sync = [_parse_and_chunk(f, "smart") for f in files]
    a_sync = asyncio.run(_gather_parse(todo, "smart", 4))

    assert [d for d, _ in a_sync] == [d for d, _ in sync] # 顺序保持
    assert [len(c) for _, c in a_sync] == [len(c) for _, c in sync]
    assert [c[0].text for _, c in a_sync] == [c[0].text for _, c in sync]


def test_gather_parse_respects_concurrency_limit(tmp_path, monkeypatch):
    """Semaphore 必须真正限制并发度，避免一次提交上千文件打满内存。"""
    from rag import pipeline

    files = _docs(tmp_path, 8)
    todo = [(f, "md5") for f in files]
    live = {"cur": 0, "max": 0}

    real = pipeline._parse_and_chunk

    def slow(path, chunker, perm_tags=None):
        """签名需与 pipeline._parse_and_chunk 保持一致（其后新增了 perm_tags 参数）。"""
        live["cur"] += 1
        live["max"] = max(live["max"], live["cur"])
        try:
            time.sleep(0.05)
            return real(path, chunker, perm_tags=perm_tags)
        finally:
            live["cur"] -= 1

    monkeypatch.setattr(pipeline, "_parse_and_chunk", slow)
    asyncio.run(pipeline._gather_parse(todo, "smart", 3))
    assert live["max"] <= 3, f"并发度超过上限: {live['max']}"


def test_run_async_works_inside_event_loop(tmp_path):
    """已在事件循环中调用时不能炸（FastAPI async 端点场景）。"""
    from rag.pipeline import _run_async

    async def outer():
        async def inner():
            return 42
        # 直接 asyncio.run 会 RuntimeError: event loop is already running
        return _run_async(inner())

    assert asyncio.run(outer()) == 42
