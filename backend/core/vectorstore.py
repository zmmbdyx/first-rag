"""Embedding 模型与向量库的进程级管理。

复用既有配置（``rag.config``）：``EMBED_MODEL`` / ``INDEX_DIR`` / ``COLLECTION_NAME``
都从环境变量读取，本模块不引入任何新的模型或密钥配置。
"""

from __future__ import annotations

from rag.config import COLLECTION_NAME, EMBED_MODEL, INDEX_DIR
from rag.pipeline import load_retriever
from rag.retriever import Retriever

__all__ = ["COLLECTION_NAME", "EMBED_MODEL", "INDEX_DIR", "get_retriever", "reload_retriever", "retriever_ready"]

# 进程级单例：嵌入模型加载一次（首次约数秒），后续请求直接复用
_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever | None:
    """返回常驻检索器；尚未加载返回 None（不抛异常，交给 /health 报告降级）。"""
    return _RETRIEVER


def reload_retriever(rebuild_bm25_if_empty: bool = True) -> Retriever | None:
    """（重新）加载检索器。

    入库新文档后 Chroma 集合内容变化，且 BM25 索引由入库流程重建，
    常驻检索器必须重新构造才能看到新块 —— 这一点与旧版 ``/ingest`` 行为一致。
    """
    global _RETRIEVER
    try:
        r = load_retriever(INDEX_DIR, COLLECTION_NAME)
        if rebuild_bm25_if_empty and r.bm25 is None and r.collection.count() > 0:
            r.rebuild_bm25()
        _RETRIEVER = r
    except Exception as e:  # noqa: BLE001 — 库缺失/未入库都不该阻止服务启动
        print(f"[core.vectorstore] 检索器加载失败（/api/health 会报告 degraded）：{e}")
        _RETRIEVER = None
    return _RETRIEVER


def retriever_ready() -> bool:
    return _RETRIEVER is not None


def collection_count() -> int | None:
    """向量库当前切片数；不可用时返回 None。"""
    if _RETRIEVER is None:
        return None
    try:
        return _RETRIEVER.collection.count()
    except Exception:  # noqa: BLE001
        return None
