"""BM25 关键词检索索引（jieba 分词 + rank_bm25）。

作用：弥补纯向量检索的短板——
- 产品型号、编号、专有名词等精确词匹配；
- 问题与原文用词一致但语义向量不敏感的场景。
索引从向量库全量导出构建，与向量检索共享同一批 chunk 与元数据。
"""

import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

_PUNCT = re.compile(r"^[\W_]+$")

jieba.setLogLevel(60)  ***REMOVED*** 关闭初始化日志


def tokenize(text: str) -> list[str]:
    toks = []
    for t in jieba.lcut(text.lower()):
        t = t.strip()
        if t and not _PUNCT.match(t):
            toks.append(t)
    return toks


class BM25Index:
    def __init__(self, chunk_ids: list[str], token_lists: list[list[str]]):
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi(token_lists) if token_lists else None

    def search(self, query_text: str, k: int = 10) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序，无匹配时返回空。"""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query_text))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunk_ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25Index | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:  ***REMOVED*** noqa: BLE001
            return None


def build_from_chunks(chunks: list[dict]) -> BM25Index:
    """chunks: vector_store.hydrate_all 的输出。"""
    ids = [c["chunk_id"] for c in chunks]
    tokens = [tokenize(c["text"]) for c in chunks]
    return BM25Index(ids, tokens)
