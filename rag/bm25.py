"""BM25 关键词检索索引（jieba 分词 + rank_bm25）。

作用：弥补纯向量检索的短板——
- 产品型号、编号、专有名词等精确词匹配；
- 问题与原文用词一致但语义向量不敏感的场景。

分词变体（BM25_VARIANT，用于消融对比）：
- precise       jieba 精确模式（默认 HMM）
- precise_dict  精确模式 + 自定义词典（型号/缩写/行业术语）
- search        搜索引擎模式 cut_for_search（细粒度切分）
- search_dict   搜索引擎模式 + 自定义词典

索引从向量库全量导出构建，与向量检索共享同一批 chunk 与元数据；
变体随索引持久化，查询端自动使用与索引一致的分词方式。
"""

import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from .config import CUSTOM_DICT

_PUNCT = re.compile(r"^[\W_]+$")

jieba.setLogLevel(60)  ***REMOVED*** 关闭初始化日志

_DICT_LOADED = False


def _ensure_dict() -> None:
    """加载自定义词典（幂等）。"""
    global _DICT_LOADED
    if _DICT_LOADED or not CUSTOM_DICT.exists():
        return
    jieba.load_userdict(str(CUSTOM_DICT))
    _DICT_LOADED = True


def _cut(text: str, variant: str) -> list[str]:
    if variant in ("search", "search_dict"):
        if variant == "search_dict":
            _ensure_dict()
        return list(jieba.cut_for_search(text))
    if variant == "precise_dict":
        _ensure_dict()
    return jieba.lcut(text)


def tokenize(text: str, variant: str = "precise") -> list[str]:
    toks = []
    for t in _cut(text.lower(), variant):
        t = t.strip()
        if t and not _PUNCT.match(t):
            toks.append(t)
    return toks


class BM25Index:
    def __init__(self, chunk_ids: list[str], token_lists: list[list[str]], variant: str = "precise"):
        self.chunk_ids = chunk_ids
        self.variant = variant
        self.bm25 = BM25Okapi(token_lists) if token_lists else None

    def search(self, query_text: str, k: int = 10) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序，无匹配时返回空。"""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query_text, self.variant))
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
                idx = pickle.load(f)
            if not hasattr(idx, "variant"):  ***REMOVED*** 兼容旧版本索引
                idx.variant = "precise"
            return idx
        except Exception:  ***REMOVED*** noqa: BLE001
            return None


def build_from_chunks(chunks: list[dict], variant: str = "precise") -> BM25Index:
    """chunks: vector_store.hydrate_all 的输出。"""
    ids = [c["chunk_id"] for c in chunks]
    tokens = [tokenize(c["text"], variant) for c in chunks]
    return BM25Index(ids, tokens, variant=variant)
