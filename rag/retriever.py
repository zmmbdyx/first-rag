"""统一检索入口：向量检索 / 关键词检索 / 混合检索（RRF 融合）。

混合检索采用 Reciprocal Rank Fusion：
    RRF(d) = Σ_路 1 / (RRF_K + rank_路(d))
只用排名不用原始分数，避免向量余弦距离与 BM25 分数两种量纲不可比的问题。
"""

from dataclasses import dataclass, field
from pathlib import Path

from . import vector_store
from .bm25 import BM25Index, build_from_chunks
from .config import (
    COLLECTION_NAME,
    FINAL_TOP_K,
    KEYWORD_TOP_K,
    RETRIEVAL_MODE,
    RERANK_ENABLED,
    RRF_K,
    RRF_P,
    VECTOR_TOP_K,
)
from .embeddings import embed_query
from .reranker import rerank, rerank_available


@dataclass
class Hit:
    chunk_id: str
    text: str
    doc_name: str
    section_path: str
    page: int
    score: float = 0.0
    vec_rank: int | None = None
    kw_rank: int | None = None
    rerank_score: float | None = None
    has_table: bool = False
    sources: list[str] = field(default_factory=list)  ***REMOVED*** 命中路径：vector/keyword

    @property
    def location(self) -> str:
        loc = self.doc_name
        if self.section_path:
            loc += f" · {self.section_path}"
        if self.page and self.page > 0:
            loc += f"（第{self.page}页）"
        return loc


class Retriever:
    """加载向量库 + BM25 索引，提供 vector / keyword / hybrid 三种检索模式。"""

    def __init__(self, index_dir, collection_name: str = COLLECTION_NAME, normalize: bool | None = None):
        client = vector_store.get_client(index_dir)
        self.collection = vector_store.get_collection(client, collection_name, create=True)
        ***REMOVED*** 余弦空间配归一化向量；旧集合（无 metadata，默认 l2）保持未归一化以兼容历史向量
        if normalize is None:
            normalize = (self.collection.metadata or {}).get("hnsw:space") == "cosine"
        self.normalize = normalize
        ***REMOVED*** BM25 索引按集合隔离，避免多集合共享同一 pickle 串位
        self.bm25_path = Path(index_dir) / f"bm25_{collection_name}.pkl"
        self.bm25: BM25Index | None = BM25Index.load(self.bm25_path)

    ***REMOVED*** ---------- 单路检索 ----------

    def vector_search(self, question: str, k: int = VECTOR_TOP_K) -> list[Hit]:
        if self.collection.count() == 0:
            return []
        qvec = embed_query(question, normalize=self.normalize)
        hits = []
        for h in vector_store.query(self.collection, qvec, k):
            dist = h.pop("distance", None)
            hits.append(Hit(**h, score=-dist if dist is not None else 0.0))
        return hits

    def keyword_search(self, question: str, k: int = KEYWORD_TOP_K) -> list[Hit]:
        if self.bm25 is None:
            return []
        pairs = self.bm25.search(question, k)
        if not pairs:
            return []
        ***REMOVED*** 一次批量取回元数据，避免逐命中回查的 N+1
        got = self.collection.get(ids=[cid for cid, _ in pairs],
                                  include=["documents", "metadatas"])
        by_id = {cid: (doc or "", meta or {}) for cid, doc, meta in
                 zip(got["ids"], got["documents"], got["metadatas"])}
        hits = []
        for cid, score in pairs:  ***REMOVED*** 保持 BM25 排名顺序
            if cid not in by_id:
                continue
            text, meta = by_id[cid]
            hits.append(Hit(
                chunk_id=cid, text=text,
                doc_name=meta.get("doc_name", ""),
                section_path=meta.get("section_path", ""),
                page=meta.get("page", -1),
                score=score,
                has_table="【表格】" in text,
            ))
        return hits

    ***REMOVED*** ---------- 混合检索 ----------

    def retrieve(self, question: str, mode: str = RETRIEVAL_MODE, k_final: int = FINAL_TOP_K,
                 k_each: int | None = None, rrf_k: int | None = None,
                 rrf_p: float | None = None, expansion: str | None = None,
                 rerank_on: bool | None = None) -> list[Hit]:
        """统一检索入口。

        - mode: vector / keyword / hybrid；
        - rrf_k / rrf_p: 融合超参（score = Σ 1/(k+rank)^p），None 时用全局配置；
        - expansion: 查询扩展词串（LLM 生成），拼接到查询后用于两路召回；
        - rerank_on: 是否用 CrossEncoder 精排（None 时按 RERANK_ENABLED 配置）。
        """
        rrf_k = RRF_K if rrf_k is None else rrf_k
        rrf_p = RRF_P if rrf_p is None else rrf_p
        k_each = k_each or max(VECTOR_TOP_K, KEYWORD_TOP_K)
        use_rerank = rerank_available() if rerank_on is None else rerank_on
        ***REMOVED*** 重排需要比 k_final 更多的候选
        k_search = max(k_each, k_final) if use_rerank else k_final
        query = f"{question} {expansion}".strip() if expansion else question

        if mode == "vector":
            hits = self.vector_search(query, k_search)
        elif mode == "keyword":
            hits = self.keyword_search(query, k_search)
        else:
            vec_hits = self.vector_search(query, k_each)
            kw_hits = self.keyword_search(query, k_each)
            if not kw_hits:      ***REMOVED*** 查询词完全不在语料中（如纯英文问题）→ 回退向量
                hits = vec_hits
            elif not vec_hits:
                hits = kw_hits
            else:
                fused: dict[str, Hit] = {}
                for rank, h in enumerate(vec_hits, start=1):
                    h.vec_rank, h.sources = rank, ["vector"]
                    fused[h.chunk_id] = h
                for rank, h in enumerate(kw_hits, start=1):
                    if h.chunk_id in fused:
                        fused[h.chunk_id].kw_rank = rank
                        fused[h.chunk_id].sources.append("keyword")
                    else:
                        h.kw_rank, h.sources = rank, ["keyword"]
                        fused[h.chunk_id] = h
                for h in fused.values():
                    h.score = sum(1.0 / (rrf_k + r) ** rrf_p
                                  for r in (h.vec_rank, h.kw_rank) if r is not None)
                hits = sorted(fused.values(), key=lambda h: h.score, reverse=True)

        if use_rerank and hits:
            hits = rerank(question, hits)
        return hits[:k_final]

    ***REMOVED*** ---------- 索引维护 ----------

    def rebuild_bm25(self, variant: str | None = None) -> int:
        from .config import BM25_VARIANT

        chunks = vector_store.hydrate_all(self.collection)
        v = variant or (self.bm25.variant if self.bm25 else BM25_VARIANT)
        self.bm25 = build_from_chunks(chunks, v)
        if self.bm25_path:
            self.bm25.save(self.bm25_path)
        return len(chunks)
