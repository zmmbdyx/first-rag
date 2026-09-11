"""统一检索入口：向量检索 / 关键词检索 / 混合检索（RRF 融合）。

混合检索采用 Reciprocal Rank Fusion：
    RRF(d) = Σ_路 1 / (RRF_K + rank_路(d))
只用排名不用原始分数，避免向量余弦距离与 BM25 分数两种量纲不可比的问题。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import vector_store
from .bm25 import BM25Index, build_from_chunks
from .config import (
    ACL_ADMIN_TAG,
    ACL_ENABLED,
    ACL_PUBLIC_TAG,
    ACL_STRICT,
    COLLECTION_NAME,
    FINAL_TOP_K,
    KEYWORD_TOP_K,
    RERANK_CANDIDATES,
    RETRIEVAL_MODE,
    RRF_K,
    RRF_P,
    VECTOR_TOP_K,
)
from .embeddings import embed_query
from .reranker import rerank, rerank_available

# 开启权限过滤时，为抵消"后过滤"造成的名额损失而放大的召回倍数。
# 敏感文档占比越高越需要放大；默认 3 倍，可用环境变量覆盖。
ACL_OVERSAMPLE = int(os.getenv("RAG_ACL_OVERSAMPLE", "3"))

# 近似去重的字符 3-gram 重合率阈值：短文本被长文本"包含"到该比例即视为同一内容
DEDUP_CONTAINMENT = 0.85


def _dedup(hits: list["Hit"], threshold: float = DEDUP_CONTAINMENT) -> list["Hit"]:
    """修复：检索结果原先没有任何去重，只有 RRF 融合阶段按 chunk_id 去重。
    但 chunk_id 不同、正文高度重复的块（重复入库、同段落在两篇文档里重复、
    表格分组后内容重叠）仍会同时占据 top-k，既浪费宝贵的上下文预算，
    又挤掉其他文档的有效证据。这里按「字符 3-gram 包含率」做保守去重，
    保留排名更靠前（分数更高）的那一块。
    """
    kept: list["Hit"] = []
    kept_grams: list[set[str]] = []
    for h in hits:
        norm = re.sub(r"[\s\W_]+", "", h.text or "")
        grams = {norm[i:i + 3] for i in range(max(1, len(norm) - 2))} if norm else set()
        if grams and any(pg and len(grams & pg) / min(len(grams), len(pg)) >= threshold
                         for pg in kept_grams):
            continue
        kept.append(h)
        kept_grams.append(grams)
    return kept


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
    sources: list[str] = field(default_factory=list) # 命中路径：vector/keyword
    # 溯源与治理元数据
    perm_tags: list[str] = field(default_factory=list)
    effective_from: str = ""
    effective_to: str = ""
    doc_version: str = ""

    @property
    def location(self) -> str:
        loc = self.doc_name
        if self.section_path:
            loc += f" · {self.section_path}"
        if self.page and self.page > 0:
            loc += f"（第{self.page}页）"
        return loc


# ---------------------------------------------------------------- 权限（ACL）
def normalize_groups(user_groups: list[str] | None) -> list[str]:
    """规整用户组：去空、去重、保序。"""
    seen: list[str] = []
    for g in (user_groups or []):
        g = (g or "").strip()
        if g and g not in seen:
            seen.append(g)
    return seen


def is_visible(chunk_tags: list[str] | None, user_groups: list[str] | None) -> bool:
    """判断一个 chunk 对给定用户组是否可见。

    规则（与 config 的开关一致）：
      * 调用方未提供用户组（如 CLI / 内部调用）→ 不做限制，全部可见；
      * 用户组含 ACL_ADMIN_TAG → 全部可见；
      * chunk 标签含 public → 全员可见；
      * chunk 无标签 → 宽松模式下可见（兼容历史索引），
        严格模式（RAG_ACL_STRICT=1）下**仅管理员可见**（fail-closed）；
      * 其余情况：标签与用户组有交集才可见。
    """
    if not ACL_ENABLED:
        return True
    groups = normalize_groups(user_groups)
    if not groups:
        # 没有用户上下文时不在这里做限制：调用方（API 层）必须保证
        # 生产请求一定带组；CLI/评测等内部调用则天然是受信上下文。
        return True
    if ACL_ADMIN_TAG and ACL_ADMIN_TAG in groups:
        return True

    tags = [t for t in (chunk_tags or []) if t]
    if not tags:
        return not ACL_STRICT
    if ACL_PUBLIC_TAG and ACL_PUBLIC_TAG in tags:
        return True
    return bool(set(tags) & set(groups))


class Retriever:
    """加载向量库 + BM25 索引，提供 vector / keyword / hybrid 三种检索模式。"""

    def __init__(self, index_dir, collection_name: str = COLLECTION_NAME, normalize: bool | None = None):
        client = vector_store.get_client(index_dir)
        self.collection = vector_store.get_collection(client, collection_name, create=True)
        # 余弦空间配归一化向量；旧集合（无 metadata，默认 l2）保持未归一化以兼容历史向量
        if normalize is None:
            normalize = (self.collection.metadata or {}).get("hnsw:space") == "cosine"
        self.normalize = normalize
        # BM25 索引按集合隔离，避免多集合共享同一 pickle 串位
        self.bm25_path = Path(index_dir) / f"bm25_{collection_name}.pkl"
        self.bm25: BM25Index | None = BM25Index.load(self.bm25_path)

    # ---------- 单路检索 ----------

    def vector_search(self, question: str, k: int = VECTOR_TOP_K,
                      user_groups: list[str] | None = None) -> list[Hit]:
        if self.collection.count() == 0:
            return []
        qvec = embed_query(question, normalize=self.normalize)
        hits = []
        for h in vector_store.query(self.collection, qvec, k):
            dist = h.pop("distance", None)
            hit = Hit(**h, score=-dist if dist is not None else 0.0)
            if is_visible(hit.perm_tags, user_groups):
                hits.append(hit)
        return hits

    def keyword_search(self, question: str, k: int = KEYWORD_TOP_K,
                       user_groups: list[str] | None = None) -> list[Hit]:
        if self.bm25 is None:
            return []
        pairs = self.bm25.search(question, k)
        if not pairs:
            return []
        # 一次批量取回元数据，避免逐命中回查的 N+1
        got = self.collection.get(ids=[cid for cid, _ in pairs],
                                  include=["documents", "metadatas"])
        by_id = {cid: (doc or "", meta or {}) for cid, doc, meta in
                 zip(got["ids"], got["documents"], got["metadatas"])}
        hits = []
        for cid, score in pairs: # 保持 BM25 排名顺序
            if cid not in by_id:
                continue
            text, meta = by_id[cid]
            tags = vector_store.meta_to_tags(meta.get("perm_tags"))
            if not is_visible(tags, user_groups):
                continue
            hits.append(Hit(
                chunk_id=cid, text=text,
                doc_name=meta.get("doc_name", ""),
                section_path=meta.get("section_path", ""),
                page=meta.get("page", -1),
                score=score,
                has_table="【表格】" in text,
                perm_tags=tags,
                effective_from=meta.get("effective_from", "") or "",
                effective_to=meta.get("effective_to", "") or "",
            ))
        return hits

    # ---------- 混合检索 ----------

    def retrieve(self, question: str, mode: str = RETRIEVAL_MODE, k_final: int = FINAL_TOP_K,
                 k_each: int | None = None, rrf_k: int | None = None,
                 rrf_p: float | None = None, expansion: str | None = None,
                 rerank_on: bool | None = None,
                 user_groups: list[str] | None = None) -> list[Hit]:
        """统一检索入口。

        - mode: vector / keyword / hybrid；
        - rrf_k / rrf_p: 融合超参（score = Σ 1/(k+rank)^p），None 时用全局配置；
        - expansion: 查询扩展词串（LLM 生成），拼接到查询后用于两路召回；
        - rerank_on: 是否用 CrossEncoder 精排（None 时按 RERANK_ENABLED 配置）；
        - user_groups: 调用方所属用户组，用于**权限过滤**。传 None 表示受信上下文
          （CLI / 离线评测 / 内部批处理），不做限制；API 层必须显式传入。
        """
        rrf_k = RRF_K if rrf_k is None else rrf_k
        rrf_p = RRF_P if rrf_p is None else rrf_p
        k_each = k_each or max(VECTOR_TOP_K, KEYWORD_TOP_K)
        use_rerank = rerank_available() if rerank_on is None else rerank_on
        # 重排需要比 k_final 更多的候选
        # 修复：原先只取 max(k_each, k_final)，使 RERANK_CANDIDATES（默认 20）从未生效——
        # 重排最多只能看到 10 个候选，配置项形同虚设。这里把重排候选数纳入召回规模。
        k_search = max(k_each, k_final, RERANK_CANDIDATES) if use_rerank else k_final

        # 权限过滤发生在检索之后（见 vector_store.query 的说明），
        # 因此这里放大召回量，避免过滤后候选不足导致 top-k 缩水。
        groups = normalize_groups(user_groups)
        filtering = ACL_ENABLED and bool(groups) and ACL_ADMIN_TAG not in groups
        if filtering:
            k_search *= max(1, ACL_OVERSAMPLE)
            k_each *= max(1, ACL_OVERSAMPLE)

        query = f"{question} {expansion}".strip() if expansion else question

        if mode == "vector":
            hits = self.vector_search(query, k_search, user_groups)
        elif mode == "keyword":
            hits = self.keyword_search(query, k_search, user_groups)
        else:
            vec_hits = self.vector_search(query, k_each, user_groups)
            kw_hits = self.keyword_search(query, k_each, user_groups)
            if not kw_hits: # 查询词完全不在语料中（如纯英文问题）→ 回退向量
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

        # 修复：先做近似去重再去重排/截断，避免重复块挤占最终 top-k 名额
        hits = _dedup(hits)
        if not hits:
            return []
        if use_rerank:
            hits = rerank(question, hits)
        return hits[:k_final]

    # ---------- 索引维护 ----------

    def rebuild_bm25(self, variant: str | None = None) -> int:
        from .config import BM25_VARIANT

        chunks = vector_store.hydrate_all(self.collection)
        v = variant or (self.bm25.variant if self.bm25 else BM25_VARIANT)
        self.bm25 = build_from_chunks(chunks, v)
        if self.bm25_path:
            self.bm25.save(self.bm25_path)
        return len(chunks)
