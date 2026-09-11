"""Chroma 向量库封装：入库（幂等 upsert）、查询、全量导出、文档管理。

ACL 说明
--------
Chroma 的 metadata 只接受标量，因此 ``perm_tags`` 以**逗号分隔字符串**存储
（如 ``"hr,admin"``）。检索时用 ``where={"perm_tags": {"$in": [...]}}`` 过滤 ——
Chroma 的 ``$in`` 对标量字段做「字段值 ∈ 给定集合」判断，因此这里额外把
"标签串包含任一用户组"的候选一并取回后做二次校验，避免漏召回。
"""

import chromadb

from .config import COLLECTION_NAME


def get_client(index_dir):
    return chromadb.PersistentClient(path=str(index_dir))


def get_collection(client, name: str = COLLECTION_NAME, create: bool = True):
    """默认余弦距离 + L2 归一化向量：相似度只看方向，长度不干扰排序。"""
    if create:
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    try:
        return client.get_collection(name)
    except Exception: # noqa: BLE001
        return None


def tags_to_meta(tags: list[str] | None) -> str:
    """标签列表 -> Chroma 可存的标量（逗号分隔，去重保序）。"""
    seen: list[str] = []
    for t in (tags or []):
        t = (t or "").strip()
        if t and t not in seen:
            seen.append(t)
    return ",".join(seen)


def meta_to_tags(value) -> list[str]:
    """metadata 里的标签串 -> 列表。"""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [t.strip() for t in str(value or "").split(",") if t.strip()]


def upsert_chunks(collection, chunks, vectors) -> int:
    """chunks: rag.chunking.Chunk 列表；vectors: 对齐的向量矩阵。"""
    batch = 128
    total = 0
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.upsert(
            ids=[c.chunk_id for c in part],
            documents=[c.embed_text() for c in part],
            metadatas=[
                {
                    "doc_name": c.doc_name,
                    "section_path": c.section_path,
                    "page": int(c.page) if c.page else -1,
                    "chunk_index": j,
                    # 权限与时效元数据：供检索过滤与冲突消解使用
                    "perm_tags": tags_to_meta(getattr(c, "perm_tags", None)),
                    "effective_from": getattr(c, "effective_from", "") or "",
                    "effective_to": getattr(c, "effective_to", "") or "",
                    "doc_version": getattr(c, "doc_version", "") or "",
                }
                for j, c in enumerate(part, start=i)
            ],
            embeddings=vectors[i : i + batch],
        )
        total += len(part)
    return total


def _to_hits(res) -> list[dict]:
    hits = []
    if not res or not res.get("ids"):
        return hits
    ids = res["ids"][0]
    docs = res.get("documents", [[None] * len(ids)])[0]
    metas = res.get("metadatas", [[{}] * len(ids)])[0]
    dists = res.get("distances", [[None] * len(ids)])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        hits.append(
            {
                "chunk_id": cid,
                "text": doc or "",
                "doc_name": meta.get("doc_name", ""),
                "section_path": meta.get("section_path", ""),
                "page": meta.get("page", -1),
                "has_table": "【表格】" in (doc or ""),
                "perm_tags": meta_to_tags(meta.get("perm_tags")),
                "effective_from": meta.get("effective_from", "") or "",
                "effective_to": meta.get("effective_to", "") or "",
                "doc_version": meta.get("doc_version", "") or "",
                "distance": dist,
            }
        )
    return hits


def query(collection, qvec, k: int = 10) -> list[dict]:
    # 修复：原实现 `k = max(1, min(k, collection.count()))` 在空集合时会把 0 抬回 1，
    # 紧随其后的 `if k == 0: return []` 是永远走不到的死代码——空库查询仍会以
    # n_results=1 打到 Chroma（轻则返回空、重则抛 "Number of requested results > count"）。
    # 改为先判空直接返回，再对 k 做合法的区间钳制。
    #
    # 注意：权限过滤**不在这里**做。Chroma 的 metadata 过滤发生在 ANN 检索阶段，
    # 若在此处过滤会改变 n_results 的语义（可能返回不足 k 条），且各版本对
    # $in 语义的实现差异较大。ACL 统一在 Retriever 层做「多召回 + 后过滤」，
    # 语义明确且可测；代价是敏感文档较多时需要放大召回量（见 retriever 的 acl_oversample）。
    n = collection.count()
    if n <= 0:
        return []
    k = max(1, min(k, n))
    res = collection.query(query_embeddings=[qvec.tolist()], n_results=k,
                           include=["documents", "metadatas", "distances"])
    return _to_hits(res)


def hydrate_all(collection) -> list[dict]:
    """导出全部 chunk（构建 BM25 索引用）。"""
    got = collection.get(include=["documents", "metadatas"])
    out = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        meta = meta or {}
        out.append(
            {
                "chunk_id": cid,
                "text": doc or "",
                "doc_name": meta.get("doc_name", ""),
                "section_path": meta.get("section_path", ""),
                "page": meta.get("page", -1),
                "perm_tags": meta_to_tags(meta.get("perm_tags")),
                "effective_from": meta.get("effective_from", "") or "",
                "effective_to": meta.get("effective_to", "") or "",
            }
        )
    return out


def delete_doc(collection, doc_name: str) -> None:
    collection.delete(where={"doc_name": doc_name})


def set_doc_tags(collection, doc_name: str, tags: list[str]) -> int:
    """更新某文档全部 chunk 的权限标签（改密级）。返回受影响块数。"""
    got = collection.get(where={"doc_name": doc_name}, include=["metadatas"])
    ids = got.get("ids") or []
    if not ids:
        return 0
    value = tags_to_meta(tags)
    metas = []
    for meta in (got.get("metadatas") or []):
        m = dict(meta or {})
        m["perm_tags"] = value
        metas.append(m)
    collection.update(ids=ids, metadatas=metas)
    return len(ids)


def doc_stats(collection) -> dict[str, int]:
    stats: dict[str, int] = {}
    for item in hydrate_all(collection):
        stats[item["doc_name"]] = stats.get(item["doc_name"], 0) + 1
    return stats


def doc_catalog(collection) -> list[dict]:
    """文档清单（含权限标签与块数），供管理接口与合规审计使用。"""
    catalog: dict[str, dict] = {}
    for item in hydrate_all(collection):
        name = item["doc_name"]
        row = catalog.setdefault(name, {
            "doc_name": name,
            "chunks": 0,
            "perm_tags": set(),
            "effective_from": item.get("effective_from", ""),
            "sections": 0,
        })
        row["chunks"] += 1
        row["perm_tags"].update(item.get("perm_tags") or [])
        if item.get("section_path"):
            row["sections"] += 1
    out = []
    for row in catalog.values():
        row["perm_tags"] = sorted(row["perm_tags"])
        out.append(row)
    return sorted(out, key=lambda r: r["doc_name"])


def docs_without_tags(collection) -> list[dict]:
    """列出**没有任何权限标签**的文档（升级到带 ACL 版本后需要归类的历史数据）。

    这类文档在宽松模式（RAG_ACL_STRICT=0）下等同公开，在严格模式下仅管理员可见。
    生产开启严格模式前，应先用 ``backfill_missing_tags`` 或 ``set_doc_tags``
    把它们归类到正确的密级。
    """
    return [row for row in doc_catalog(collection) if not row["perm_tags"]]


def backfill_missing_tags(collection, tags: list[str] | None = None) -> tuple[int, int]:
    """给"无标签"的历史 chunk 补上默认标签。

    返回 (受影响文档数, 受影响块数)。这是**一次性迁移**，用于把升级前入库的
    数据纳入 ACL 体系；迁移前建议先跑 ``docs_without_tags`` 人工确认这些文档
    的密级 —— 无脑补成 public 会把本该保密的文档变成公开。
    """
    from .config import ACL_DEFAULT_TAGS

    value = tags_to_meta(tags if tags is not None else ACL_DEFAULT_TAGS)
    got = collection.get(include=["metadatas"])
    ids: list[str] = []
    metas: list[dict] = []
    for cid, meta in zip(got.get("ids") or [], got.get("metadatas") or []):
        m = dict(meta or {})
        if meta_to_tags(m.get("perm_tags")):
            continue
        m["perm_tags"] = value
        ids.append(cid)
        metas.append(m)
    if not ids:
        return 0, 0
    for i in range(0, len(ids), 256):
        collection.update(ids=ids[i:i + 256], metadatas=metas[i:i + 256])
    docs = {m.get("doc_name", "") for m in metas}
    return len(docs), len(ids)
