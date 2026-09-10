"""Chroma 向量库封装：入库（幂等 upsert）、查询、全量导出、文档管理。"""

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
        hits.append(
            {
                "chunk_id": cid,
                "text": doc or "",
                "doc_name": (meta or {}).get("doc_name", ""),
                "section_path": (meta or {}).get("section_path", ""),
                "page": (meta or {}).get("page", -1),
                "has_table": "【表格】" in (doc or ""),
                "distance": dist,
            }
        )
    return hits


def query(collection, qvec, k: int = 10) -> list[dict]:
    # 修复：原实现 `k = max(1, min(k, collection.count()))` 在空集合时会把 0 抬回 1，
    # 紧随其后的 `if k == 0: return []` 是永远走不到的死代码——空库查询仍会以
    # n_results=1 打到 Chroma（轻则返回空、重则抛 "Number of requested results > count"）。
    # 改为先判空直接返回，再对 k 做合法的区间钳制。
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
            }
        )
    return out


def delete_doc(collection, doc_name: str) -> None:
    collection.delete(where={"doc_name": doc_name})


def doc_stats(collection) -> dict[str, int]:
    stats: dict[str, int] = {}
    for item in hydrate_all(collection):
        stats[item["doc_name"]] = stats.get(item["doc_name"], 0) + 1
    return stats
