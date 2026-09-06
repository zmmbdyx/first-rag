"""入库与问答编排：把解析、切分、向量化、索引、检索、生成串成完整流程。"""

import time
from pathlib import Path

from .chunking import Chunk, flatten_doc_text, naive_chunk, smart_chunk
from .config import (
    COLLECTION_NAME,
    FINAL_TOP_K,
    INDEX_DIR,
    RETRIEVAL_MODE,
    SAMPLES_DIR,
)
from .embeddings import embed_texts
from .llm import answer_question
from .parsers import SUPPORTED_EXTS, parse_file
from .retriever import Retriever
from . import vector_store


def collect_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(f for f in sorted(p.iterdir()) if f.suffix.lower() in SUPPORTED_EXTS)
        elif p.is_file():
            files.append(p)
    return files


def ingest(
    paths: list[str | Path],
    index_dir=INDEX_DIR,
    collection_name: str = COLLECTION_NAME,
    chunker: str = "smart",  ***REMOVED*** smart | naive
    rebuild_bm25: bool = True,
    quiet: bool = False,
) -> dict:
    """解析 → 切分 → 向量化 → 写入 Chroma → 重建 BM25。同一文档重复入库自动覆盖。"""
    files = collect_files(paths)
    if not files:
        raise FileNotFoundError(f"未找到可解析的文档（支持 {sorted(SUPPORTED_EXTS)}）")

    log = (lambda *a: None) if quiet else print
    index_dir = Path(index_dir)
    client = vector_store.get_client(index_dir)
    collection = vector_store.get_collection(client, collection_name, create=True)

    t0 = time.time()
    all_chunks: list[Chunk] = []
    per_doc: dict[str, int] = {}
    for f in files:
        parsed = parse_file(f)
        if chunker == "naive":
            chunks = naive_chunk(flatten_doc_text(parsed), parsed.doc_name)
        else:
            chunks = smart_chunk(parsed)
        vector_store.delete_doc(collection, parsed.doc_name)  ***REMOVED*** 幂等：先清旧块
        all_chunks.extend(chunks)
        per_doc[parsed.doc_name] = len(chunks)
        log(f"  📄 {parsed.doc_name}: {len(parsed.blocks)} 个结构块 → {len(chunks)} 个 chunk")

    log(f"🧠 计算向量（{len(all_chunks)} 块）...")
    vectors = embed_texts([c.embed_text() for c in all_chunks])
    vector_store.upsert_chunks(collection, all_chunks, vectors)

    n_bm25 = 0
    if rebuild_bm25:
        retriever = Retriever(index_dir, collection_name)
        n_bm25 = retriever.rebuild_bm25()

    stats = {
        "docs": per_doc,
        "total_chunks": len(all_chunks),
        "collection_count": collection.count(),
        "bm25_chunks": n_bm25,
        "elapsed": time.time() - t0,
    }
    log(f"✅ 入库完成：{len(per_doc)} 篇文档 / {collection.count()} 块 / 用时 {stats['elapsed']:.1f}s")
    return stats


def load_retriever(index_dir=INDEX_DIR, collection_name: str = COLLECTION_NAME) -> Retriever:
    return Retriever(index_dir, collection_name)


def ask(
    question: str,
    retriever: Retriever,
    mode: str = RETRIEVAL_MODE,
    k_final: int = FINAL_TOP_K,
    model: str | None = None,
    return_hits: bool = False,
) -> dict:
    """完整问答：检索 → 生成 → 溯源。"""
    from .config import LLM_MODEL

    t0 = time.time()
    hits = retriever.retrieve(question, mode=mode, k_final=k_final)
    t_retrieval = time.time() - t0

    result = answer_question(question, hits, model=model or LLM_MODEL)
    result.update(
        {
            "question": question,
            "mode": mode,
            "retrieval_latency": t_retrieval,
            "hits": [
                {
                    "doc_name": h.doc_name,
                    "section_path": h.section_path,
                    "page": h.page,
                    "location": h.location,
                    "score": round(h.score, 4),
                    "sources": h.sources,
                    "snippet": h.text[:200],
                }
                for h in hits
            ],
        }
    )
    if not return_hits:
        result.pop("hits", None)
    return result


if __name__ == "__main__":
    ingest([SAMPLES_DIR])
