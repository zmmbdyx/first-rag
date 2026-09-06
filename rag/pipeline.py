"""入库与问答编排：解析、切分、向量化、索引、检索、生成、改写、安全、审计的完整流程。"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chunking import Chunk, flatten_doc_text, naive_chunk, smart_chunk
from .config import (
    COLLECTION_NAME,
    FINAL_TOP_K,
    INDEX_DIR,
    LLM_MODEL,
    RETRIEVAL_MODE,
    SAMPLES_DIR,
)
from .embeddings import embed_texts
from .llm import answer_question, build_context, expand_query_llm
from .parsers import SUPPORTED_EXTS, parse_file
from .retriever import Retriever
from . import vector_store

MANIFEST_FILE = "manifest.json"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(f for f in sorted(p.iterdir()) if f.suffix.lower() in SUPPORTED_EXTS)
        elif p.is_file():
            files.append(p)
    return files


def _parse_and_chunk(path: Path, chunker: str):
    parsed = parse_file(path)
    if chunker == "naive":
        chunks = naive_chunk(flatten_doc_text(parsed), parsed.doc_name)
    else:
        chunks = smart_chunk(parsed)
    return parsed.doc_name, chunks


def ingest(
    paths: list[str | Path],
    index_dir=INDEX_DIR,
    collection_name: str = COLLECTION_NAME,
    chunker: str = "smart",  ***REMOVED*** smart | naive
    rebuild_bm25: bool = True,
    quiet: bool = False,
    incremental: bool = True,
    workers: int = 4,
) -> dict:
    """解析 → 切分 → 向量化 → 写入 Chroma → 重建 BM25。

    增量更新：基于文件 MD5 清单（manifest.json），内容未变且切分器一致的文档
    直接跳过，仅重新处理变更文档。同一文档重复入库自动覆盖旧块。
    """
    files = collect_files(paths)
    if not files:
        raise FileNotFoundError(f"未找到可解析的文档（支持 {sorted(SUPPORTED_EXTS)}）")

    log = (lambda *a: None) if quiet else print
    index_dir = Path(index_dir)
    client = vector_store.get_client(index_dir)
    collection = vector_store.get_collection(client, collection_name, create=True)

    manifest_path = index_dir / MANIFEST_FILE
    manifest: dict = {}
    if incremental and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}

    todo, skipped = [], []
    for f in files:
        digest = _md5(f)
        entry = manifest.get(f.name, {})
        if incremental and entry.get("md5") == digest and entry.get("chunker") == chunker:
            skipped.append(f.name)
            continue
        todo.append((f, digest))

    t0 = time.time()
    if skipped:
        log(f"⏭️  增量跳过 {len(skipped)} 篇未变更文档")
    if todo:
        log(f"📄 解析并切分 {len(todo)} 篇文档（{workers} 线程）...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda t: _parse_and_chunk(t[0], chunker), todo))

        all_chunks: list[Chunk] = []
        per_doc: dict[str, int] = {}
        for (doc_name, chunks), (f, digest) in zip(results, todo):
            vector_store.delete_doc(collection, doc_name)  ***REMOVED*** 幂等：先清旧块
            all_chunks.extend(chunks)
            per_doc[doc_name] = len(chunks)
            manifest[doc_name] = {"md5": digest, "chunker": chunker, "chunks": len(chunks)}
            log(f"  📄 {doc_name}: {len(chunks)} 个 chunk")

        log(f"🧠 计算向量（{len(all_chunks)} 块）...")
        vectors = embed_texts([c.embed_text() for c in all_chunks])
        vector_store.upsert_chunks(collection, all_chunks, vectors)
    else:
        per_doc = {}

    n_bm25 = 0
    if rebuild_bm25 and (todo or not (index_dir / f"bm25_{collection_name}.pkl").exists()):
        retriever = Retriever(index_dir, collection_name)
        n_bm25 = retriever.rebuild_bm25()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    stats = {
        "docs": per_doc,
        "skipped": skipped,
        "new_chunks": sum(per_doc.values()),
        "collection_count": collection.count(),
        "bm25_chunks": n_bm25,
        "elapsed": time.time() - t0,
    }
    log(f"✅ 入库完成：新入库 {len(per_doc)} 篇 / 跳过 {len(skipped)} 篇 / "
        f"库内共 {collection.count()} 块 / 用时 {stats['elapsed']:.1f}s")
    return stats


def load_retriever(index_dir=INDEX_DIR, collection_name: str = COLLECTION_NAME) -> Retriever:
    return Retriever(index_dir, collection_name)


***REMOVED*** ---------- 问答入口 ----------


def chat(
    message: str,
    history: list[dict] | None = None,
    retriever: Retriever | None = None,
    mode: str = RETRIEVAL_MODE,
    k_final: int = FINAL_TOP_K,
    model: str | None = None,
    use_llm_rewrite: bool = True,
    audit_enabled: bool = True,
    return_hits: bool = False,
) -> dict:
    """多轮对话统一入口：chat(message, history) -> answer, sources, ...

    流程：输入安全检查（拦截注入/超长） → 查询改写（规则+LLM 指代消解）
    → 混合检索 → 生成（含引用编号校验与自动重生成） → 审计日志。
    输入被拦截时抛出 security.InputBlocked（拦截记录已写入日志）。
    """
    from .security import audit, check_input
    from .rewrite import rewrite_query

    clean = check_input(message)
    history = history or []
    rw = rewrite_query(clean, history, use_llm=use_llm_rewrite)
    model = model or LLM_MODEL
    retriever = retriever or load_retriever(INDEX_DIR)

    from .config import EXPAND_QUERY

    expansion = expand_query_llm(rw["query"], model=model) if EXPAND_QUERY else None
    t0 = time.time()
    hits = retriever.retrieve(rw["query"], mode=mode, k_final=k_final, expansion=expansion)
    retrieval_latency = time.time() - t0

    result = answer_question(rw["query"], hits, model=model)
    result.update({
        "question": message,
        "query_used": rw["query"],
        "rewrite_method": rw["method"],
        "expansion": expansion,
        "mode": mode,
        "retrieval_latency": retrieval_latency,
        "blocked": False,
        "hits": [
            {
                "doc_name": h.doc_name,
                "section_path": h.section_path,
                "page": h.page,
                "location": h.location,
                "score": round(h.score, 4),
                "rerank_score": round(h.rerank_score, 4) if h.rerank_score is not None else None,
                "has_table": h.has_table,
                "sources": h.sources,
                "snippet": h.text[:200],
            }
            for h in hits
        ],
    })

    ***REMOVED*** ---- 指标落库（延迟 / Token / 错误率；含阈值告警） ----
    from .llm import _last_usage
    from .metrics import estimate_tokens, record

    usage = _last_usage
    p_tokens = usage.prompt_tokens if usage else estimate_tokens(build_context(hits) + rw["query"])
    c_tokens = usage.completion_tokens if usage else estimate_tokens(result["answer"])
    result["tokens"] = {"prompt": p_tokens, "completion": c_tokens}
    record(model, mode, retrieval_latency * 1000, result["latency"] * 1000,
           p_tokens, c_tokens, error=False)
    if audit_enabled:
        audit({
            "type": "chat",
            "question": message,
            "query_used": rw["query"],
            "rewrite_method": rw["method"],
            "expansion": expansion,
            "mode": mode,
            "model": model,
            "prompt": f"{build_context(hits)}\n\n问题：{rw['query']}",
            "retrieved": [
                {"doc": h.doc_name, "section": h.section_path, "page": h.page,
                 "score": round(h.score, 4), "chunk_id": h.chunk_id}
                for h in hits
            ],
            "answer": result["answer"],
            "citations": result["citations"],
            "forged_citations": result["forged_citations"],
            "latency": round(result["latency"], 3),
        })
    if not return_hits:
        result.pop("hits", None)
    return result


def ask(
    question: str,
    retriever: Retriever,
    mode: str = RETRIEVAL_MODE,
    k_final: int = FINAL_TOP_K,
    model: str | None = None,
    return_hits: bool = False,
) -> dict:
    """单轮问答（兼容旧接口）：内部走 chat 全流程（安全检查/审计）。"""
    return chat(question, history=None, retriever=retriever, mode=mode,
                k_final=k_final, model=model, return_hits=return_hits)


if __name__ == "__main__":
    ingest([SAMPLES_DIR])
