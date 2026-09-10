"""入库与问答编排：解析、切分、向量化、索引、检索、生成、改写、安全、审计的完整流程。"""

import asyncio
import hashlib
import json
import os
import sys
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


***REMOVED*** ---------- 异步入库：解析阶段真并发 ----------

async def _parse_and_chunk_async(path: Path, chunker: str, sem: "asyncio.Semaphore"):
    """在线程池里执行阻塞式解析（PyMuPDF/python-docx 是同步库，且底层释放 GIL）。

    用 Semaphore 限制并发上限，避免一次提交上千个文件把内存打满。
    """
    async with sem:
        return await asyncio.to_thread(_parse_and_chunk, path, chunker)


async def _gather_parse(todo: list, chunker: str, workers: int):
    """并发解析全部待处理文档，**保持与 todo 相同的顺序**返回。"""
    sem = asyncio.Semaphore(max(1, workers))
    tasks = [_parse_and_chunk_async(f, chunker, sem) for f, _ in todo]
    return await asyncio.gather(*tasks)


def _run_async(coro):
    """在同步调用栈里跑协程，兼容"已在事件循环中"的调用场景。

    - 普通同步调用（CLI / FastAPI 的同步线程池端点）：直接 asyncio.run；
    - 已经被包在事件循环里（例如从 async 端点直接调用 ingest）：当前线程无法再
      run 一个循环，改到独立线程里新建事件循环执行，避免 RuntimeError。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import threading

    box: dict = {}

    def _worker():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as e:  ***REMOVED*** noqa: BLE001 — 跨线程回传原始异常
            box["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def ingest(
    paths: list[str | Path],
    index_dir=INDEX_DIR,
    collection_name: str = COLLECTION_NAME,
    chunker: str = "smart",  ***REMOVED*** smart | naive
    rebuild_bm25: bool = True,
    quiet: bool = False,
    incremental: bool = True,
    workers: int = 4,
    async_parse: bool | None = None,
) -> dict:
    """解析 → 切分 → 向量化 → 写入 Chroma → 重建 BM25。

    增量更新：基于文件 MD5 清单（manifest.json），内容未变且切分器一致的文档
    直接跳过，仅重新处理变更文档。同一文档重复入库自动覆盖旧块。

    并发模型：解析阶段默认走 **asyncio 事件循环 + 线程池**（`asyncio.to_thread`），
    相比串行解析能显著压缩入库墙钟时间；`async_parse=False` 可退回串行/线程池
    两种旧路径（用于基准对比）。
    """
    from .config import ASYNC_INGEST

    if async_parse is None:
        async_parse = ASYNC_INGEST

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
    parse_elapsed = 0.0
    if skipped:
        log(f"⏭️  增量跳过 {len(skipped)} 篇未变更文档")
    if todo:
        _t_parse = time.time()
        if async_parse:
            log(f"📄 异步解析并切分 {len(todo)} 篇文档（asyncio + 线程池，并发 {workers}）...")
            results = _run_async(_gather_parse(todo, chunker, workers))
        else:
            log(f"📄 解析并切分 {len(todo)} 篇文档（{workers} 线程）...")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda t: _parse_and_chunk(t[0], chunker), todo))
        parse_elapsed = time.time() - _t_parse

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

    ***REMOVED*** ---- 入库成功 → 知识库版本 +1，让所有旧问答缓存立即失效 ----
    ***REMOVED*** 放在最后一步：只有索引真正写成功才失效，否则会把还能用的缓存白白清掉。
    cache_version = 0
    if per_doc:
        from . import cache as qa_cache
        cache_version = qa_cache.bump_kb_version()

    stats = {
        "docs": per_doc,
        "skipped": skipped,
        "new_chunks": sum(per_doc.values()),
        "collection_count": collection.count(),
        "bm25_chunks": n_bm25,
        "elapsed": time.time() - t0,
        "parse_elapsed": parse_elapsed,
        "async_parse": bool(async_parse),
        "cache_version": cache_version,
    }
    log(f"✅ 入库完成：新入库 {len(per_doc)} 篇 / 跳过 {len(skipped)} 篇 / "
        f"库内共 {collection.count()} 块 / 用时 {stats['elapsed']:.1f}s"
        + (f"（解析 {parse_elapsed:.1f}s）" if per_doc else ""))
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
    use_cache: bool = True,
) -> dict:
    """多轮对话统一入口：chat(message, history) -> answer, sources, ...

    流程：输入安全检查（拦截注入/超长） → 查询改写（规则+LLM 指代消解）
    → **Redis 问答缓存** → 混合检索 → 生成（含引用编号校验与自动重生成） → 审计日志。
    输入被拦截时抛出 security.InputBlocked（拦截记录已写入日志）。

    use_cache: 是否允许命中/写入 Redis 问答缓存（Redis 不可用时自动降级为不走缓存）。
    """
    from .security import audit, check_input
    from .rewrite import rewrite_query
    from . import cache as qa_cache

    clean = check_input(message)
    history = history or []
    rw = rewrite_query(clean, history, use_llm=use_llm_rewrite)
    model = model or LLM_MODEL
    retriever = retriever or load_retriever(INDEX_DIR)

    ***REMOVED*** ---- Redis 问答缓存查询（高频问题直接返回，跳过检索与生成） ----
    ***REMOVED*** 键用改写前的原始问题：改写依赖 history，同一原始问题在不同会话里会被改写
    ***REMOVED*** 成不同检索式，但最终答案应当一致；且若拿改写后的问题做键，缓存命中率会被
    ***REMOVED*** 历史上下文人为拉低，失去"高频问答对"的意义。
    ckey = None
    if use_cache and qa_cache.available():
        ckey = qa_cache.cache_key(clean, mode, k_final, model)
        cached = qa_cache.get(ckey)
        if os.getenv("RAG_CACHE_DEBUG"):
            print(f"[cache-debug] key={ckey} hit={bool(cached)}", file=sys.stderr)
        if cached:
            cached.update({
                "question": message,
                "query_used": cached.get("query_used") or clean,
                "mode": mode,
                "blocked": False,
                "cache_hit": True,
                ***REMOVED*** 缓存命中 = 零生成耗时，延迟真实反映"直接返回"的收益
                "latency": 0.0,
                "retrieval_latency": 0.0,
            })
            cached.pop("cache_hit_ts", None)
            if audit_enabled:
                audit({"type": "chat_cache_hit", "question": message,
                       "cache_key": ckey, "mode": mode, "model": model,
                       "answer": cached.get("answer", "")})
            if not return_hits:
                cached.pop("hits", None)
            return cached

    from .config import EXPAND_QUERY

    expansion = expand_query_llm(rw["query"], model=model) if EXPAND_QUERY else None
    t0 = time.time()
    hits = retriever.retrieve(rw["query"], mode=mode, k_final=k_final, expansion=expansion)
    retrieval_latency = time.time() - t0

    ***REMOVED*** 修复：区分「知识库为空」与「检索无命中」——原先两种情况的答案文案完全一样
    ***REMOVED*** （都是"知识库为空，请先调用入库脚本导入文档"），检索不到内容时会误导用户去重复入库。
    kb_is_empty = False
    if not hits:
        try:
            kb_is_empty = retriever.collection.count() == 0
        except Exception:  ***REMOVED*** noqa: BLE001 — 统计失败不影响主流程，按"无命中"处理
            kb_is_empty = False

    t_gen0 = time.time()
    try:
        result = answer_question(rw["query"], hits, model=model, empty_kb=kb_is_empty)
    except Exception:
        ***REMOVED*** 修复：生成失败此前会直接抛出，metrics 一条都不落库，导致 logs/metrics.db 的
        ***REMOVED*** error_rate 恒为 0、ALERT_ERROR_RATE 阈值告警永远不可能触发（监控形同虚设）。
        ***REMOVED*** 这里补记一条 error=True 的指标后原样抛出，保证调用方语义不变。
        from .metrics import record as _record
        _record(model, mode, retrieval_latency * 1000, (time.time() - t_gen0) * 1000,
                0, 0, error=True)
        raise
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
    from .llm import get_last_usage
    from .metrics import estimate_tokens, record

    usage = get_last_usage()
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
    ***REMOVED*** ---- 写入 Redis 问答缓存（只缓存成功生成的答案） ----
    if ckey:
        to_cache = {k: v for k, v in result.items() if k != "hits"}
        to_cache.pop("cache_hit", None)
        ok = qa_cache.set(ckey, to_cache)
        if os.getenv("RAG_CACHE_DEBUG"):
            print(f"[cache-debug] set key={ckey} ok={ok}", file=sys.stderr)
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
