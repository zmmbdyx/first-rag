"""FastAPI 服务化入口：把与 UI 解耦的 rag/ 核心包暴露为 RESTful API。

与 Streamlit（app.py）的关系：
- **同一套核心包**——两边都只调用 rag.pipeline / rag.retriever，不重复实现任何 RAG 逻辑；
- Streamlit 用于交互式演示，FastAPI 用于服务化与集成（前端分离、被其他系统调用）。
- 检索器与嵌入模型是**进程级单例**（lifespan 里初始化一次），多 worker 部署时
  每个 worker 各持一份 —— 生产上应把嵌入/重排拆成独立推理服务，避免显存重复占用。

启动：
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    # 或 python api_server.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag import cache as qa_cache # noqa: E402
from rag import vector_store # noqa: E402
from rag.config import ( # noqa: E402
    API_KEYS,
    CACHE_TTL,
    COLLECTION_NAME,
    FINAL_TOP_K,
    INDEX_DIR,
    MODEL_OPTIONS,
    RETRIEVAL_MODE,
    UPLOAD_DIR,
)
from rag.parsers import SUPPORTED_EXTS # noqa: E402
from rag.pipeline import chat, ingest, load_retriever # noqa: E402
from rag.security import InputBlocked # noqa: E402

STATE: dict = {"retriever": None, "started_at": time.time()}


def _hit_to_dict(h) -> dict:
    """把内部 Hit 对象转成可 JSON 序列化的溯源结构。

    统一在 API 边界转换，避免内部 dataclass 泄漏到响应体里
    （曾导致 FastAPI 响应校验 500：sources 声明为 dict 却拿到 Hit 对象）。
    """
    return {
        "doc_name": getattr(h, "doc_name", ""),
        "section_path": getattr(h, "section_path", ""),
        "page": getattr(h, "page", -1),
        "location": getattr(h, "location", ""),
        "score": round(getattr(h, "score", 0.0), 4),
        "rerank_score": (round(h.rerank_score, 4)
                         if getattr(h, "rerank_score", None) is not None else None),
        "has_table": bool(getattr(h, "has_table", False)),
        "sources": list(getattr(h, "sources", []) or []),
        "snippet": (getattr(h, "text", "") or "")[:200],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载检索器（含 BM25 索引），避免首个请求承担冷启动延迟。"""
    try:
        r = load_retriever(INDEX_DIR, COLLECTION_NAME)
        if r.bm25 is None and r.collection.count() > 0:
            r.rebuild_bm25()
        STATE["retriever"] = r
    except Exception as e: # noqa: BLE001 — 库缺失不该阻止服务启动，健康检查会暴露
        print(f"[api] 检索器加载失败（/health 会报告 degraded）：{e}")
        STATE["retriever"] = None
    if qa_cache.available():
        print(f"[api] Redis 问答缓存已启用：{qa_cache.REDIS_URL} (TTL {CACHE_TTL}s)")
    else:
        print("[api] Redis 问答缓存未启用（降级：每次请求走完整链路）")
    yield
    STATE["retriever"] = None


app = FastAPI(
    title="企业知识库 RAG 问答 API",
    description="检索增强生成问答服务：混合检索 + 答案溯源 + Redis 问答缓存",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------- 鉴权 ----------------
def auth(authorization: str | None = Header(default=None),
         x_api_key: str | None = Header(default=None)) -> None:
    """可选 Bearer / X-API-Key 鉴权。

    API_KEYS 为空表示不鉴权（本地开发）；非空时必须匹配其中一个。
    """
    if not API_KEYS:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="无效或缺失的 API Key")


# ---------------- 请求 / 响应模型（Pydantic 校验） ----------------
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    history: list[Message] = Field(default_factory=list, description="多轮历史（用于指代消解）")
    mode: str = Field(default=RETRIEVAL_MODE, pattern="^(vector|keyword|hybrid)$")
    k: int = Field(default=FINAL_TOP_K, ge=1, le=20, description="送入大模型的片段数")
    model: str | None = Field(default=None, description="模型标识，支持 模型名@端点别名")
    use_cache: bool = Field(default=True, description="是否允许命中 Redis 问答缓存")
    return_hits: bool = Field(default=True, description="是否返回命中的片段明细（溯源）")


class Source(BaseModel):
    doc_name: str
    section_path: str = ""
    page: int = -1
    location: str = ""
    score: float = 0.0
    rerank_score: float | None = None
    has_table: bool = False
    sources: list[str] = Field(default_factory=list)
    snippet: str = ""


class AskResponse(BaseModel):
    answer: str
    question: str
    query_used: str = ""
    rewrite_method: str = ""
    mode: str = ""
    citations: list[int] = Field(default_factory=list)
    forged_citations: list[int] = Field(default_factory=list)
    citation_warning: str = ""
    latency: float = 0.0
    retrieval_latency: float = 0.0
    cache_hit: bool = False
    tokens: dict = Field(default_factory=dict)
    # sources 是内部的 Hit 对象列表（用于构造上下文），类型不定；
    # 对外开放的溯源信息走 hits 字段。这里不声明严格类型，避免响应校验报 500。
    sources: list = Field(default_factory=list)
    hits: list[Source] = Field(default_factory=list)


class IngestResponse(BaseModel):
    docs: dict
    skipped: list[str]
    new_chunks: int
    collection_count: int
    bm25_chunks: int
    elapsed: float
    parse_elapsed: float = 0.0
    async_parse: bool = False
    cache_version: int = 0


# ---------------- 基础端点 ----------------
@app.get("/health", summary="健康检查")
def health():
    r = STATE["retriever"]
    count = None
    try:
        count = r.collection.count() if r else None
    except Exception: # noqa: BLE001
        pass
    return {
        "status": "ok" if r is not None else "degraded",
        "collection": COLLECTION_NAME,
        "chunks": count,
        "bm25_ready": bool(r and r.bm25 is not None),
        "cache": qa_cache.stats(),
        "uptime_s": round(time.time() - STATE["started_at"], 1),
    }


@app.get("/models", summary="可用模型清单")
def models():
    return {"models": MODEL_OPTIONS}


@app.get("/cache/stats", summary="缓存命中统计")
def cache_stats():
    return qa_cache.stats()


@app.post("/cache/clear", summary="清空问答缓存", dependencies=[Depends(auth)])
def cache_clear():
    return {"removed": qa_cache.clear(), "stats": qa_cache.stats()}


# ---------------- 问答 ----------------
@app.post("/ask", response_model=AskResponse, summary="RAG 问答（含答案溯源）")
def ask(req: AskRequest, _: None = Depends(auth)):
    r = STATE["retriever"]
    if r is None:
        raise HTTPException(status_code=503, detail="检索器未就绪：请先执行入库（POST /ingest）")
    try:
        result = chat(
            req.question,
            history=[m.model_dump() for m in req.history],
            retriever=r,
            mode=req.mode,
            k_final=req.k,
            model=req.model,
            return_hits=req.return_hits,
            use_cache=req.use_cache,
        )
    except InputBlocked as e:
        raise HTTPException(status_code=400, detail=f"输入被安全策略拦截：{e}") from e
    # sources 内部是 Hit 对象（供上层构造上下文），对外统一转成可序列化的结构
    result["sources"] = [_hit_to_dict(h) for h in (result.get("sources") or [])]
    return result


@app.post("/ask/stream", summary="RAG 问答（SSE 流式）")
def ask_stream(req: AskRequest, _: None = Depends(auth)):
    """以 SSE 逐事件推送：先发溯源片段，再发答案，最后发统计。

    说明：当前生成端是一次性返回的，因此 answer 事件为单块；
    接入支持流式的 chat_stream 后，无需改动本端点协议即可逐 token 推送。
    """
    import json

    r = STATE["retriever"]
    if r is None:
        raise HTTPException(status_code=503, detail="检索器未就绪")

    def gen():
        try:
            result = chat(
                req.question,
                history=[m.model_dump() for m in req.history],
                retriever=r, mode=req.mode, k_final=req.k,
                model=req.model, return_hits=True, use_cache=req.use_cache,
            )
        except InputBlocked as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)}, ensure_ascii=False)}\n\n"
            return
        except Exception as e: # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'detail': str(e)}, ensure_ascii=False)}\n\n"
            return
        for h in result.get("hits", []):
            yield f"event: source\ndata: {json.dumps(h, ensure_ascii=False)}\n\n"
        yield ("event: answer\ndata: "
               + json.dumps({"answer": result["answer"], "citations": result.get("citations", []),
                             "cache_hit": result.get("cache_hit", False)}, ensure_ascii=False)
               + "\n\n")
        yield ("event: done\ndata: "
               + json.dumps({"latency": result.get("latency", 0.0),
                             "tokens": result.get("tokens", {})}, ensure_ascii=False)
               + "\n\n")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------- 入库 ----------------
@app.post("/ingest", response_model=IngestResponse, summary="上传并入库文档（异步并发解析）",
          dependencies=[Depends(auth)])
def ingest_upload(files: list[UploadFile] = File(..., description="待入库文档"),
                  incremental: bool = Query(default=True),
                  workers: int = Query(default=4, ge=1, le=32)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400,
                                detail=f"不支持的文件类型 {suffix}，支持 {sorted(SUPPORTED_EXTS)}")
        # 只取文件名，防路径穿越（../）
        dest = UPLOAD_DIR / Path(f.filename or "upload").name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)

    stats = ingest(saved, index_dir=INDEX_DIR, collection_name=COLLECTION_NAME,
                   incremental=incremental, workers=workers, quiet=True)
    # 入库会重建 BM25 索引，检索器需重新加载才能看到新块
    STATE["retriever"] = load_retriever(INDEX_DIR, COLLECTION_NAME)
    return stats


@app.post("/ingest/path", response_model=IngestResponse, summary="按服务器路径入库",
          dependencies=[Depends(auth)])
def ingest_path(paths: list[str] = Query(..., description="文件或目录路径"),
                incremental: bool = Query(default=True),
                workers: int = Query(default=4, ge=1, le=32)):
    if not paths:
        raise HTTPException(status_code=400, detail="paths 不能为空")
    stats = ingest(paths, index_dir=INDEX_DIR, collection_name=COLLECTION_NAME,
                   incremental=incremental, workers=workers, quiet=True)
    STATE["retriever"] = load_retriever(INDEX_DIR, COLLECTION_NAME)
    return stats


if __name__ == "__main__":
    import uvicorn

    from rag.config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)
