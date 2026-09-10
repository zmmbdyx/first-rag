"""FastAPI 入口：注册路由、CORS、生命周期初始化。

启动：
    uvicorn backend.main:app --reload --port 8000
    # 或
    python -m backend.main
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import chat, conversations, upload
from backend.config import settings
from backend.core.vectorstore import collection_count, reload_retriever, retriever_ready
from backend.models import init_db
from backend.schemas import HealthResponse
from rag import cache as qa_cache
from rag.config import COLLECTION_NAME, MODEL_OPTIONS

STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建表 + 预热检索器（含 BM25），避免首个请求承担冷启动延迟。"""
    init_db()
    print(f"[backend] 会话数据库就绪：{settings.database_url}")

    reload_retriever()
    if retriever_ready():
        print(f"[backend] 检索器就绪：集合 {COLLECTION_NAME}，切片数 {collection_count()}")
    else:
        print("[backend] 检索器未就绪（/api/health 会报告 degraded；上传文档后自动恢复）")

    if qa_cache.available():
        print("[backend] Redis 问答缓存已启用")
    else:
        print("[backend] Redis 问答缓存未启用（自动降级：每次请求走完整链路）")
    yield


app = FastAPI(
    title="企业知识库 RAG 问答 API",
    description=(
        "前后端分离架构的后端服务：FastAPI + 检索增强生成。\n\n"
        "- `POST /api/chat`：SSE 流式问答，逐 token 推送并附带引用来源\n"
        "- `/api/conversations`：会话与消息历史管理\n"
        "- `POST /api/upload`：文档上传并向量化入库\n"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(upload.router)


@app.get("/api/health", response_model=HealthResponse, tags=["system"], summary="健康检查")
def health() -> HealthResponse:
    """服务与知识库状态；检索器不可用时返回 ``degraded`` 而不是直接报错。"""
    ready = retriever_ready()
    return HealthResponse(
        status="ok" if ready else "degraded",
        collection=COLLECTION_NAME,
        chunks=collection_count(),
        bm25_ready=bool(ready),
        retriever_ready=ready,
        models=MODEL_OPTIONS,
        cache=qa_cache.stats(),
        upload_dir=str(settings.upload_path),
    )


@app.get("/", include_in_schema=False)
def index():
    return JSONResponse({
        "service": "rag-backend",
        "docs": "/docs",
        "health": "/api/health",
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
