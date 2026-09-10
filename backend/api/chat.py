"""对话接口：``POST /api/chat``（SSE 流式输出）。

SSE 事件协议（前端按此解析）
---------------------------
::

    event: conversation   data: {"conversation_id": "...", "title": "..."}
    event: rewrite        data: {"query": "...", "method": "llm"}
    event: sources        data: [SourceItem, ...]
    event: status         data: {"stage": "generating"}
    event: reasoning      data: {"delta": "..."}      # 思考过程增量
    event: content        data: {"delta": "..."}      # 正文增量（逐 token）
    event: done           data: {ChatDone}
    event: error          data: {"detail": "...", "code": "..."}

每个 ``data`` 都是单行 JSON（``ensure_ascii=False``，中文不转义）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.deps import require_api_key
from backend.config import settings
from backend.core.rag_chain import ChatParams, stream_rag
from backend.models import session_scope
from backend.schemas import ChatRequest, SourceItem
from backend.services import conversation_service as cs

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，否则流式会被攒成一坨
}


def _sse(event: str, data: object) -> str:
    """按 SSE 规范拼一帧：单行 JSON，避免多行 data 被前端拼接出错。"""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def _generate(req: ChatRequest) -> AsyncIterator[str]:
    """SSE 事件生成器：负责持久化与事件封装，RAG 编排交给 rag_chain。"""
    # ---- 1) 会话与历史（放进线程：SQLAlchemy 是同步的）----
    def _prepare() -> tuple[str, list[dict], str, str | None, str]:
        """返回 (会话ID, 历史, 标题, 会话模型, 本轮实际提问)。"""
        with session_scope() as db:
            conv = cs.get_or_create_conversation(db, req.conversation_id, model=req.model)

            if req.regenerate:
                # 重新生成：复用最近一条用户提问，并删除它之后的回答（覆盖而非追加）
                message = cs.prepare_regenerate(db, conv.id)
                if message is None:
                    raise ValueError("没有可重新生成的提问")
                history = cs.recent_history(db, conv.id)
                db.refresh(conv)
                return conv.id, history, conv.title, conv.model, message

            history = cs.recent_history(db, conv.id)
            cs.add_message(db, conv.id, "user", req.message)
            db.refresh(conv)
            return conv.id, history, conv.title, conv.model, req.message

    try:
        conv_id, history, title, conv_model, message = await asyncio.to_thread(_prepare)
    except ValueError as e:
        # 没有可重跑的提问属于客户端用法问题
        yield _sse("error", {"detail": str(e), "code": "nothing_to_regenerate"})
        return
    except Exception as e:  # noqa: BLE001
        yield _sse("error", {"detail": f"会话初始化失败：{e}", "code": "db_error"})
        return

    yield _sse("conversation", {"conversation_id": conv_id, "title": title})

    params = ChatParams(
        message=message,
        conversation_id=conv_id,
        model=req.model or conv_model,
        mode=req.mode or "hybrid",
        top_k=req.top_k or settings.default_top_k,
        temperature=req.temperature if req.temperature is not None else 0.3,
        thinking=bool(req.thinking),
        use_cache=req.use_cache,
        history=history,
    )

    # ---- 2) 跑 RAG 链，边收边转发 ----
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    sources: list[SourceItem] = []
    done_payload: dict = {}
    wrote = False

    try:
        async for name, payload in stream_rag(params):
            if name == "sources":
                sources = payload
                yield _sse("sources", [s.model_dump() for s in sources])
            elif name == "rewrite":
                yield _sse("rewrite", payload)
            elif name == "status":
                yield _sse("status", payload)
            elif name == "reasoning":
                reasoning_parts.append(payload)
                yield _sse("reasoning", {"delta": payload})
            elif name == "content":
                answer_parts.append(payload)
                yield _sse("content", {"delta": payload})
            elif name == "done":
                done_payload = payload
            elif name == "error":
                yield _sse("error", payload)
                # 出错也要把用户消息留在会话里，否则历史断档；助手侧记录错误
                await asyncio.to_thread(
                    _persist, conv_id, "", "", sources, payload.get("detail", ""), _meta(params)
                )
                wrote = True
                return
    except asyncio.CancelledError:
        # 客户端断开或前端点了"停止生成"：保存已生成的部分内容后正常收尾
        await asyncio.to_thread(
            _persist, conv_id, "".join(answer_parts), "".join(reasoning_parts),
            sources, "已中止生成", {**_meta(params), "latency": 0.0},
        )
        raise

    # ---- 3) 落库助手回答，并推送收尾事件 ----
    done_payload.setdefault("model", params.model or "")
    if not wrote:
        persisted = await asyncio.to_thread(
            _persist, conv_id, "".join(answer_parts), "".join(reasoning_parts),
            sources, done_payload.get("error", ""), done_payload,
        )
        done_payload["message_id"] = persisted.get("message_id")
        done_payload["title"] = persisted.get("title", title)

    done_payload["conversation_id"] = conv_id
    # answer / reasoning 已经作为增量推过了，收尾事件不再重复整个正文
    yield _sse("done", {k: v for k, v in done_payload.items() if k not in ("answer", "reasoning")})


def _meta(params: ChatParams) -> dict:
    """从请求参数里提取要落库的元信息（失败路径也要记，便于排查）。"""
    return {
        "model": params.model or "",
        "query_used": "",
        "rewrite_method": "",
        "latency": 0.0,
        "retrieval_latency": 0.0,
        "ttft": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_hit": False,
    }


def _persist(
    conv_id: str,
    answer: str,
    reasoning: str,
    sources: list[SourceItem],
    error: str,
    meta: dict,
) -> dict:
    """把助手消息写入数据库，返回新消息 ID 与当前标题。"""
    with session_scope() as db:
        msg = cs.add_message(
            db,
            conv_id,
            "assistant",
            answer,
            reasoning=reasoning,
            sources=sources,
            query_used=meta.get("query_used", ""),
            rewrite_method=meta.get("rewrite_method", ""),
            model=meta.get("model", ""),
            latency=float(meta.get("latency") or 0.0),
            retrieval_latency=float(meta.get("retrieval_latency") or 0.0),
            ttft=float(meta.get("ttft") or 0.0),
            prompt_tokens=int(meta.get("prompt_tokens") or 0),
            completion_tokens=int(meta.get("completion_tokens") or 0),
            cache_hit=bool(meta.get("cache_hit")),
            error=error or "",
        )
        conv = cs.get_conversation(db, conv_id)
        return {"message_id": msg.id, "title": conv.title if conv else ""}


@router.post("/chat", summary="对话（SSE 流式输出）", dependencies=[Depends(require_api_key)])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """接收消息并以 ``text/event-stream`` 逐 token 推送回答与引用来源。

    ``regenerate=true`` 时复用最近一条用户提问重跑，并覆盖它之后的回答
    （而不是在历史里再追加一条相同的提问）。
    """
    if not req.regenerate and not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空（regenerate=true 时除外）")
    return StreamingResponse(_generate(req), media_type="text/event-stream", headers=SSE_HEADERS)
