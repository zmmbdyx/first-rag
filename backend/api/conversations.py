"""会话管理接口：创建 / 列表 / 消息 / 删除 / 重命名。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.models import get_db
from backend.schemas import (
    ConversationCreate,
    ConversationOut,
    ConversationRename,
    MessageOut,
    SourceItem,
)
from backend.services import conversation_service as cs

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _to_out(conv) -> ConversationOut:
    return ConversationOut(
        conversation_id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=int(conv.message_count or 0),
    )


@router.post("", response_model=ConversationOut, status_code=201, summary="创建新对话")
def create_conversation(payload: ConversationCreate | None = None, db: Session = Depends(get_db)):
    """新建一个空会话；标题缺省为「新对话」，首条消息到达时会自动改写。"""
    title = payload.title if payload else None
    model = payload.model if payload else None
    conv = cs.create_conversation(db, title=title, model=model)
    return _to_out(conv)


@router.get("", response_model=list[ConversationOut], summary="获取对话列表")
def list_conversations(
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """按更新时间倒序返回会话列表（最近活动的排最前）。"""
    return [_to_out(c) for c in cs.list_conversations(db, limit=limit, offset=offset)]


@router.get("/{conversation_id}/messages", response_model=list[MessageOut], summary="获取对话的全部消息")
def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    """按时间正序返回全部消息，assistant 消息带引用来源与耗时统计。"""
    conv = cs.get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    out: list[MessageOut] = []
    for m in cs.list_messages(db, conversation_id):
        sources: list[SourceItem] = []
        for s in (m.sources or []):
            try:
                sources.append(SourceItem(**s))
            except Exception:  # noqa: BLE001 — 老数据字段缺失时跳过，不让整个历史 500
                continue
        out.append(MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            reasoning=m.reasoning or "",
            sources=sources,
            created_at=m.created_at,
            model=m.model or "",
            latency=m.latency or 0.0,
            retrieval_latency=m.retrieval_latency or 0.0,
            ttft=m.ttft or 0.0,
            cache_hit=bool(m.cache_hit),
            error=m.error or "",
        ))
    return out


@router.delete("/{conversation_id}", summary="删除对话")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    if not cs.delete_conversation(db, conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "conversation_id": conversation_id}


@router.put("/{conversation_id}/title", response_model=ConversationOut, summary="重命名对话")
def rename_conversation(
    conversation_id: str, payload: ConversationRename, db: Session = Depends(get_db)
):
    conv = cs.rename_conversation(db, conversation_id, payload.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _to_out(conv)
