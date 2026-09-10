"""会话管理接口的请求 / 响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.common import SourceItem


class ConversationCreate(BaseModel):
    """POST /api/conversations 请求体。"""

    title: str | None = Field(default=None, max_length=200, description="可选标题，默认「新对话」")
    model: str | None = Field(default=None, description="本会话默认模型")


class ConversationOut(BaseModel):
    """会话概要。``conversation_id`` 与文档约定的字段名保持一致。"""

    model_config = ConfigDict(from_attributes=True)

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationRename(BaseModel):
    """PUT /api/conversations/{id}/title 请求体。"""

    title: str = Field(..., min_length=1, max_length=200)


class MessageOut(BaseModel):
    """GET /api/conversations/{id}/messages 的单条消息。"""

    id: int | None = None
    role: str
    content: str
    reasoning: str = ""
    sources: list[SourceItem] = Field(default_factory=list)
    created_at: datetime
    model: str = ""
    latency: float = 0.0
    retrieval_latency: float = 0.0
    ttft: float = 0.0
    cache_hit: bool = False
    error: str = ""
