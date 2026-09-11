"""会话（Conversation）数据模型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.database import Base

if TYPE_CHECKING:  # 仅类型检查期导入，避免循环依赖
    from backend.models.message import Message


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """带时区的当前时间（存库统一 UTC）。"""
    return datetime.now(timezone.utc)


class Conversation(Base):
    """一次多轮对话。标题默认取首条用户消息的前 N 个字符。"""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(200), default="新对话", nullable=False)
    # 会话归属：用于**按用户隔离**。此前没有该字段，GET /api/conversations
    # 会返回所有人的会话 —— 多用户场景下的越权读。
    # 空字符串表示"未启用用户隔离"（本地开发/单用户），此时沿用旧行为放行。
    owner_id: Mapped[str] = mapped_column(String(128), default="", nullable=False, index=True)
    # 是否已由用户手动重命名：重命名后不再自动改标题
    title_locked: Mapped[bool] = mapped_column(default=False, nullable=False)
    model: Mapped[str | None] = mapped_column(String(120), default=None)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False, index=True
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Conversation {self.id} {self.title!r} msgs={self.message_count}>"
