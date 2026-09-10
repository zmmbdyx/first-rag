"""消息（Message）数据模型。

一条消息 = 一次 user 提问或一次 assistant 回答。
assistant 消息额外持久化**引用来源**、思考过程与耗时/token 统计，
这样刷新页面重放历史时，引用卡片与元信息都能完整还原。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.conversation import utcnow
from backend.models.database import Base

if TYPE_CHECKING:
    from backend.models.conversation import Conversation


class Message(Base):
    """会话中的单条消息。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant

    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # 思考过程（模型支持 reasoning_content 时才有）
    reasoning: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # 引用来源：[{doc_name, section_path, page, score, snippet, location, has_table, sources}]
    sources: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # ---- 运行时元信息（回答质量与性能排查用）----
    query_used: Mapped[str] = mapped_column(Text, default="", nullable=False)  # 改写后的检索查询
    rewrite_method: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    model: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    latency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    retrieval_latency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    ttft: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 首字耗时
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Message {self.id} {self.role} len={len(self.content)}>"
