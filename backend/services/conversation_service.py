"""会话与消息的持久化服务。

**为什么自己管理消息历史而不是用 ``ConversationBufferMemory``**：
对话要能在刷新页面、换设备后完整还原（含引用来源与耗时），必须落库；
LangChain 的内存对象活不过进程重启，且它只保留纯文本，丢掉 sources。
因此这里把历史存进数据库，每轮按"最近 N 轮"取出来喂给模型。
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Conversation, Message
from backend.schemas import SourceItem

# 去掉标题里的 Markdown 前缀符号，避免侧边栏出现 "## 标题"
_TITLE_NOISE = re.compile(r"^[#>\-*\s`]+")


def new_conversation_id() -> str:
    """生成前端可直接使用的会话 ID（32 位十六进制）。"""
    return uuid.uuid4().hex


def make_title(text: str, max_len: int | None = None) -> str:
    """从首条用户消息生成标题：去 Markdown 噪声 + 折叠空白 + 截断。"""
    limit = max_len or settings.title_max_len
    clean = _TITLE_NOISE.sub("", (text or "").strip())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return "新对话"
    return clean[:limit] + ("…" if len(clean) > limit else "")


# ---------------------------------------------------------------- 会话
def create_conversation(
    db: Session,
    title: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
) -> Conversation:
    """新建会话。``conversation_id`` 可由前端指定（便于先发消息后落库）。"""
    conv = Conversation(
        title=(title or "新对话").strip()[:200] or "新对话",
        title_locked=bool(title),  # 用户显式给了标题就算锁定，不再自动改名
        model=model,
    )
    if conversation_id:
        conv.id = conversation_id[:32]
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversation(db: Session, conversation_id: str) -> Conversation | None:
    return db.get(Conversation, conversation_id)


def get_or_create_conversation(
    db: Session, conversation_id: str, model: str | None = None
) -> Conversation:
    """取会话，不存在则按给定 ID 创建。

    前端"新建对话"时本地先生成 ID 就直接发消息，这里兜底落库，
    省掉一次「先建会话再发消息」的往返。
    """
    conv = get_conversation(db, conversation_id)
    if conv is None:
        conv = create_conversation(db, model=model, conversation_id=conversation_id)
    elif model and not conv.model:
        conv.model = model
        db.commit()
    return conv


def list_conversations(db: Session, limit: int = 200, offset: int = 0) -> list[Conversation]:
    """按更新时间倒序返回会话列表（最近的排最前）。"""
    stmt = (
        select(Conversation)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).all())


def rename_conversation(db: Session, conversation_id: str, title: str) -> Conversation | None:
    """重命名会话；标记为已锁定，之后不再被自动标题覆盖。"""
    conv = get_conversation(db, conversation_id)
    if conv is None:
        return None
    conv.title = title.strip()[:200] or conv.title
    conv.title_locked = True
    db.commit()
    db.refresh(conv)
    return conv


def delete_conversation(db: Session, conversation_id: str) -> bool:
    """删除会话（消息由 cascade 一并删除）。"""
    conv = get_conversation(db, conversation_id)
    if conv is None:
        return False
    db.delete(conv)
    db.commit()
    return True


def count_messages(db: Session, conversation_id: str) -> int:
    """该会话当前实际消息数（走 conversation_id 索引）。"""
    return int(
        db.scalar(select(func.count(Message.id)).where(Message.conversation_id == conversation_id))
        or 0
    )


def touch_conversation(db: Session, conv: Conversation | None) -> None:
    """刷新会话的更新时间与消息计数（列表排序与侧边栏计数依赖它）。"""
    if conv is None:
        return
    conv.message_count = count_messages(db, conv.id)
    db.add(conv)
    db.commit()


# ---------------------------------------------------------------- 消息
def list_messages(db: Session, conversation_id: str) -> list[Message]:
    """按时间正序返回某会话的全部消息（前端按此顺序渲染）。"""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    return list(db.scalars(stmt).all())


def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    *,
    reasoning: str = "",
    sources: list[SourceItem] | None = None,
    query_used: str = "",
    rewrite_method: str = "",
    model: str = "",
    latency: float = 0.0,
    retrieval_latency: float = 0.0,
    ttft: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_hit: bool = False,
    error: str = "",
) -> Message:
    """写入一条消息，并同步刷新所属会话的计数与更新时间。"""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content or "",
        reasoning=reasoning or "",
        sources=[s.model_dump() for s in (sources or [])],
        query_used=query_used or "",
        rewrite_method=rewrite_method or "",
        model=model or "",
        latency=latency or 0.0,
        retrieval_latency=retrieval_latency or 0.0,
        ttft=ttft or 0.0,
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        cache_hit=bool(cache_hit),
        error=error or "",
    )
    db.add(msg)
    # 显式推进 updated_at：SQLAlchemy 的 onupdate 只在字段真的变化时触发，
    # 而这里改的是 message_count，靠它带动时间戳不可靠。
    conv = get_conversation(db, conversation_id)
    if conv is not None:
        from backend.models.conversation import utcnow

        conv.updated_at = utcnow()
        # 首条用户消息 + 未锁定时自动生成标题
        if role == "user" and not conv.title_locked and (conv.title in ("", "新对话")):
            conv.title = make_title(content)
    db.commit()
    db.refresh(msg)

    # message_count 一律按实际行数重算，不做 +1 累加：累加式计数在
    # 「重新生成」这类先删后增的流程里必然漂移（实测出现过库里 2 条、
    # 计数显示 3 条）。COUNT 走 conversation_id 索引，代价可忽略。
    touch_conversation(db, get_conversation(db, conversation_id))
    return msg


def recent_history(db: Session, conversation_id: str, rounds: int | None = None) -> list[dict]:
    """取最近 N 轮对话（一问一答算一轮），用于多轮改写与生成。

    返回 ``[{"role": ..., "content": ...}]``，按时间正序。只取 role/content，
    不带 sources —— 历史里塞引用片段会白白烧 token。
    """
    n = settings.history_rounds if rounds is None else rounds
    if n <= 0:
        return []
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role.in_(("user", "assistant")))
        .order_by(desc(Message.created_at), desc(Message.id))
        .limit(n * 2)
    )
    rows = list(db.scalars(stmt).all())
    rows.reverse()
    return [
        {"role": m.role, "content": m.content}
        for m in rows
        if (m.content or "").strip()
    ]


def clear_all(db: Session) -> int:
    """清空所有会话（开发/演示用）。"""
    n = int(db.scalar(select(func.count(Conversation.id))) or 0)
    db.execute(delete(Message))
    db.execute(delete(Conversation))
    db.commit()
    return n


# ---------------------------------------------------------------- 重新生成
def prepare_regenerate(db: Session, conversation_id: str) -> str | None:
    """为「重新生成」做准备：取回最近一次用户提问，并删除它之后的消息。

    返回要重跑的提问文本；没有可重跑的提问时返回 None。

    为什么要在后端做：一次问答在库里是 (user, assistant) 两条消息，
    「重新生成」语义上是**替换**这一轮的答案，而不是再追加一轮。
    若让前端直接重发同一条消息，历史里就会出现两条一模一样的用户提问，
    刷新后看起来像问了两次。
    """
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "user")
        .order_by(desc(Message.created_at), desc(Message.id))
        .limit(1)
    )
    last_user = db.scalar(stmt)
    if last_user is None:
        return None

    keep_id = int(last_user.id)
    text = last_user.content or ""

    # 删掉这条提问之后的所有消息（即上一轮的回答，可能还包含更早的中断残留）。
    # 用 ORM 逐个删除而不是 Core 批量 delete：这些实例已加载进 Session，
    # 批量删除会让它们的状态与数据库不一致。
    stale = db.scalars(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id > keep_id,
        )
    ).all()
    for m in stale:
        db.delete(m)

    # message_count 是缓存字段，删除后必须重算，否则侧边栏计数虚高
    conv = get_conversation(db, conversation_id)
    if conv is not None:
        from backend.models.conversation import utcnow

        conv.message_count = count_messages(db, conversation_id)
        conv.updated_at = utcnow()
    db.commit()

    # 关键：上面删掉的实例仍被 Session 的 identity map 持有，commit 默认
    # 不刷新已加载对象（expire_on_commit=False）。若不清空，紧接着的
    # recent_history() 查询重新取出这些行时会命中"已被删除"的实例，抛
    # ObjectDeletedError（表现为"会话初始化失败：Instance has been deleted"）。
    db.expire_all()
    return text
