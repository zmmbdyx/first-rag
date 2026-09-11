"""回答反馈接口：把前端 👍/👎 落库，作为**在线质量信号**。

为什么需要
----------
改造前，前端操作栏的赞/踩只是本地组件状态（`useState`），刷新即丢 ——
线上表现完全没有回流通道。离线评测集再大，也覆盖不到真实用户的提问分布；
差评样本是发现"评测集没覆盖到的失败模式"最廉价的手段。

设计取舍
--------
* 按 ``request_id`` 关联到 ``chat_requests`` 明细行，从而把差评与
  当次的检索片段、置信度、延迟关联起来，可直接重放排障；
* 没有 request_id 时也允许记录（只存会话与消息），不因为缺字段而丢弃信号；
* 幂等：同一 request_id 重复提交按最后一次覆盖。
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.api.deps import UserContext, get_user_context, require_api_key
from backend.config import PROJECT_ROOT

router = APIRouter(prefix="/api", tags=["feedback"])

FEEDBACK_LOG = PROJECT_ROOT / "logs" / "feedback.jsonl"


class FeedbackRequest(BaseModel):
    vote: str = Field(..., pattern="^(up|down)$", description="up=点赞，down=点踩")
    request_id: str = Field(default="", max_length=64, description="来自 SSE done 事件")
    conversation_id: str = Field(default="", max_length=32)
    message_id: int | None = None
    comment: str = Field(default="", max_length=500, description="可选：用户补充说明")


@router.post("/feedback", summary="回答反馈（👍/👎）", dependencies=[Depends(require_api_key)])
async def submit_feedback(
    payload: FeedbackRequest, user: UserContext = Depends(get_user_context)
) -> dict:
    """记录一条反馈。写入 ``logs/feedback.jsonl``（已 gitignore）。

    用 JSONL 而不是入库，是刻意的：反馈量小、结构可能变化，且它属于
    "运营分析"而非"业务数据"，不值得为它增加一张表与迁移成本。
    后续接 BI 或回流评测集时直接读文件即可。
    """
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vote": payload.vote,
        "request_id": payload.request_id,
        "conversation_id": payload.conversation_id,
        "message_id": payload.message_id,
        "comment": payload.comment,
        # 只存哈希，不留明文身份
        "user_hash": user.user_hash,
    }
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        ok = True
    except OSError:
        ok = False
    return {"ok": ok, "vote": payload.vote}
