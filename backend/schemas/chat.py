"""对话接口的请求 / 响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。

    ``conversation_id`` 必填但允许是**尚未存在**的 ID —— 前端"新建对话"时先在本地
    生成 ID 并发起首条消息，后端若查不到就按该 ID 落库，省掉一次建会话往返。
    """

    message: str = Field(default="", max_length=4000, description="用户消息；regenerate=true 时可省略")
    conversation_id: str = Field(..., min_length=1, max_length=32, description="会话 ID")
    model: str | None = Field(default=None, description="模型标识，支持「模型名@端点别名」")
    mode: str | None = Field(default=None, pattern="^(vector|keyword|hybrid)$", description="检索模式")
    top_k: int | None = Field(default=None, ge=1, le=20, description="送入大模型的片段数")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking: bool | None = Field(default=None, description="是否开启思考模式（模型支持时）")
    use_cache: bool = Field(default=True, description="是否允许命中 Redis 问答缓存")
    regenerate: bool = Field(
        default=False,
        description=(
            "重新生成最后一条回答：复用最近一条用户提问重跑，"
            "并**覆盖**该提问之后的消息，不在历史里留下重复的用户气泡。"
            "为 true 时 `message` 可省略。"
        ),
    )


class ChatDone(BaseModel):
    """SSE ``done`` 事件的负载：一次问答的收尾统计与元信息。"""

    message_id: int | None = None
    conversation_id: str
    title: str = ""
    latency: float = Field(default=0.0, description="总耗时（秒）")
    retrieval_latency: float = Field(default=0.0, description="检索耗时（秒）")
    ttft: float = Field(default=0.0, description="首字耗时（秒）")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit: bool = False
    citations: list[int] = Field(default_factory=list, description="答案中真实存在的引用编号")
    forged_citations: list[int] = Field(default_factory=list, description="答案中伪造的引用编号")
    citation_warning: str = ""
    query_used: str = Field(default="", description="改写后实际用于检索的查询")
    rewrite_method: str = ""
