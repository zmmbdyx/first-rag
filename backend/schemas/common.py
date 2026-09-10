"""跨领域共用的模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.config import settings


class SourceItem(BaseModel):
    """一条引用来源（对应一个检索命中的文档片段）。

    字段与 ``rag.retriever.Hit`` 对齐，另加 ``index``（引用编号，与答案里的 [n] 对应）
    与 ``similarity``（归一化到 0~1，便于前端画相似度条）。
    """

    index: int = Field(default=0, description="引用编号，与答案里的 [n] 对应")
    doc_name: str = Field(default="", description="文档名")
    section_path: str = Field(default="", description="章节路径，如「入职 > 试用期」")
    page: int = Field(default=-1, description="页码，-1 表示无页码信息")
    location: str = Field(default="", description="人类可读定位，如「手册.pdf P3 · 试用期」")
    score: float = Field(default=0.0, description="融合检索分数")
    rerank_score: float | None = Field(default=None, description="重排分数（开启重排时）")
    has_table: bool = Field(default=False, description="是否来自表格块")
    sources: list[str] = Field(default_factory=list, description="命中路径，如 ['vector', 'bm25']")
    snippet: str = Field(default="", description="原文片段")
    similarity: float | None = Field(default=None, description="归一化相似度 0~1，供前端展示")


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    collection: str = ""
    chunks: int | None = None
    bm25_ready: bool = False
    retriever_ready: bool = False
    models: list[str] = Field(default_factory=list)
    cache: dict = Field(default_factory=dict)
    upload_dir: str = ""
    default_top_k: int = settings.default_top_k
    auth_required: bool = Field(
        default=False, description="是否启用了 API Key 鉴权（前端据此提示用户配置密钥）"
    )
