"""文档上传接口的响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """POST /api/upload 响应。

    ``status``：``done`` 已入库可检索 ｜ ``processing`` 已保存但未入库
    （``INGEST_ON_UPLOAD=false``）｜ ``failed`` 入库失败（``message`` 给出原因）。
    """

    file_id: str
    filename: str
    status: str = Field(description="done | processing | failed")
    size: int = Field(default=0, description="文件字节数")
    chunks: int = Field(default=0, description="新增切片数")
    collection_count: int = Field(default=0, description="向量库当前总切片数")
    message: str = ""
