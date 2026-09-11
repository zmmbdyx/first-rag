"""文档上传接口：``POST /api/upload``。

支持 PDF / Word(.docx) / TXT / Markdown，上传后自动向量化入库并刷新检索器，
因此紧接着的提问就能检索到新文档（与旧版 Streamlit 上传体验一致）。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.api.deps import require_admin_key, require_api_key
from backend.config import settings
from backend.schemas import UploadResponse
from backend.services import upload_service
from rag.parsers import SUPPORTED_EXTS

router = APIRouter(prefix="/api", tags=["upload"], dependencies=[Depends(require_api_key)])


@router.post("/upload", response_model=UploadResponse, summary="上传文档（PDF/Word/TXT/Markdown）")
async def upload_document(
    file: UploadFile = File(..., description="待入库文档"),
    perm_tags: str = Query(
        default="",
        description="文档级权限标签，逗号分隔（如 'hr,admin'）。留空则用 "
                    "RAG_ACL_DEFAULT_TAGS（默认 public）。敏感材料请显式指定，"
                    "否则默认对全体可见。",
    ),
    _admin: None = Depends(require_admin_key),
):
    """保存并向量化一份文档。

    上传会**改变知识库内容**（影响所有人的检索结果），因此按管理动作处理：
    配置了 ``ADMIN_KEYS`` 时要求管理员密钥。

    入库是 CPU 密集且阻塞的（解析 + 切分 + 嵌入 + 写库），放到线程池执行，
    避免卡住事件循环影响其他请求的流式输出。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    tags = [t.strip() for t in perm_tags.split(",") if t.strip()] or None
    result = await asyncio.to_thread(
        upload_service.process_upload, file.filename, file.file, tags
    )

    if result.status == "failed":
        # 文件类型不支持属于客户端错误；入库失败属于服务端问题
        code = 400 if "不支持的文件类型" in result.message or "上限" in result.message else 500
        raise HTTPException(status_code=code, detail=result.message)

    return UploadResponse(
        file_id=result.file_id,
        filename=result.filename,
        status=result.status,
        size=result.size,
        chunks=result.chunks,
        collection_count=result.collection_count,
        message=result.message,
    )


@router.get("/upload/supported", summary="支持的文件类型")
def supported_types() -> dict:
    from rag.config import ACL_DEFAULT_TAGS, ACL_STRICT

    return {
        "extensions": sorted(SUPPORTED_EXTS),
        "max_size_mb": settings.max_upload_mb,
        "ingest_on_upload": settings.ingest_on_upload,
        "default_perm_tags": ACL_DEFAULT_TAGS,
        "acl_strict": ACL_STRICT,
    }
