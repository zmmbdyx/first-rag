"""文档上传与入库服务。

安全要点（沿用项目既有做法）：
* **文件名消毒**：只取 ``Path(name).name``，杜绝 ``../../etc/passwd`` 之类的路径穿越；
* **扩展名白名单**：与 ``rag.parsers.SUPPORTED_EXTS`` 保持一致（PDF/Word/TXT/Markdown）；
* **大小限制**：边写边计数，超限立即中断并删除半成品，避免磁盘被写满；
* **存储隔离**：文件落在 ``UPLOAD_DIR``（默认 ``backend/uploads``），该目录已被 .gitignore 排除。

入库本身完全复用 ``rag.pipeline.ingest``（MD5 增量、并发解析、智能切分、Chroma 写入、
BM25 重建），本服务不重复实现任何切分或向量化逻辑。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rag.config import COLLECTION_NAME, INDEX_DIR
from rag.parsers import SUPPORTED_EXTS
from rag.pipeline import ingest

from backend.config import settings
from backend.core.vectorstore import reload_retriever

# 一次读写的分片大小（用于边写边限流，而不是先读完再判断大小）
CHUNK = 1024 * 1024


@dataclass
class UploadResult:
    """单个文件的上传/入库结果。"""

    file_id: str
    filename: str
    status: str  # done | processing | failed
    size: int = 0
    chunks: int = 0
    collection_count: int = 0
    message: str = ""


def sanitize_filename(name: str) -> str:
    """只保留文件名部分，防止路径穿越。"""
    return Path(name or "upload").name or "upload"


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTS


def save_upload(filename: str, stream, max_bytes: int | None = None) -> tuple[Path, str, int]:
    """把上传流写入磁盘，返回 (路径, file_id, 字节数)。超限抛 ValueError。"""
    limit = (max_bytes if max_bytes is not None else settings.max_upload_mb * 1024 * 1024)
    upload_dir = settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    safe = sanitize_filename(filename)
    # 加时间戳前缀：同名文件重复上传不互相覆盖，也便于人工排查
    dest = upload_dir / f"{int(time.time())}_{file_id[:8]}_{safe}"

    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                buf = stream.read(CHUNK)
                if not buf:
                    break
                size += len(buf)
                if size > limit:
                    raise ValueError(f"文件超过大小上限 {limit // (1024 * 1024)}MB")
                out.write(buf)
    except Exception:
        dest.unlink(missing_ok=True)  # 清理半成品
        raise
    return dest, file_id, size


def ingest_file(path: Path) -> dict:
    """把已落盘的文件向量化入库，并让常驻检索器重新加载。"""
    stats = ingest(
        [path],
        index_dir=INDEX_DIR,
        collection_name=COLLECTION_NAME,
        incremental=True,
        quiet=True,
    )
    # 入库会重建 BM25 索引，常驻检索器必须重建才能看到新块
    reload_retriever(rebuild_bm25_if_empty=False)
    return stats


def process_upload(filename: str, stream) -> UploadResult:
    """完整处理一次上传：校验 → 落盘 → （可选）入库。"""
    safe = sanitize_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        return UploadResult(
            file_id="",
            filename=safe,
            status="failed",
            message=f"不支持的文件类型 {suffix or '(无扩展名)'}，支持：{sorted(SUPPORTED_EXTS)}",
        )

    try:
        path, file_id, size = save_upload(safe, stream)
    except ValueError as e:
        return UploadResult(file_id="", filename=safe, status="failed", message=str(e))

    result = UploadResult(file_id=file_id, filename=safe, status="processing", size=size)

    if not settings.ingest_on_upload:
        result.message = "已保存；INGEST_ON_UPLOAD=false，尚未向量化入库"
        return result

    try:
        stats = ingest_file(path)
        docs = stats.get("docs") or {}
        result.chunks = sum(int(v) for v in docs.values())
        result.collection_count = int(stats.get("collection_count") or 0)
        if result.chunks == 0 and stats.get("skipped"):
            # MD5 命中说明这份文档之前已经入过库，属于正常情况而非失败
            result.status = "done"
            result.message = "内容与已有文档一致（MD5 命中），跳过重复入库"
        else:
            result.status = "done"
            result.message = f"已入库 {result.chunks} 个切片"
    except Exception as e:  # noqa: BLE001 — 入库失败要如实回传，而不是让请求 500
        result.status = "failed"
        result.message = f"入库失败：{e}"
    return result
