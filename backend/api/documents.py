"""知识库文档管理接口：清单 / 改密级 / 删除。

为什么单独一组接口
------------------
* **合规**：此前没有删除文档的 API，无法响应"删除某份文档及其派生的全部向量"
  这类要求（`vector_store.delete_doc` 存在但没有暴露）；
* **权限最小化**：改密级与删除是**管理动作**，与问答密钥分开校验
  （``ADMIN_KEYS``），避免把服务密钥发给前端后连带给了删库能力。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import require_admin_key, require_api_key
from backend.core.vectorstore import get_retriever, reload_retriever
from rag import vector_store

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 清单允许普通鉴权用户查看（前端要展示"知识库有哪些文档"），
# 但改密级/删除必须管理员密钥。
_read_deps = [Depends(require_api_key)]
_write_deps = [Depends(require_api_key), Depends(require_admin_key)]


class DocTagsUpdate(BaseModel):
    tags: list[str] = Field(..., description="新的权限标签，如 ['hr','admin']；空列表表示仅管理员可见")


class BackfillRequest(BaseModel):
    tags: list[str] = Field(
        default_factory=list,
        description="给「无标签」的历史文档补的标签；留空用 RAG_ACL_DEFAULT_TAGS",
    )
    dry_run: bool = Field(default=True, description="True 只报告将要影响的文档，不实际写入")


def _collection():
    r = get_retriever()
    if r is None:
        raise HTTPException(status_code=503, detail="检索器未就绪：请先入库文档")
    return r.collection


def _invalidate_caches() -> int:
    """让问答缓存失效，并返回新的知识库版本号。

    **这是删除/改密级后必须做的一步**：问答缓存（Redis）的键里嵌了知识库
    版本号，而版本号此前只在 ``ingest()`` 里递增。于是删除文档后：
    索引里已经没有该文档，但**缓存仍会返回删除前生成的答案**，回答里继续
    引用一份"已删除"的文档 —— 表现为删除操作形同虚设（实测踩到过）。
    改密级同理：旧答案是按旧权限检索出来的，不能继续复用。

    递增版本号会让所有旧键自然失效（键名不同），不需要遍历删除。
    """
    from rag import cache as qa_cache

    try:
        return int(qa_cache.bump_kb_version())
    except Exception:  # noqa: BLE001 — Redis 不可用时不影响主流程
        return 0


def _resolve_doc_name(collection, name: str) -> str | None:
    """把用户给的文档名解析成库里实际存储的名字。

    上传时落盘文件名会带 ``时间戳_随机串_`` 前缀（防同名覆盖），检索侧展示时
    会剥掉这层前缀（见 ``backend.core.rag_chain.display_name``）。因此用户从
    引用卡片上复制到的通常是**展示名**，而库里存的是带前缀的名字 ——
    这里做一次容错解析，否则"删除文档"这类操作会莫名其妙 404。
    """
    from backend.core.rag_chain import display_name

    for row in vector_store.doc_catalog(collection):
        stored = row["doc_name"]
        if stored == name or display_name(stored) == name:
            return stored
    return None


@router.get("", summary="文档清单（含权限标签与块数）", dependencies=_read_deps)
async def list_documents() -> dict:
    """列出知识库中的文档及其密级，便于审计"哪些文档对谁可见"。"""

    def _run() -> tuple[list[dict], list[dict]]:
        collection = _collection()
        return vector_store.doc_catalog(collection), vector_store.docs_without_tags(collection)

    docs, untagged = await asyncio.to_thread(_run)
    return {
        "count": len(docs),
        "documents": docs,
        # 未归类文档数：开启严格模式前必须处理，否则它们会对普通用户不可见
        "untagged_count": len(untagged),
        "untagged_documents": [d["doc_name"] for d in untagged][:50],
    }


@router.post("/backfill-tags", summary="给无标签的历史文档补默认密级（需管理员密钥）",
             dependencies=_write_deps)
async def backfill_tags(payload: BackfillRequest) -> dict:
    """一次性迁移：把升级前入库（无 perm_tags）的 chunk 纳入 ACL 体系。

    ⚠️ 默认 ``dry_run=True``：**先看清会影响哪些文档再执行**。无脑补成 public
    会把本该保密的文档变成公开。历史文档的正确做法是先用本接口 dry-run 列出，
    再用 ``PUT /api/documents/{name}/tags`` 逐份归类。
    """

    def _run() -> tuple[int, int, list[str]]:
        collection = _collection()
        pending = vector_store.docs_without_tags(collection)
        names = [d["doc_name"] for d in pending]
        if payload.dry_run or not names:
            return 0, 0, names
        docs, chunks = vector_store.backfill_missing_tags(collection, payload.tags or None)
        return docs, chunks, names

    docs, chunks, names = await asyncio.to_thread(_run)
    if not payload.dry_run and chunks:
        await asyncio.to_thread(reload_retriever, False)
    return {
        "dry_run": payload.dry_run,
        "affected_documents": len(names),
        "documents": names[:100],
        "updated_documents": docs,
        "updated_chunks": chunks,
        "note": ("这是 dry-run，未写入。确认密级无误后以 dry_run=false 再调一次。"
                 if payload.dry_run else "已写入并刷新检索器。"),
    }


@router.put("/{doc_name:path}/tags", summary="修改文档权限标签（需管理员密钥）",
            dependencies=_write_deps)
async def update_doc_tags(doc_name: str, payload: DocTagsUpdate) -> dict:
    """改密级：更新该文档全部 chunk 的 ``perm_tags``。

    支持用**展示名**（引用卡片上的名字）或库内存储名，自动解析。
    """

    def _run() -> tuple[int, str]:
        collection = _collection()
        resolved = _resolve_doc_name(collection, doc_name)
        if resolved is None:
            return -1, ""
        return vector_store.set_doc_tags(collection, resolved, payload.tags), resolved

    n, resolved = await asyncio.to_thread(_run)
    if n < 0:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_name}")
    await asyncio.to_thread(reload_retriever, False)
    version = await asyncio.to_thread(_invalidate_caches)
    return {"ok": True, "doc_name": resolved, "updated_chunks": n, "tags": payload.tags,
            "cache_version": version}


@router.delete("/{doc_name:path}", summary="删除文档及其全部向量（需管理员密钥）",
               dependencies=_write_deps)
async def delete_document(doc_name: str) -> dict:
    """彻底删除一份文档：向量、元数据与 BM25 索引项。

    这是"被遗忘权"的落地路径：调用后该文档不再出现在任何检索结果中。
    支持用展示名或库内存储名。注意：**不会**回溯修改历史会话里已引用该文档
    的回答文本（那是当时的真实记录）；如需一并清理，请删除对应会话。
    """

    def _run() -> tuple[int, int, str]:
        collection = _collection()
        resolved = _resolve_doc_name(collection, doc_name)
        if resolved is None:
            return -1, 0, ""
        removed = int(vector_store.doc_stats(collection).get(resolved, 0))
        vector_store.delete_doc(collection, resolved)
        return removed, collection.count(), resolved

    removed, remaining, resolved = await asyncio.to_thread(_run)
    if removed < 0:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_name}")
    await asyncio.to_thread(reload_retriever, False)
    # 必须让问答缓存失效：否则删除后仍会返回"删前生成、还在引用该文档"的答案
    version = await asyncio.to_thread(_invalidate_caches)
    return {"ok": True, "doc_name": resolved, "removed_chunks": removed,
            "collection_count": remaining, "cache_version": version}
