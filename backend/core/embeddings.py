"""Embedding 模型说明与元信息。

**模型本身不做任何改动** —— 仍由 ``rag.embeddings`` 提供，通过
``rag.vector_store`` 在 Chroma 建集合时使用。本模块只负责：

* 把当前生效的嵌入配置暴露给 API（``/api/health``、前端"关于"面板）；
* 保证"嵌入模型没被这次重构换掉"这件事可被检查 —— 换嵌入模型会让已有
  向量库全部失效，是重构中最容易踩的坑之一。

当前配置（来自环境变量，默认值见 ``rag/config.py``）：
    EMBED_MODEL      默认 shibing624/text2vec-base-chinese，本地运行、无需联网
    INDEX_DIR        向量库目录，默认 <项目根>/chroma_db
    COLLECTION_NAME  集合名，默认 rag_chunks
"""

from __future__ import annotations

from rag.config import COLLECTION_NAME, EMBED_MODEL
from rag.embeddings import embed_texts, embed_query  # noqa: F401  再导出，供上层统一入口

__all__ = ["COLLECTION_NAME", "EMBED_MODEL", "embed_query", "embed_texts", "embedding_info"]


def embedding_info() -> dict:
    """嵌入与向量库的只读元信息（不含任何密钥）。"""
    return {
        "embed_model": EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "note": "本地句向量模型，CPU/CUDA 自适应；未使用任何云端 embedding 接口",
    }
