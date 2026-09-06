"""面向企业知识库的 RAG 核心包。

模块职责：
- parsers      多格式文档解析（PDF/Word/TXT/MD）→ 结构化 Block
- chunking     结构感知智能切分 + 朴素切分（评测基线）
- embeddings   本地句向量模型封装
- vector_store Chroma 向量库封装
- bm25         BM25 关键词索引（jieba 分词）
- retriever    向量/关键词/混合（RRF 融合）检索
- llm          大模型调用：带引用的答案生成 + 评测判分
- pipeline     入库与问答编排（答案溯源出口）
"""

__version__ = "1.0.0"
