***REMOVED*** 大规模入库与查询基准

- 规模：1000 篇文档 / 5000 个 chunk（纯 CPU）
- 首次入库：**255.3s**（20 块/s，含解析/切分/向量编码/双索引构建）
- 增量入库（MD5 未变）：**1.25s**，跳过 1000/1000 篇
- 检索延迟（向量+BM25+RRF，不含 LLM）：**P50 84ms / P95 109ms**
- 进程内存：基线 118MB → 入库后 957MB → 检索后 951MB

> 环境：本地 CPU（无 GPU），嵌入模型 text2vec-base-chinese。
> 10 万块级扩展方案：Chroma 按文档哈希分片多集合，或迁移 Milvus（scripts/migrate_to_milvus.py）。