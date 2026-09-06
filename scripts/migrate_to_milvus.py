"""将 Chroma 集合迁移到 Milvus（10 万块级规模方案）。

用法（需先 pip install pymilvus，并部署 Milvus 2.4+/Milvus Lite）：
  python scripts/migrate_to_milvus.py --uri http://localhost:19530 --collection rag_chunks
  python scripts/migrate_to_milvus.py --uri milvus_lite.db   ***REMOVED*** Milvus Lite 本地文件模式

迁移内容：全部 chunk 的 id / 文本 / 元数据 / 向量；Milvus 侧建 Collection：
  id(VARCHAR 主键) · embedding(FLOAT_VECTOR, 余弦) · doc_name/section_path(VARCHAR) · page(INT64)
并在 embedding 字段建 HNSW 索引（M=16, efConstruction=200）。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import COLLECTION_NAME, INDEX_DIR  ***REMOVED*** noqa: E402
from rag import vector_store  ***REMOVED*** noqa: E402

BATCH = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--milvus-collection", default="rag_chunks")
    args = ap.parse_args()

    try:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    except ImportError:
        print("❌ 缺少 pymilvus：pip install pymilvus")
        sys.exit(1)

    client = vector_store.get_client(Path(args.index_dir))
    col = vector_store.get_collection(client, args.collection, create=False)
    if col is None or col.count() == 0:
        print(f"❌ Chroma 集合 {args.collection} 不存在或为空")
        sys.exit(1)

    connections.connect(uri=args.uri)
    if utility.has_collection(args.milvus_collection):
        print(f"删除已存在的集合 {args.milvus_collection}")
        Collection(args.milvus_collection).drop()

    fields = [
        FieldSchema("id", DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
        FieldSchema("doc_name", DataType.VARCHAR, max_length=512),
        FieldSchema("section_path", DataType.VARCHAR, max_length=1024),
        FieldSchema("page", DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="RAG chunks migrated from Chroma")
    milvus_col = Collection(args.milvus_collection, schema)
    dim = 768  ***REMOVED*** text2vec-base-chinese

    got = col.get(include=["documents", "metadatas", "embeddings"])
    n = len(got["ids"])
    print(f"迁移 {n} 条向量 → {args.uri}/{args.milvus_collection}")
    for i in range(0, n, BATCH):
        sl = slice(i, i + BATCH)
        rows = [
            got["ids"][sl],
            [e[:dim] for e in got["embeddings"][sl]],
            [(m or {}).get("doc_name", "") for m in got["metadatas"][sl]],
            [(m or {}).get("section_path", "") for m in got["metadatas"][sl]],
            [int((m or {}).get("page", -1)) for m in got["metadatas"][sl]],
        ]
        milvus_col.insert(rows)
        print(f"  ... {min(i + BATCH, n)}/{n}")

    milvus_col.flush()
    milvus_col.create_index("embedding", index_params={
        "index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
    milvus_col.load()
    print(f"✅ 迁移完成：{milvus_col.num_entities} 条，HNSW(COSINE) 索引已建立")


if __name__ == "__main__":
    main()
