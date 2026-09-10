"""知识库入库 CLI：解析 → 智能切分 → 向量化 → Chroma + BM25 索引。

用法：
  python scripts/build_kb.py # 入库 data/samples 下全部示例文档
  python scripts/build_kb.py D:\\docs other.pdf # 入库指定文件/目录
  python scripts/build_kb.py --rebuild # 清空后重建
  python scripts/build_kb.py --naive # 用朴素切分（评测基线用）
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import COLLECTION_NAME, INDEX_DIR, SAMPLES_DIR # noqa: E402
from rag.pipeline import ingest # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RAG 知识库入库")
    ap.add_argument("paths", nargs="*", default=None, help="文件或目录（默认 data/samples）")
    ap.add_argument("--rebuild", action="store_true", help="清空集合后重建")
    ap.add_argument("--naive", action="store_true", help="朴素固定窗口切分（评测基线）")
    ap.add_argument("--collection", default=COLLECTION_NAME, help="Chroma 集合名")
    args = ap.parse_args()

    paths = args.paths or [str(SAMPLES_DIR)]
    index_dir = Path(INDEX_DIR)

    if args.rebuild:
        from rag import vector_store

        client = vector_store.get_client(index_dir)
        try:
            client.delete_collection(args.collection)
            print(f"🗑️  已清空集合 {args.collection}")
        except Exception: # noqa: BLE001
            pass

    print(f"📥 入库 → {index_dir}（集合 {args.collection}，切分={'朴素' if args.naive else '智能'}）")
    stats = ingest(paths, index_dir=index_dir, collection_name=args.collection,
                   chunker="naive" if args.naive else "smart")
    print("\n各文档 chunk 数：")
    for doc, n in stats["docs"].items():
        print(f"  - {doc}: {n}")


if __name__ == "__main__":
    main()
