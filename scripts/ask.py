"""知识库问答 CLI：检索 → 大模型生成 → 答案溯源。

用法：
  python scripts/ask.py "试用期多长？"
  python scripts/ask.py "年假有几天" --mode vector     ***REMOVED*** 对比纯向量检索
  python scripts/ask.py                                ***REMOVED*** 进入多轮交互
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import COLLECTION_NAME, FINAL_TOP_K, INDEX_DIR, RETRIEVAL_MODE  ***REMOVED*** noqa: E402
from rag.pipeline import ask, load_retriever  ***REMOVED*** noqa: E402


def answer_once(retriever, question: str, mode: str, k: int) -> None:
    result = ask(question, retriever, mode=mode, k_final=k, return_hits=True)
    print(f"\n{'=' * 60}")
    print(f"❓ {question}")
    print(f"{'-' * 60}")
    print(f"💬 {result['answer']}")
    print(f"{'-' * 60}")
    print(f"🔎 检索 {result['retrieval_latency']:.2f}s · 生成 {result['latency']:.2f}s"
          f" · 模式 {mode} · 召回 {len(result['hits'])} 块")
    for i, h in enumerate(result["hits"], 1):
        flag = " ⬅ 引用" if i in result["citations"] else ""
        print(f"  [{i}] {h['location']}  (score={h['score']}, 路={','.join(h['sources'])}){flag}")


def main():
    ap = argparse.ArgumentParser(description="RAG 知识库问答（带答案溯源）")
    ap.add_argument("question", nargs="*", help="问题；不传则进入交互模式")
    ap.add_argument("--mode", default=RETRIEVAL_MODE, choices=["vector", "keyword", "hybrid"])
    ap.add_argument("--k", type=int, default=FINAL_TOP_K, help="召回条数")
    ap.add_argument("--collection", default=COLLECTION_NAME)
    args = ap.parse_args()

    index_dir = Path(INDEX_DIR)
    if not (index_dir.exists() and any(index_dir.iterdir())):
        print("❌ 知识库为空，请先运行: python scripts/build_kb.py")
        sys.exit(1)

    retriever = load_retriever(index_dir, args.collection)
    if args.question:
        answer_once(retriever, " ".join(args.question), args.mode, args.k)
        return

    print("💬 交互模式（q 退出）。问题会基于知识库回答并标注来源。")
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            answer_once(retriever, q, args.mode, args.k)


if __name__ == "__main__":
    main()
