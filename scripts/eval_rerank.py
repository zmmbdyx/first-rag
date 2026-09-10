"""Reranker 开关对比实验（纯检索，无需 LLM）：同一索引上 RERANK on/off 的检索指标对比。

输出 eval/results/rerank_compare.json 与 rerank_compare.md。
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag.config import FINAL_TOP_K  ***REMOVED*** noqa: E402
from rag.retriever import Retriever  ***REMOVED*** noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_eval import eval_retrieval, aggregate_retrieval, load_questions  ***REMOVED*** noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_DIR = EVAL_DIR / ".eval_db" / "rerank_cmp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions_v2.jsonl"))
    args = ap.parse_args()

    from rag.pipeline import ingest

    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        ingest([CORPUS_DIR], index_dir=INDEX_DIR, chunker="smart", quiet=True)

    questions = [q for q in load_questions(Path(args.questions)) if q["answerable"] and not q.get("multi_turn")]
    retriever = Retriever(INDEX_DIR)

    results = {"n_questions": len(questions), "k": FINAL_TOP_K}
    for flag, label in ((False, "rerank_off"), (True, "rerank_on")):
        print(f"== {label} ==")
        t0 = time.time()
        cfg = {"mode": "hybrid", "rewrite": False, "rerank": flag}
        rows = eval_retrieval(questions, retriever, cfg, FINAL_TOP_K, use_llm_rewrite=False)
        elapsed = time.time() - t0
        agg = aggregate_retrieval(rows)
        agg["avg_retrieval_ms"] = round(elapsed / len(questions) * 1000, 1)
        from run_eval import per_qtype_table
        agg["per_qtype"] = per_qtype_table(rows, None)
        results[label] = agg
        print(f"   R@1={agg['answer_recall@1']:.1%} R@3={agg['answer_recall@3']:.1%} "
              f"R@5={agg['answer_recall@5']:.1%} MRR={agg['mrr']:.3f} "
              f"({agg['avg_retrieval_ms']}ms/题)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rerank_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    off, on = results["rerank_off"], results["rerank_on"]
    md = ["***REMOVED*** Reranker 开关对比（同索引、同题集，纯检索）", "",
          f"题集：{results['n_questions']} 道可回答题（single+multi_hop+paraphrase+adversarial）；k={results['k']}", "",
          "| 配置 | R@1 | R@3 | R@5 | MRR | 均耗时/题 |", "|---|---|---|---|---|---|",
          f"| 重排关闭 | {off['answer_recall@1']:.1%} | {off['answer_recall@3']:.1%} "
          f"| {off['answer_recall@5']:.1%} | {off['mrr']:.3f} | {off['avg_retrieval_ms']}ms |",
          f"| 重排开启 | {on['answer_recall@1']:.1%} | {on['answer_recall@3']:.1%} "
          f"| {on['answer_recall@5']:.1%} | {on['mrr']:.3f} | {on['avg_retrieval_ms']}ms |", "",
          "***REMOVED******REMOVED*** 分题型 R@5", "", "| 题型 | 关闭 | 开启 |", "|---|---|---|"]
    for t in sorted(set(off["per_qtype"]) | set(on["per_qtype"])):
        o = off["per_qtype"].get(t, {}).get("recall@5")
        n = on["per_qtype"].get(t, {}).get("recall@5")
        md.append(f"| {t} | {o:.0%} | {n:.0%} |")
    (RESULTS_DIR / "rerank_compare.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✅ 对比报告 {RESULTS_DIR / 'rerank_compare.md'}")


if __name__ == "__main__":
    main()
