"""检索调优实验：RRF 网格搜索、BM25 分词变体对比、LLM 查询扩展对比。

流程：
1. 建一次智能切分索引；对每题缓存向量路 top-20 与关键词路 top-20 的排名；
2. RRF 网格：k ∈ {10,30,60,100} × p ∈ {1,1.5,2,3}，基于缓存的排名快速融合；
3. BM25 变体：precise / precise_dict / search / search_dict 分别重建索引再融合；
4. 查询扩展：LLM 生成补充关键词后检索，对比开/关；
5. 最优组合与默认配置做配对 bootstrap 显著性检验。

结果输出 eval/results/tuning.json 与 tuning.md。
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.bm25 import build_from_chunks  ***REMOVED*** noqa: E402
from rag.config import RRF_K, RRF_P  ***REMOVED*** noqa: E402
from rag.stats import paired_bootstrap_diff  ***REMOVED*** noqa: E402
from rag.vector_store import hydrate_all  ***REMOVED*** noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_eval import _gold_spans, _is_gold_chunk, build_index, load_questions  ***REMOVED*** noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
CORPUS_DIR = ROOT / "data" / "corpus"

RRF_GRID_K = [10, 30, 60, 100]
RRF_GRID_P = [1.0, 1.5, 2.0, 3.0]
BM25_VARIANTS = ["precise", "precise_dict", "search", "search_dict"]


def fuse_metrics(vec_hits, kw_hits, spans, gold_doc, rrf_k, rrf_p, k_final=5):
    """基于两路命中列表直接融合并计算指标（不重新检索）。"""
    vec_ranks = {h.chunk_id: i + 1 for i, h in enumerate(vec_hits)}
    kw_ranks = {h.chunk_id: i + 1 for i, h in enumerate(kw_hits)}
    scores: dict[str, float] = {}
    for cid, r in vec_ranks.items():
        scores[cid] = 1.0 / (rrf_k + r) ** rrf_p
    for cid, r in kw_ranks.items():
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r) ** rrf_p
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:k_final]
    by_id = {h.chunk_id: h for h in vec_hits}
    by_id.update({h.chunk_id: h for h in kw_hits})
    ranks = [i + 1 for i, cid in enumerate(ranked_ids)
             if _is_gold_chunk(by_id[cid], gold_doc, spans)]
    return ranks


def metrics_from_ranks(all_ranks):
    n = len(all_ranks)
    r3 = sum(1 for ranks in all_ranks if ranks and ranks[0] <= 3) / n
    r5 = sum(1 for ranks in all_ranks if ranks) / n
    mrr = sum(1.0 / ranks[0] for ranks in all_ranks if ranks) / n
    return {"recall@3": round(r3, 4), "recall@5": round(r5, 4), "mrr": round(mrr, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions_v2.jsonl"))
    ap.add_argument("--skip-expansion", action="store_true", help="跳过 LLM 查询扩展实验")
    args = ap.parse_args()

    questions = [q for q in load_questions(Path(args.questions)) if q["answerable"] and not q.get("multi_turn")]
    print(f"调优实验：{len(questions)} 道单轮可回答题")

    retriever = build_index("tune", "smart", CORPUS_DIR)

    ***REMOVED*** ---- 1. 缓存两路候选 ----
    print("缓存向量/关键词两路 top-20 ...")
    vec_cache, kw_cache = [], []
    for q in questions:
        vec_cache.append(retriever.vector_search(q["question"], 20))
        kw_cache.append(retriever.keyword_search(q["question"], 20))

    ***REMOVED*** ---- 2. RRF 网格搜索 ----
    print(f"RRF 网格搜索 k×p = {len(RRF_GRID_K)}×{len(RRF_GRID_P)} ...")
    grid = []
    for k in RRF_GRID_K:
        for p in RRF_GRID_P:
            ranks = [fuse_metrics(v, w, _gold_spans(q["gold_answer"]), q["gold_doc"], k, p)
                     for q, v, w in zip(questions, vec_cache, kw_cache)]
            m = metrics_from_ranks(ranks)
            grid.append({"k": k, "p": p, **m})
    grid.sort(key=lambda g: (g["recall@5"], g["mrr"]), reverse=True)
    best = grid[0]
    for g in grid:
        mark = " ⬅ 最优" if g is best else ("（当前默认）" if g["k"] == RRF_K and g["p"] == RRF_P else "")
        print(f"  k={g['k']:>3} p={g['p']}: R@3={g['recall@3']:.1%} R@5={g['recall@5']:.1%} MRR={g['mrr']:.3f}{mark}")

    ***REMOVED*** ---- 3. BM25 变体 ----
    print("BM25 分词变体对比 ...")
    chunks = hydrate_all(retriever.collection)
    variant_rows = []
    default_ranks = [fuse_metrics(v, w, _gold_spans(q["gold_answer"]), q["gold_doc"], best["k"], best["p"])
                     for q, v, w in zip(questions, vec_cache, kw_cache)]
    for variant in BM25_VARIANTS:
        retriever.bm25 = build_from_chunks(chunks, variant)
        kw2 = [retriever.keyword_search(q["question"], 20) for q in questions]
        ranks = [fuse_metrics(v, w, _gold_spans(q["gold_answer"]), q["gold_doc"], best["k"], best["p"])
                 for q, v, w in zip(questions, vec_cache, kw2)]
        m = metrics_from_ranks(ranks)
        variant_rows.append({"variant": variant, **m})
        print(f"  {variant:<14} R@3={m['recall@3']:.1%} R@5={m['recall@5']:.1%} MRR={m['mrr']:.3f}")
    best_variant = max(variant_rows, key=lambda x: (x["recall@5"], x["mrr"]))["variant"]
    retriever.bm25 = build_from_chunks(chunks, best_variant)
    print(f"  ⬅ 最优变体: {best_variant}")

    def full_ranks_with_current():
        kw2 = [retriever.keyword_search(q["question"], 20) for q in questions]
        return [fuse_metrics(v, w, _gold_spans(q["gold_answer"]), q["gold_doc"], best["k"], best["p"])
                for q, v, w in zip(questions, vec_cache, kw2)]

    ***REMOVED*** ---- 4. 查询扩展 ----
    expansion_rows = []
    if not args.skip_expansion:
        from rag.llm import expand_query_llm

        print("LLM 查询扩展实验（生成扩展词 → 检索）...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            expansions = list(pool.map(expand_query_llm, [q["question"] for q in questions]))
        t0 = time.time()
        ranks_exp = []
        for q, exp in zip(questions, expansions):
            hits = retriever.retrieve(q["question"], mode="hybrid", k_final=5, k_each=10, expansion=exp)
            ranks_exp.append([i + 1 for i, h in enumerate(hits)
                              if _is_gold_chunk(h, q["gold_doc"], _gold_spans(q["gold_answer"]))])
        base = metrics_from_ranks(default_ranks)
        exp_m = metrics_from_ranks(ranks_exp)
        expansion_rows = [{"setting": "off", **base}, {"setting": "on", **exp_m,
                                                       "avg_expansion_time": round((time.time() - t0) / len(questions), 2)}]
        for r in expansion_rows:
            print(f"  扩展={r['setting']:<4} R@3={r['recall@3']:.1%} R@5={r['recall@5']:.1%} MRR={r['mrr']:.3f}")

    ***REMOVED*** ---- 5. 显著性检验（配对 bootstrap）----
    sig = {}

    def paired(name, a_ranks, b_ranks, metric_idx):
        a = [1.0 if ranks and ranks[0] <= metric_idx else 0.0 for ranks in a_ranks]
        b = [1.0 if ranks and ranks[0] <= metric_idx else 0.0 for ranks in b_ranks]
        return paired_bootstrap_diff(b, a)

    tuned_ranks = full_ranks_with_current()
    sig["RRF最优 vs 默认(60,1.0) R@5"] = paired("tuned", tuned_ranks, default_ranks, 5)
    if best_variant != "precise":
        retriever.bm25 = build_from_chunks(chunks, "precise")
        precise_ranks = full_ranks_with_current()
        retriever.bm25 = build_from_chunks(chunks, best_variant)
        sig[f"BM25[{best_variant}] vs precise R@5"] = paired("variant", tuned_ranks, precise_ranks, 5)
    if expansion_rows:
        with ThreadPoolExecutor(max_workers=6) as pool:
            expansions = list(pool.map(expand_query_llm, [q["question"] for q in questions]))
        ranks_exp = []
        for q, exp in zip(questions, expansions):
            hits = retriever.retrieve(q["question"], mode="hybrid", k_final=5, k_each=10, expansion=exp)
            ranks_exp.append([i + 1 for i, h in enumerate(hits)
                              if _is_gold_chunk(h, q["gold_doc"], _gold_spans(q["gold_answer"]))])
        sig["查询扩展on vs off R@5"] = paired("expansion", ranks_exp, default_ranks, 5)

    results = {
        "rrf_grid": grid, "best_rrf": {"k": best["k"], "p": best["p"]},
        "bm25_variants": variant_rows, "best_variant": best_variant,
        "query_expansion": expansion_rows,
        "significance": {k: v for k, v in sig.items()},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tuning.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["***REMOVED*** 检索调优实验", "",
          f"实验集：{len(questions)} 道单轮可回答题（206 题评测集）；检索深度每路 20，最终取 5。", "",
          "***REMOVED******REMOVED*** RRF 网格搜索（score = Σ 1/(k+rank)^p）", "",
          "| k | p | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|---|"]
    for g in grid:
        md.append(f"| {g['k']} | {g['p']} | {g['recall@3']:.1%} | {g['recall@5']:.1%} | {g['mrr']:.3f} |")
    md += ["", f"**最优参数：k={best['k']}, p={best['p']}**", "",
           "***REMOVED******REMOVED*** BM25 分词变体（配自定义词典前后 / 搜索模式）", "",
           "| 变体 | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|"]
    for r in variant_rows:
        md.append(f"| {r['variant']} | {r['recall@3']:.1%} | {r['recall@5']:.1%} | {r['mrr']:.3f} |")
    md += ["", f"**最优变体：{best_variant}**"]
    if expansion_rows:
        md += ["", "***REMOVED******REMOVED*** LLM 查询扩展", "", "| 设置 | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|"]
        for r in expansion_rows:
            extra = f"（均摊 {r.get('avg_expansion_time', 0):.2f}s/题）" if "avg_expansion_time" in r else ""
            md.append(f"| {r['setting']} {extra}| {r['recall@3']:.1%} | {r['recall@5']:.1%} | {r['mrr']:.3f} |")
    md += ["", "***REMOVED******REMOVED*** 显著性检验（配对bootstrap，R@5）", "", "| 对比 | 差值 | 95%CI | p | n |", "|---|---|---|---|---|"]
    for name, s in sig.items():
        md.append(f"| {name} | {s['diff']:+.1%} | [{s['lo']:+.1%}, {s['hi']:+.1%}] | {s['p']:.4f} | {s['n']} |")
    (RESULTS_DIR / "tuning.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n✅ 调优结果已写入 {RESULTS_DIR / 'tuning.md'}")


if __name__ == "__main__":
    main()
