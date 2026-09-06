"""RAG 效果评测：三种配置对比，产出检索指标 + 端到端回答质量。

配置（消融实验）：
  baseline      固定窗口朴素切分 + 纯向量检索     （常见入门做法）
  smart_vector  结构感知智能切分 + 纯向量检索     （切分的贡献）
  hybrid        结构感知智能切分 + 混合检索 RRF   （完整系统，切分+检索的贡献）

指标：
  检索（26 道可回答题）：答案级 Recall@1/3/5、MRR、文档级 Recall@5
  生成（30 题，LLM-as-Judge 判分）：准确率(correct)、correct+partial、
        引用率、溯源准确率（引用的来源文档/章节是否正确）、拒答正确率（4 题）

用法：
  python scripts/run_eval.py                     ***REMOVED*** 全量评测（需 .env 配置 API_KEY）
  python scripts/run_eval.py --retrieval-only    ***REMOVED*** 只跑检索指标（无需 API）
  python scripts/run_eval.py --limit 5           ***REMOVED*** 冒烟测试
"""

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import API_KEY, FINAL_TOP_K, JUDGE_MODEL  ***REMOVED*** noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_INDEX_DIR = EVAL_DIR / ".eval_db"
SAMPLES_DIR = ROOT / "data" / "samples"

CONFIGS = {
    "baseline": {"label": "基线：固定窗口切分 + 纯向量检索", "chunker": "naive", "mode": "vector"},
    "smart_vector": {"label": "智能切分 + 纯向量检索", "chunker": "smart", "mode": "vector"},
    "hybrid": {"label": "智能切分 + 混合检索（完整系统）", "chunker": "smart", "mode": "hybrid"},
}

_norm = lambda s: re.sub(r"[^\w\u4e00-\u9fff]+", "", s.lower())  ***REMOVED*** noqa: E731


def _gold_spans(gold_answer: str) -> list[str]:
    """金标答案按「；」拆成多个原文片段，命中要求同一 chunk 包含全部片段。"""
    return [s for s in (_norm(p) for p in re.split(r"[；;\n]", gold_answer)) if s]


def _is_gold_chunk(hit, gold_doc: str, spans: list[str]) -> bool:
    if hit.doc_name != gold_doc or not spans:
        return False
    text = _norm(hit.text)
    return all(sp in text for sp in spans)


def load_questions(limit: int | None = None) -> list[dict]:
    questions = [json.loads(l) for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return questions[:limit] if limit else questions


def build_index(name: str, chunker: str):
    """为某配置重建独立评测索引（互不污染）。"""
    idx_dir = EVAL_INDEX_DIR / name
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    from rag.pipeline import ingest
    from rag.retriever import Retriever

    ingest([SAMPLES_DIR], index_dir=idx_dir, chunker=chunker, quiet=True)
    return Retriever(idx_dir)


***REMOVED*** ---------- 检索评测 ----------


def eval_retrieval(questions: list[dict], retriever, mode: str, k: int) -> list[dict]:
    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"]}
        if not q["answerable"]:
            rows.append(row)
            continue
        hits = retriever.retrieve(q["question"], mode=mode, k_final=k, k_each=10)
        gold_doc, spans = q["gold_doc"], _gold_spans(q["gold_answer"])
        for kk in (1, 3, 5):
            top = hits[:kk]
            row[f"doc_hit@{kk}"] = any(h.doc_name == gold_doc for h in top)
            row[f"ans_hit@{kk}"] = any(_is_gold_chunk(h, gold_doc, spans) for h in top)
        ranks = [i + 1 for i, h in enumerate(hits[:k]) if _is_gold_chunk(h, gold_doc, spans)]
        row["first_hit_rank"] = ranks[0] if ranks else None
        row["retrieved"] = [h.location for h in hits[:k]]
        rows.append(row)
    return rows


def aggregate_retrieval(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("answerable")]
    n = len(rows)
    if n == 0:
        return {}
    agg = {"n": n}
    for kk in (1, 3, 5):
        agg[f"answer_recall@{kk}"] = round(sum(r[f"ans_hit@{kk}"] for r in rows) / n, 4)
    agg["doc_recall@5"] = round(sum(r["doc_hit@5"] for r in rows) / n, 4)
    agg["mrr"] = round(sum(1.0 / r["first_hit_rank"] for r in rows if r["first_hit_rank"]) / n, 4)
    return agg


***REMOVED*** ---------- 生成评测（LLM-as-Judge） ----------


def _judge_one(q: dict, hits: list, k: int) -> dict:
    from rag.llm import answer_question, judge_answer

    out = {"id": q["id"], "question": q["question"], "answerable": q["answerable"]}
    result = answer_question(q["question"], hits[:k])
    out["answer"] = result["answer"]
    out["latency"] = round(result["latency"], 2)
    out["has_citation"] = result["has_citation"]
    out["cited"] = [
        {"doc_name": s.doc_name, "section_path": s.section_path, "page": s.page}
        for s in result["sources"]
    ]
    try:
        j = judge_answer(q["question"], q["gold_answer"], result["answer"],
                         answerable=q["answerable"], model=JUDGE_MODEL)
        out["verdict"], out["judge_reason"] = j["verdict"], j["reason"]
    except Exception as e:  ***REMOVED*** noqa: BLE001
        out["verdict"], out["judge_reason"] = "error", str(e)[:150]

    gold_doc, gold_sec = q["gold_doc"], q["gold_section"]
    out["citation_correct"] = bool(
        q["answerable"] and out["cited"]
        and any(c["doc_name"] == gold_doc and (not gold_sec or gold_sec in c["section_path"])
                for c in out["cited"]))
    return out


def eval_generation(questions: list[dict], retriever, mode: str, k: int, workers: int = 4) -> list[dict]:
    from rag.llm import get_client

    get_client()  ***REMOVED*** 提前初始化并校验配置
    jobs = []
    for q in questions:
        ***REMOVED*** 注意：拒答题同样要真实检索，模型需要面对"看似相关实则无关"的片段做出正确拒答
        hits = retriever.retrieve(q["question"], mode=mode, k_final=min(k, 10), k_each=10)
        jobs.append((q, hits))

    print(f"    生成+判分 {len(jobs)} 题（{workers} 并发）...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge_one, q, hits, k) for q, hits in jobs]
        rows = []
        for i, f in enumerate(futures, 1):
            rows.append(f.result())
            if i % 10 == 0:
                print(f"      ... {i}/{len(jobs)}（{time.time() - t0:.0f}s）")
    rows.sort(key=lambda r: r["id"])
    return rows


def aggregate_generation(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    n, nu = len(ans), len(unans)
    agg = {"n_answerable": n, "n_unanswerable": nu}
    if n:
        n_correct = sum(r["verdict"] == "correct" for r in ans)
        n_partial = sum(r["verdict"] == "partial" for r in ans)
        agg["accuracy"] = round(n_correct / n, 4)
        agg["accuracy_soft"] = round((n_correct + 0.5 * n_partial) / n, 4)
        agg["n_partial"] = n_partial
        agg["n_wrong"] = sum(r["verdict"] == "wrong" for r in ans)
        agg["n_error"] = sum(r["verdict"] == "error" for r in ans)
        agg["citation_rate"] = round(sum(r["has_citation"] for r in ans) / n, 4)
        agg["citation_accuracy"] = round(sum(r["citation_correct"] for r in ans) / n, 4)
        agg["avg_latency"] = round(sum(r["latency"] for r in ans) / n, 2)
    if nu:
        agg["refusal_correct"] = round(sum(r["verdict"] == "correct" for r in unans) / nu, 4)
    return agg


***REMOVED*** ---------- 报告 ----------


def write_report(results: dict, retrieval_only: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["***REMOVED*** RAG 效果评测报告", "",
             f"- 评测集：{results['_meta']['n_questions']} 题"
             f"（{results['_meta']['n_answerable']} 道可回答 + "
             f"{results['_meta']['n_unanswerable']} 道知识库未覆盖题）",
             f"- 评测文档：data/samples 下 4 篇（PDF / Word / TXT ×2）",
             f"- 评测时间：{results['_meta']['time']}",
             f"- 判分方式：检索指标按金标答案子串匹配；回答质量由大模型判分（LLM-as-Judge）",
             ""]

    lines += ["***REMOVED******REMOVED*** 一、检索效果对比（26 道可回答题）", "",
              "| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR | 文档级Recall@5 |",
              "|---|---|---|---|---|---|"]
    order = ["baseline", "smart_vector", "hybrid"]
    for name in order:
        if name not in results:
            continue
        r = results[name].get("retrieval", {})
        if not r:
            continue
        lines.append(
            f"| {CONFIGS[name]['label']} | {r['answer_recall@1']:.1%} | {r['answer_recall@3']:.1%} "
            f"| {r['answer_recall@5']:.1%} | {r['mrr']:.3f} | {r['doc_recall@5']:.1%} |")

    if not retrieval_only:
        lines += ["", "***REMOVED******REMOVED*** 二、端到端回答质量（30 题，LLM 判分）", "",
                  "| 配置 | 准确率(correct) | correct+partial | 引用率 | 溯源准确率 | 拒答正确率 | 平均生成耗时 |",
                  "|---|---|---|---|---|---|---|"]
        for name in order:
            if name not in results:
                continue
            g = results[name].get("generation", {})
            if not g:
                continue
            lines.append(
                f"| {CONFIGS[name]['label']} | {g['accuracy']:.1%} | {g['accuracy_soft']:.1%} "
                f"| {g['citation_rate']:.1%} | {g['citation_accuracy']:.1%} "
                f"| {g.get('refusal_correct', 0):.0%} | {g['avg_latency']:.1f}s |")

    lines += ["", "> - Recall@k：前 k 个检索块中包含标准答案片段的比例（答案级）。",
              "> - MRR：首个命中块排名倒数的平均值，衡量排序质量。",
              "> - 引用率：回答中带 [n] 引用标注的比例；溯源准确率：引用的文档/章节指向金标位置的比例。",
              "> - 拒答正确率：4 道知识库未覆盖题，模型正确回答「未找到」的比例。"]
    out = RESULTS_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


***REMOVED*** ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser(description="RAG 效果评测")
    ap.add_argument("--configs", default="baseline,smart_vector,hybrid",
                    help="逗号分隔：baseline,smart_vector,hybrid")
    ap.add_argument("--retrieval-only", action="store_true", help="只跑检索指标（无需 API Key）")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 题（冒烟测试）")
    ap.add_argument("--k", type=int, default=FINAL_TOP_K, help="送入大模型的最终条数")
    args = ap.parse_args()

    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.iterdir()):
        print("❌ 未找到示例文档，请先运行: python scripts/make_samples.py")
        sys.exit(1)

    questions = load_questions(args.limit)
    n_ans = sum(1 for q in questions if q["answerable"])
    names = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    if not args.retrieval_only and not API_KEY:
        print("⚠️  未配置 API_KEY，自动降级为 --retrieval-only（检索指标无需大模型）")
        args.retrieval_only = True

    print(f"评测集：{len(questions)} 题（可回答 {n_ans}）｜配置：{names}"
          f"｜模式：{'仅检索' if args.retrieval_only else '检索+生成+判分'}")

    results = {"_meta": {
        "n_questions": len(questions), "n_answerable": n_ans,
        "n_unanswerable": len(questions) - n_ans,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_top_k": args.k, "retrieval_only": args.retrieval_only,
    }}

    for name in names:
        cfg = CONFIGS[name]
        print(f"\n===== 配置 [{name}] {cfg['label']} =====")
        retriever = build_index(name, cfg["chunker"])
        print(f"  🔍 检索评测（mode={cfg['mode']}）...")
        ret_rows = eval_retrieval(questions, retriever, cfg["mode"], args.k)
        results[name] = {"retrieval": aggregate_retrieval(ret_rows)}
        r = results[name]["retrieval"]
        print(f"     Recall@5={r['answer_recall@5']:.1%}  MRR={r['mrr']:.3f}  文档级Recall@5={r['doc_recall@5']:.1%}")

        if not args.retrieval_only:
            gen_rows = eval_generation(questions, retriever, cfg["mode"], args.k)
            results[name]["generation"] = aggregate_generation(gen_rows)
            results[name]["generation_detail"] = gen_rows
            g = results[name]["generation"]
            print(f"     准确率={g['accuracy']:.1%}  引用率={g['citation_rate']:.1%}  溯源准确率={g['citation_accuracy']:.1%}")

        ***REMOVED*** 检索明细一并留存
        results[name]["retrieval_detail"] = ret_rows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(results, args.retrieval_only)
    print(f"\n✅ 评测完成：\n   结果 JSON: {RESULTS_DIR / 'eval_results.json'}\n   报告: {report}")


if __name__ == "__main__":
    main()
