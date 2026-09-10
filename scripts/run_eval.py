"""RAG 效果评测 v2：206 题语料、五项指标、bootstrap 置信区间、消融与显著性检验。

配置（消融设计，单一变量递增）：
  baseline      固定窗口朴素切分 + 纯向量检索          （常见入门做法）
  smart_vector  结构感知智能切分 + 纯向量检索          （隔离"切分"贡献）
  hybrid        智能切分 + 混合检索 RRF                （隔离"混合检索"贡献）
  full          混合检索 + 多轮查询改写                （隔离"改写"贡献，
                 仅评测多轮指代题——单轮路径与 hybrid 完全一致）

指标：
  检索：答案级 Recall@1/3/5、MRR、文档级 Recall@5
  生成：准确率(correct/partial/wrong, LLM-as-Judge)、忠实度(0-1)、答案相关性(0-1)、
        引用率/溯源准确率/引用校验、拒答正确率
所有主要指标给出 95% bootstrap 置信区间；关键对比给出配对显著性检验。

用法：
  python scripts/run_eval.py                          ***REMOVED*** 全量评测（4 配置）
  python scripts/run_eval.py --retrieval-only         ***REMOVED*** 只跑检索指标
  python scripts/run_eval.py --configs baseline,hybrid
  python scripts/run_eval.py --legacy                 ***REMOVED*** 兼容 v1 的 30 题回归
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
from rag.stats import bootstrap_ci, fmt_ci, paired_bootstrap_diff  ***REMOVED*** noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_INDEX_DIR = EVAL_DIR / ".eval_db"
CORPUS_DIR = ROOT / "data" / "corpus"
SAMPLES_DIR = ROOT / "data" / "samples"

CONFIGS = {
    "baseline": {"label": "基线：固定窗口切分+纯向量", "chunker": "naive", "mode": "vector", "rewrite": False},
    "smart_vector": {"label": "智能切分+纯向量", "chunker": "smart", "mode": "vector", "rewrite": False},
    "hybrid": {"label": "智能切分+混合检索", "chunker": "smart", "mode": "hybrid", "rewrite": False},
    "full": {"label": "完整系统(混合检索+多轮改写)", "chunker": "smart", "mode": "hybrid", "rewrite": True},
}

_norm = lambda s: re.sub(r"[^\w\u4e00-\u9fff]+", "", s.lower())  ***REMOVED*** noqa: E731


def _gold_spans(gold_answer: str) -> list[str]:
    return [s for s in (_norm(p) for p in re.split(r"[；;\n]", gold_answer)) if s]


def _is_gold_chunk(hit, gold_doc: str, spans: list[str]) -> bool:
    if hit.doc_name != gold_doc or not spans:
        return False
    text = _norm(hit.text)
    return all(sp in text for sp in spans)


def _multi_gold_covered(hits: list, gold_items: list[dict]) -> tuple[bool, int | None]:
    """多跳题：每个金标项（文档+片段）都需在 hits 中被命中（可在不同块）。
    返回 (是否全部覆盖, 完整覆盖时的最大首中排名)。"""
    max_rank = 0
    for g in gold_items:
        spans = [s for s in (_norm(x) for x in g["spans"]) if s]
        ranks = [i + 1 for i, h in enumerate(hits)
                 if h.doc_name == g["doc"] and all(sp in _norm(h.text) for sp in spans)]
        if not ranks:
            return False, None
        max_rank = max(max_rank, ranks[0])
    return True, max_rank


def _qtype(q: dict) -> str:
    if q.get("qtype"):
        return q["qtype"]
    if not q["answerable"]:
        return "refusal"
    return "multi_turn" if q.get("multi_turn") else "single"


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    questions = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return questions[:limit] if limit else questions


def build_index(name: str, chunker: str, corpus: Path):
    idx_dir = EVAL_INDEX_DIR / name
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    from rag.pipeline import ingest
    from rag.retriever import Retriever

    ingest([corpus], index_dir=idx_dir, chunker=chunker, quiet=True)
    return Retriever(idx_dir)


***REMOVED*** ---------- 检索评测 ----------


def eval_retrieval(questions: list[dict], retriever, cfg: dict, k: int, use_llm_rewrite: bool) -> list[dict]:
    from rag.rewrite import rewrite_query

    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "doc_type": q.get("doc_type", ""), "multi_turn": q.get("multi_turn", False),
               "qtype": _qtype(q)}
        if not q["answerable"]:
            rows.append(row)
            continue
        query = q["question"]
        method = "none"
        if cfg["rewrite"] and q.get("history"):
            rw = rewrite_query(query, [{"role": "user", "content": h} for h in q["history"]],
                               use_llm=use_llm_rewrite)
            query, method = rw["query"], rw["method"]
        row["query_used"], row["rewrite_method"] = query, method
        hits = retriever.retrieve(query, mode=cfg["mode"], k_final=k, k_each=10,
                                  rerank_on=cfg.get("rerank"))
        gold_doc, spans = q["gold_doc"], _gold_spans(q["gold_answer"])
        if q.get("qtype") == "multi_hop" and q.get("gold"):
            ***REMOVED*** 修复：原先这里硬编码 hits[:5]，与函数参数 k（--k 可调）脱节——
            ***REMOVED*** 一旦用 --k 调小/调大召回窗口，多跳题的 first_hit_rank 仍按 5 计算，
            ***REMOVED*** 导致 MRR 与 Recall@k 口径不一致。统一改用 k。
            covered, max_rank = _multi_gold_covered(hits[:k], q["gold"])
            for kk in (1, 3, 5):
                c, _ = _multi_gold_covered(hits[:kk], q["gold"])
                row[f"ans_hit@{kk}"] = c
                row[f"doc_hit@{kk}"] = all(any(h.doc_name == g["doc"] for h in hits[:kk]) for g in q["gold"])
            row["first_hit_rank"] = max_rank if covered else None
        else:
            for kk in (1, 3, 5):
                top = hits[:kk]
                row[f"doc_hit@{kk}"] = any(h.doc_name == gold_doc for h in top)
                row[f"ans_hit@{kk}"] = any(_is_gold_chunk(h, gold_doc, spans) for h in top)
            ranks = [i + 1 for i, h in enumerate(hits[:k]) if _is_gold_chunk(h, gold_doc, spans)]
            row["first_hit_rank"] = ranks[0] if ranks else None
        rows.append(row)
    return rows


def aggregate_retrieval(rows: list[dict]) -> dict:
    ans = [r for r in rows if r.get("answerable")]
    if not ans:
        return {}
    agg = {"n": len(ans)}
    for kk in (1, 3, 5):
        agg[f"answer_recall@{kk}"] = round(sum(r[f"ans_hit@{kk}"] for r in ans) / len(ans), 4)
    agg["doc_recall@5"] = round(sum(r["doc_hit@5"] for r in ans) / len(ans), 4)
    agg["mrr"] = round(sum(1.0 / r["first_hit_rank"] for r in ans if r["first_hit_rank"]) / len(ans), 4)
    return agg


***REMOVED*** ---------- 生成评测（判分 + 忠实度 + 相关性） ----------


def _judge_one(q: dict, hits: list, k: int, query: str | None = None) -> dict:
    from rag.llm import answer_question, judge_answer, judge_faithfulness, judge_relevance

    out = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
           "doc_type": q.get("doc_type", ""), "multi_turn": q.get("multi_turn", False),
           "qtype": _qtype(q)}
    used = hits[:k]
    ***REMOVED*** 多轮题用改写后的独立问题生成（上下文也是按改写问题检索的）
    try:
        result = answer_question(query or q["question"], used, max_tokens=512)
    except Exception as e:  ***REMOVED*** noqa: BLE001 — 单题失败不中断整体评测（如配额耗尽）
        out.update({"answer": "", "latency": 0.0, "has_citation": False, "citations": [],
                    "forged_citations": [], "citation_retry": 0,
                    "citation_warning": "", "cited": [],
                    "verdict": "error", "judge_reason": f"生成失败: {str(e)[:120]}",
                    "faithfulness": None, "relevance": None, "citation_correct": None})
        return out
    out["answer"] = result["answer"]
    out["latency"] = round(result["latency"], 2)
    out["has_citation"] = result["has_citation"]
    out["citations"] = result["citations"]
    out["forged_citations"] = result["forged_citations"]
    out["citation_retry"] = result["citation_retry"]
    out["citation_warning"] = result["citation_warning"]
    out["cited"] = [{"doc_name": s.doc_name, "section_path": s.section_path, "page": s.page}
                    for s in result["sources"]]
    try:
        j = judge_answer(q["question"], q["gold_answer"], result["answer"],
                         answerable=q["answerable"], model=JUDGE_MODEL)
        out["verdict"], out["judge_reason"] = j["verdict"], j["reason"]
    except Exception as e:  ***REMOVED*** noqa: BLE001
        out["verdict"], out["judge_reason"] = "error", str(e)[:150]

    if q["answerable"]:
        out["faithfulness"] = judge_faithfulness(result["answer"], used)
        out["relevance"] = judge_relevance(q["question"], result["answer"])
        gold_doc, gold_sec = q["gold_doc"], q["gold_section"]
        out["citation_correct"] = bool(out["cited"] and any(
            c["doc_name"] == gold_doc and (not gold_sec or gold_sec in c["section_path"])
            for c in out["cited"]))
    else:
        out["faithfulness"] = out["relevance"] = out["citation_correct"] = None
    return out


def eval_generation(questions: list[dict], retriever, cfg: dict, k: int,
                    workers: int = 6, use_llm_rewrite: bool = True) -> list[dict]:
    from rag.llm import get_client
    from rag.rewrite import rewrite_query

    get_client()[0]  ***REMOVED*** 提前初始化并校验配置
    jobs = []
    for q in questions:
        query = q["question"]
        if cfg["rewrite"] and q.get("history"):
            query = rewrite_query(query, [{"role": "user", "content": h} for h in q["history"]],
                                  use_llm=use_llm_rewrite)["query"]
        ***REMOVED*** 修复：与 eval_retrieval 保持一致地传入 rerank_on——
        ***REMOVED*** 否则同一配置下"检索指标"与"生成指标"基于不同的候选集（一个重排了、一个没重排），
        ***REMOVED*** 报告里的检索/生成两栏数据无法互相解释。
        hits = retriever.retrieve(query, mode=cfg["mode"], k_final=min(k, 10), k_each=10,
                                  rerank_on=cfg.get("rerank"))
        jobs.append((q, hits, query))

    print(f"    生成+四维判分 {len(jobs)} 题（{workers} 并发）...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge_one, q, hits, k, query) for q, hits, query in jobs]
        rows = []
        for i, (f, (_, _, query)) in enumerate(zip(futures, jobs), 1):
            r = f.result()
            r["query_used"] = query
            rows.append(r)
            if i % 20 == 0:
                print(f"      ... {i}/{len(jobs)}（{time.time() - t0:.0f}s）")
    rows.sort(key=lambda r: r["id"])
    return rows


def aggregate_generation(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    agg = {"n_answerable": len(ans), "n_unanswerable": len(unans)}
    if ans:
        n = len(ans)
        nc = sum(r["verdict"] == "correct" for r in ans)
        np_ = sum(r["verdict"] == "partial" for r in ans)
        n_f = sum(r["faithfulness"] is not None for r in ans)
        n_r = sum(r["relevance"] is not None for r in ans)
        agg.update({
            "accuracy": round(nc / n, 4),
            "accuracy_soft": round((nc + 0.5 * np_) / n, 4),
            "n_correct": nc, "n_partial": np_, "n_wrong": sum(r["verdict"] == "wrong" for r in ans),
            "n_error": sum(r["verdict"] == "error" for r in ans),
            "citation_rate": round(sum(bool(r["has_citation"]) for r in ans) / n, 4),
            "citation_valid_rate": round(sum(not r["citation_warning"] for r in ans) / n, 4),
            "n_citation_retry": sum(r["citation_retry"] for r in ans),
            "citation_accuracy": round(sum(bool(r["citation_correct"]) for r in ans) / n, 4),
            "faithfulness": round(sum(r["faithfulness"] for r in ans if r["faithfulness"] is not None) / max(1, n_f), 4),
            "relevance": round(sum(r["relevance"] for r in ans if r["relevance"] is not None) / max(1, n_r), 4),
            "avg_latency": round(sum(r["latency"] for r in ans) / n, 2),
        })
    if unans:
        agg["refusal_correct"] = round(sum(r["verdict"] == "correct" for r in unans) / len(unans), 4)
    return agg


***REMOVED*** ---------- 置信区间数组 / 分类型 ----------


def collect_arrays(ret_rows: list[dict], gen_rows: list[dict] | None) -> dict:
    arrs = {}
    ans_ret = [r for r in ret_rows if r.get("answerable")]
    for kk in (1, 3, 5):
        arrs[f"recall@{kk}"] = [float(r[f"ans_hit@{kk}"]) for r in ans_ret]
    arrs["mrr"] = [1.0 / r["first_hit_rank"] if r.get("first_hit_rank") else 0.0 for r in ans_ret]
    if gen_rows:
        g = [r for r in gen_rows if r["answerable"]]
        arrs["accuracy"] = [1.0 if r["verdict"] == "correct" else (0.5 if r["verdict"] == "partial" else 0.0)
                            for r in g]
        arrs["faithfulness"] = [r["faithfulness"] for r in g]
        arrs["relevance"] = [r["relevance"] for r in g]
        arrs["refusal"] = [1.0 if r["verdict"] == "correct" else 0.0
                           for r in gen_rows if not r["answerable"]]
    return arrs


def per_type_table(ret_rows: list[dict], gen_rows: list[dict] | None) -> dict:
    types = sorted({r["doc_type"] for r in ret_rows if r.get("doc_type")})
    out = {}
    for t in types:
        rs = [r for r in ret_rows if r.get("doc_type") == t and r.get("answerable")]
        entry = {"n": len(rs), "recall@5": round(sum(r["ans_hit@5"] for r in rs) / len(rs), 4)} if rs else {}
        if gen_rows:
            gs = [r for r in gen_rows if r.get("doc_type") == t and r["answerable"]]
            if gs:
                entry["accuracy"] = round(sum(r["verdict"] == "correct" for r in gs) / len(gs), 4)
                entry["n_gen"] = len(gs)
        out[t] = entry
    return out


def per_qtype_table(ret_rows: list[dict], gen_rows: list[dict] | None) -> dict:
    """按题型（单轮/多跳/改写/对抗/拒答/多轮）分项。"""
    qtypes = sorted({r.get("qtype", "single") for r in ret_rows if r.get("answerable")})
    out = {}
    for t in qtypes:
        rs = [r for r in ret_rows if r.get("qtype") == t and r.get("answerable")]
        if not rs:
            continue
        entry = {"n": len(rs), "recall@5": round(sum(r["ans_hit@5"] for r in rs) / len(rs), 4)}
        if gen_rows:
            gs = [r for r in gen_rows if r.get("qtype") == t and r["answerable"]]
            if gs:
                entry["accuracy"] = round(sum(r["verdict"] == "correct" for r in gs) / len(gs), 4)
                entry["n_gen"] = len(gs)
        out[t] = entry
    return out


***REMOVED*** ---------- 报告 ----------


def write_report(results: dict, args) -> Path:
    order = [c for c in ("baseline", "smart_vector", "hybrid", "full") if c in results]
    lines = ["***REMOVED*** RAG 效果评测报告 v2（升级后）", "",
             f"- 评测集：`{results['_meta']['questions_file']}`，{results['_meta']['n_questions']} 题"
             f"（可回答 {results['_meta']['n_answerable']}（含多轮 {results['_meta']['n_multi_turn']}）"
             f"+ 拒答 {results['_meta']['n_unanswerable']}，拒答占比 {results['_meta']['refusal_ratio']:.1%}）",
             f"- 语料：{results['_meta']['n_docs']} 篇企业文档（制度/合同/说明书/表格/FAQ/技术文档）",
             "- 判分：检索=金标原文片段匹配；生成=LLM-as-Judge + 忠实度/相关性逐题判分",
             f"- 置信区间：{args.bootstrap} 次bootstrap重采样的95%CI；评测时间：{results['_meta']['time']}", ""]

    lines += ["***REMOVED******REMOVED*** 一、检索效果（可回答题，均值 [95%CI]）", "",
              "| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR |",
              "|---|---|---|---|---|"]
    for name in order:
        r, a = results[name].get("retrieval", {}), results[name].get("_arrays", {})
        if not r:
            continue
        r1 = fmt_ci(*bootstrap_ci(a.get("recall@1", [])))
        r3 = fmt_ci(*bootstrap_ci(a.get("recall@3", [])))
        r5 = fmt_ci(*bootstrap_ci(a.get("recall@5", [])))
        mrr = fmt_ci(*bootstrap_ci(a.get("mrr", [])), pct=False)
        lines.append(f"| {CONFIGS[name]['label']}（{name}） | {r1} | {r3} | {r5} | {mrr} |")

    gen_names = [n for n in order if results[n].get("generation")]
    if gen_names:
        lines += ["", "***REMOVED******REMOVED*** 二、端到端回答质量（五项指标，均值 [95%CI]）", "",
                  "| 配置 | 准确率(correct) | 正确+部分 | 忠实度 | 相关性 | 引用率 | 溯源准确率 | 拒答正确率 |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in gen_names:
            g = results[name]["generation"]
            a = results[name].get("_arrays", {})
            acc = fmt_ci(*bootstrap_ci(a.get("accuracy", [])))
            faith = fmt_ci(*bootstrap_ci(a.get("faithfulness", [])), pct=False)
            rel = fmt_ci(*bootstrap_ci(a.get("relevance", [])), pct=False)
            ref = fmt_ci(*bootstrap_ci(a.get("refusal", [])))
            lines.append(
                f"| {CONFIGS[name]['label']} | {acc} | {g['accuracy_soft']:.1%} | {faith} | {rel} "
                f"| {g['citation_rate']:.1%} | {g['citation_accuracy']:.1%} | {ref} |")
        lines += ["", "> 忠实度=答案句子被检索块支持的比例；相关性=直接回答问题且无冗余的程度（均由 LLM 0~1 判分）。"]

    lines += ["", "***REMOVED******REMOVED*** 三、按文档类型分项（Recall@5 / 准确率）", ""]
    types = sorted({t for name in order for t in results[name].get("per_type", {})})
    lines += ["| 类型 | 题数 | " + " | ".join(f"{CONFIGS[n]['label']}" for n in order) + " |",
              "|---|---|" + "---|" * len(order)]
    for t in types:
        cells = []
        n = next((results[name]["per_type"].get(t, {}).get("n", "-") for name in order
                  if t in results[name].get("per_type", {})), "-")
        for name in order:
            e = results[name].get("per_type", {}).get(t, {})
            r5 = f"{e['recall@5']:.0%}" if e else "-"
            acc = f"{e['accuracy']:.0%}" if "accuracy" in e else "-"
            cells.append(f"{r5} / {acc}")
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")

    qtypes = sorted({t for name in order for t in results[name].get("per_qtype", {})})
    if qtypes:
        lines += ["", "***REMOVED******REMOVED*** 三之二、按题型分项（Recall@5 / 准确率）", "",
                  "| 题型 | 题数 | " + " | ".join(f"{CONFIGS[n]['label']}" for n in order) + " |",
                  "|---|---|" + "---|" * len(order)]
        for t in qtypes:
            cells = []
            n = next((results[name]["per_qtype"].get(t, {}).get("n", "-") for name in order
                      if t in results[name].get("per_qtype", {})), "-")
            for name in order:
                e = results[name].get("per_qtype", {}).get(t, {})
                r5 = f"{e['recall@5']:.0%}" if e else "-"
                acc = f"{e['accuracy']:.0%}" if "accuracy" in e else "-"
                cells.append(f"{r5} / {acc}")
            lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")

    lines += ["", "***REMOVED******REMOVED*** 四、多轮查询改写消融（仅多轮指代题，改写前 vs 改写后）", "",
              "| 配置 | 题数 | Recall@5 | 准确率 |", "|---|---|---|---|"]
    for name in order:
        ret_rows = results[name].get("retrieval_detail", [])
        mt_ret = [r for r in ret_rows if r.get("multi_turn") and r.get("answerable")]
        if not mt_ret:
            continue
        r5 = sum(r["ans_hit@5"] for r in mt_ret) / len(mt_ret)
        g = [r for r in results[name].get("generation_detail", []) if r.get("multi_turn")]
        acc = f"{sum(r['verdict'] == 'correct' for r in g) / len(g):.0%}" if g else "-"
        lines.append(f"| {CONFIGS[name]['label']} | {len(mt_ret)} | {r5:.1%} | {acc} |")

    sig = results.get("_significance", {})
    if sig:
        lines += ["", "***REMOVED******REMOVED*** 五、关键对比的显著性（配对bootstrap，双侧）", "",
                  "| 对比 | 指标 | 差值 | 95%CI | p值 | n |", "|---|---|---|---|---|---|"]
        for key, s in sig.items():
            lines.append(f"| {key} | {s['metric']} | {s['diff']:+.1%} | [{s['lo']:+.1%}, {s['hi']:+.1%}] "
                         f"| {s['p']:.4f} | {s['n']} |")
        lines += ["", "> p<0.05 视为显著；95%CI 不含 0 与之等价。"]

    lines += ["", "***REMOVED******REMOVED*** 六、说明与局限", "",
              "- 金标答案一律为文档原文片段（生成时校验“片段⊆文档文本”），答案级命中要求同一召回块包含全部片段；",
              "- full 配置的单轮路径与 hybrid 完全一致（改写仅在多轮触发），故仅在多轮题上单独评测；",
              "- 忠实度/相关性/判分由大模型完成（与被测模型同源），存在自判偏差；",
              "- 拒答题 34 题、多轮题 16 题，其 CI 较宽，结论以方向性为主。"]
    out = RESULTS_DIR / "report_v2.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


***REMOVED*** ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser(description="RAG 效果评测 v2")
    ap.add_argument("--configs", default="baseline,smart_vector,hybrid,full")
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions_v2.jsonl"))
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--legacy", action="store_true", help="使用 v1 的 30 题回归评测")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=FINAL_TOP_K)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="eval_results_v2.json", help="结果文件名（写入 eval/results/）")
    ap.add_argument("--no-llm-rewrite", action="store_true", help="多轮改写离线降级为规则改写")
    args = ap.parse_args()

    if args.legacy:
        args.questions = str(EVAL_DIR / "questions.jsonl")
        args.corpus = str(SAMPLES_DIR)
    qpath, corpus = Path(args.questions), Path(args.corpus)
    if not corpus.exists() or not any(corpus.iterdir()):
        print(f"❌ 语料不存在: {corpus}（先运行 scripts/make_corpus.py）")
        sys.exit(1)

    questions = load_questions(qpath, args.limit)
    names = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    ***REMOVED*** full 只需评测多轮题（单轮路径与 hybrid 一致）
    eval_q = {name: ([q for q in questions if q.get("multi_turn")] if CONFIGS[name]["rewrite"] else questions)
              for name in names}
    use_llm_rewrite = bool(API_KEY) and not args.no_llm_rewrite
    if not args.retrieval_only and not API_KEY:
        print("⚠️  未配置 API_KEY，自动降级为 --retrieval-only")
        args.retrieval_only = True

    n_ans = sum(1 for q in questions if q["answerable"])
    print(f"评测集：{len(questions)} 题（可回答 {n_ans}，拒答 {len(questions) - n_ans}）"
          f"｜语料：{len(list(corpus.glob('*')))} 篇｜配置：{names}")

    results = {"_meta": {
        "n_questions": len(questions), "n_answerable": n_ans,
        "n_unanswerable": len(questions) - n_ans,
        "n_multi_turn": sum(1 for q in questions if q.get("multi_turn")),
        "refusal_ratio": round((len(questions) - n_ans) / len(questions), 4),
        "n_docs": len(list(corpus.glob("*"))),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_top_k": args.k, "retrieval_only": args.retrieval_only,
        "questions_file": qpath.name,
    }}

    for name in names:
        cfg = CONFIGS[name]
        print(f"\n===== [{name}] {cfg['label']} =====")
        retriever = build_index(name, cfg["chunker"], corpus)
        qs = eval_q[name]
        print(f"  🔍 检索评测（mode={cfg['mode']}, n={len(qs)}）...")
        ret_rows = eval_retrieval(qs, retriever, cfg, args.k, use_llm_rewrite)
        results[name] = {"retrieval": aggregate_retrieval(ret_rows), "retrieval_detail": ret_rows}
        r = results[name]["retrieval"]
        if r:
            print(f"     Recall@5={r['answer_recall@5']:.1%}  MRR={r['mrr']:.3f}  (n={r['n']})")
        else:
            print("     （该子集无可评测题）")

        if not args.retrieval_only:
            gen_rows = eval_generation(qs, retriever, cfg, args.k, use_llm_rewrite=use_llm_rewrite)
            results[name]["generation"] = aggregate_generation(gen_rows)
            results[name]["generation_detail"] = gen_rows
            g = results[name]["generation"]
            print(f"     准确率={g['accuracy']:.1%}  忠实度={g['faithfulness']:.3f}  "
                  f"相关性={g['relevance']:.3f}  溯源准确率={g['citation_accuracy']:.1%}")

        results[name]["_arrays"] = collect_arrays(ret_rows, results[name].get("generation_detail"))
        results[name]["per_type"] = per_type_table(ret_rows, results[name].get("generation_detail"))
        results[name]["per_qtype"] = per_qtype_table(ret_rows, results[name].get("generation_detail"))

        ***REMOVED*** 每个配置完成即落盘，中断时保留已完成部分
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ***REMOVED*** 修复：这里原先硬编码写回 eval_results_v2.json，忽略了 --out 参数——
        ***REMOVED*** 用 `--out eval_results_v4_harden.json` 分次评测时，中间结果会覆盖/写错文件。
        (RESULTS_DIR / args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ***REMOVED*** ---- 显著性检验（配对，按题目对齐） ----
    sig = {}

    def by_id(rows, field, ids):
        m = {r["id"]: r.get(field) for r in rows}
        return [m.get(i) for i in ids]

    def verdict_vals(rows, ids):
        v = by_id(rows, "verdict", ids)
        return [1.0 if x == "correct" else (0.5 if x == "partial" else 0.0) for x in v]

    if "baseline" in results and "hybrid" in results:
        ids = [r["id"] for r in results["baseline"]["retrieval_detail"] if r.get("answerable")]
        for metric, field in [("Recall@5", "ans_hit@5"), ("MRR", "first_hit_rank")]:
            a = by_id(results["baseline"]["retrieval_detail"], field, ids)
            b = by_id(results["hybrid"]["retrieval_detail"], field, ids)
            if field == "first_hit_rank":
                a = [1.0 / x if x else 0.0 for x in a]
                b = [1.0 / x if x else 0.0 for x in b]
            s = paired_bootstrap_diff(b, a, n_boot=args.bootstrap)
            s["metric"] = metric
            sig["hybrid vs baseline"] = s
    if "smart_vector" in results and "hybrid" in results:
        ids = [r["id"] for r in results["smart_vector"]["retrieval_detail"] if r.get("answerable")]
        a = by_id(results["smart_vector"]["retrieval_detail"], "ans_hit@5", ids)
        b = by_id(results["hybrid"]["retrieval_detail"], "ans_hit@5", ids)
        s = paired_bootstrap_diff(b, a, n_boot=args.bootstrap)
        s["metric"] = "Recall@5"
        sig["hybrid vs smart_vector"] = s
    if "hybrid" in results and "full" in results and results["full"].get("generation_detail"):
        mt_ids = [q["id"] for q in questions if q.get("multi_turn")]
        a = verdict_vals(results["hybrid"].get("generation_detail", []), mt_ids)
        b = verdict_vals(results["full"].get("generation_detail", []), mt_ids)
        s = paired_bootstrap_diff(b, a, n_boot=args.bootstrap)
        s["metric"] = "多轮题准确率"
        sig["full vs hybrid"] = s
    results["_significance"] = sig

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ***REMOVED*** 与既有结果合并：分次调用评测不同配置时，已完成的配置保留
    out_path = RESULTS_DIR / args.out
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for k, v in prev.items():
                if k not in results:
                    results[k] = v
        except (OSError, json.JSONDecodeError):
            pass
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(results, args)
    print(f"\n✅ 评测完成：\n   {out_path}\n   {report}")


if __name__ == "__main__":
    main()
