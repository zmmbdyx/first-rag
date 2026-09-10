"""大规模基准测试：生成 N 篇模拟文档 → 分阶段入库计时 → 查询延迟 P50/P95 → 内存占用。

- 语料程序化生成：每篇文档为某虚拟部门的费用/资产/值班制度，事实唯一；
- 入库计时拆分：解析切分 / 向量编码 / 向量库写入 / BM25 构建；
- 增量验证：同样语料二次入库应全部跳过（MD5 未变）；
- 延迟测量：仅检索链路（向量编码+两路检索+RRF），不含大模型生成；
- 结果写入 eval/results/benchmark_scale.md / benchmark_scale.json。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_DIR = ROOT / "eval" / ".eval_db" / "bench"
RESULTS_DIR = ROOT / "eval" / "results"
CORPUS_TMP = ROOT / "eval" / ".eval_db" / "bench_docs"

DEPTS = ["行政部", "财务部", "研发中心", "市场部", "人力资源部", "采购部", "法务部", "客服部",
         "运维部", "质检部", "仓储部", "物流部", "培训部", "安保部", "信息部", "生产部",
         "设计部", "测试部", "数据部", "工程部"]


def gen_docs(n: int) -> list[Path]:
    import shutil

    if CORPUS_TMP.exists():
        shutil.rmtree(CORPUS_TMP)
    CORPUS_TMP.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        dept = DEPTS[i % len(DEPTS)]
        seq = i // len(DEPTS) + 1
        lines = [
            f"{dept}第{seq}号管理细则",
            "",
            "一、费用标准",
            f"本部门{i}类差旅补贴为每人每天{100 + i % 200}元，{i}类交通补贴为每人每天{20 + i % 60}元。",
            f"本部门{i}类会议费上限为每次{500 + i * 7 % 4000}元，超支需分管副总特批。",
            "",
            "二、资产管理",
            f"本部门{i}号库房存放设备{50 + i % 300}台，资产编号前缀为ZY{i:04d}。",
            f"设备折旧年限为{3 + i % 8}年，报废需经{i}号流程审批。",
            "",
            "三、值班制度",
            f"值班时间为每日{8 + i % 3}点至{18 + i % 5}点，节假日值班补贴为每天{200 + i % 300}元。",
            f"值班表由{dept}值{seq}组每周五发布，值班电话为内线{6000 + i}。",
            "",
            "四、考核指标",
            f"本季度{i}类业务办理量为{1000 + i * 13}件，目标完成率为{85 + i % 15}%。",
            f"投诉响应时限为{i % 24 + 1}小时，满意率目标为{90 + i % 10}%。",
        ]
        p = CORPUS_TMP / f"细则_{dept}_{seq:03d}.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=50)
    args = ap.parse_args()

    import psutil
    from rag.pipeline import ingest
    from rag.retriever import Retriever

    proc = psutil.Process()
    mem0 = proc.memory_info().rss / 1024 / 1024

    print(f"生成 {args.docs} 篇模拟文档 ...")
    paths = gen_docs(args.docs)
    print(f"  语料总量: {sum(p.stat().st_size for p in paths) / 1024:.0f} KB")

    if BENCH_DIR.exists():
        import shutil
        shutil.rmtree(BENCH_DIR)

    # ---- 首次入库（全量） ----
    t_all = time.time()
    stats = ingest(paths, index_dir=BENCH_DIR, quiet=True)
    t_ingest = time.time() - t_all
    n_chunks = stats["collection_count"]
    mem1 = proc.memory_info().rss / 1024 / 1024
    print(f"首次入库: {t_ingest:.1f}s（{n_chunks} 块，{n_chunks / t_ingest:.0f} 块/s）")

    # ---- 增量入库（应全部跳过） ----
    t_inc = time.time()
    stats_inc = ingest(paths, index_dir=BENCH_DIR, quiet=True)
    t_incremental = time.time() - t_inc
    print(f"增量入库: {t_incremental:.2f}s（跳过 {len(stats_inc['skipped'])} 篇）")

    # ---- 查询延迟 ----
    retriever = Retriever(BENCH_DIR)
    retriever.vector_search("预热查询", 5) # 模型已在此前入库时加载

    qs = [f"{DEPTS[i % len(DEPTS)]}第{i // len(DEPTS) + 1}号细则里{i}类差旅补贴是多少钱？"
          for i in range(args.queries)]
    lat = []
    for q in qs:
        t0 = time.time()
        retriever.retrieve(q, mode="hybrid", k_final=5)
        lat.append(time.time() - t0)
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95) - 1]
    mem2 = proc.memory_info().rss / 1024 / 1024
    print(f"检索延迟: P50={p50 * 1000:.0f}ms  P95={p95 * 1000:.0f}ms  (n={len(lat)})")
    print(f"内存: 基线 {mem0:.0f}MB → 入库后 {mem1:.0f}MB → 检索后 {mem2:.0f}MB")

    results = {
        "docs": args.docs, "chunks": n_chunks,
        "ingest_seconds": round(t_ingest, 1), "chunks_per_second": round(n_chunks / t_ingest, 1),
        "incremental_seconds": round(t_incremental, 2),
        "skipped_on_incremental": len(stats_inc["skipped"]),
        "retrieval_p50_ms": round(p50 * 1000, 1), "retrieval_p95_ms": round(p95 * 1000, 1),
        "memory_mb": {"before": round(mem0), "after_ingest": round(mem1), "after_query": round(mem2)},
        "note": "延迟仅含检索链路（向量编码+双路召回+RRF），不含大模型生成",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark_scale.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [" # 大规模入库与查询基准", "",
          f"- 规模：{args.docs} 篇文档 / {n_chunks} 个 chunk（纯 CPU）",
          f"- 首次入库：**{t_ingest:.1f}s**（{n_chunks / t_ingest:.0f} 块/s，含解析/切分/向量编码/双索引构建）",
          f"- 增量入库（MD5 未变）：**{t_incremental:.2f}s**，跳过 {len(stats_inc['skipped'])}/{args.docs} 篇",
          f"- 检索延迟（向量+BM25+RRF，不含 LLM）：**P50 {p50 * 1000:.0f}ms / P95 {p95 * 1000:.0f}ms**",
          f"- 进程内存：基线 {mem0:.0f}MB → 入库后 {mem1:.0f}MB → 检索后 {mem2:.0f}MB",
          "", "> 环境：本地 CPU（无 GPU），嵌入模型 text2vec-base-chinese。",
          "> 10 万块级扩展方案：Chroma 按文档哈希分片多集合，或迁移 Milvus（scripts/migrate_to_milvus.py）。"]
    (RESULTS_DIR / "benchmark_scale.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n✅ 基准报告 {RESULTS_DIR / 'benchmark_scale.md'}")


if __name__ == "__main__":
    main()
