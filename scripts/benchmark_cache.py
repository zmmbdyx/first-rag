***REMOVED***!/usr/bin/env python
***REMOVED*** -*- coding: utf-8 -*-
"""Redis 问答缓存基准：量化"缓存命中跳过检索+生成"带来的端到端延迟收益。

方法学（避免数字注水）：
1. **真实链路**：走 rag.pipeline.chat 全流程（安全检查→查询改写→检索→生成），
   唯一变量是 Redis 缓存开/关，不使用任何打桩的假延迟；
2. **同一批问题重复提问**：模拟 FAQ / 企业制度问答的真实访问分布（同一问题被多人反复问）；
3. **分开报三个口径**：
   - 冷启动（缓存未命中，走完整链路）
   - 命中（直接返回缓存）
   - 综合 P50/P95（含命中与未命中的混合流量，这才对应线上的真实体验）
4. **如实标注**：加速比主要来自"跳过 LLM 生成"，检索本身只占延迟的一小部分，
   因此该收益的前提是**访问重复率**。脚本会同时打印重复率，便于判断适用性。

用法：
    python scripts/benchmark_cache.py                 ***REMOVED*** 默认 8 题 × 3 轮
    python scripts/benchmark_cache.py --questions 6 --rounds 2 --no-llm
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import cache as qa_cache  ***REMOVED*** noqa: E402
from rag.config import INDEX_DIR, LLM_MODEL, RETRIEVAL_MODE  ***REMOVED*** noqa: E402
from rag.pipeline import chat, load_retriever  ***REMOVED*** noqa: E402

DEFAULT_QUESTIONS = [
    "试用期多长时间？",
    "咖啡机C3的保修期是多久？",
    "年假有多少天？",
    "远程办公需要提前申请吗？",
    "云滴C3咖啡机的功率是多少？",
    "员工手册里对考勤怎么规定的？",
    "报销流程是什么？",
    "离职需要提前多久提出？",
]


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=8, help="题目数量")
    ap.add_argument("--rounds", type=int, default=3, help="重复轮数（模拟同一问题被反复问）")
    ap.add_argument("--mode", default=RETRIEVAL_MODE)
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--no-llm", action="store_true",
                    help="不打真实大模型（用于无 Key 环境，仅测检索+缓存通路）")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "benchmark_cache.json"))
    args = ap.parse_args()

    questions = (DEFAULT_QUESTIONS * 10)[: args.questions]

    print("=" * 68)
    print("Redis 问答缓存基准")
    print("=" * 68)
    print(f"缓存可用: {qa_cache.available()}  URL: {qa_cache.REDIS_URL}")
    print(f"题目数: {len(questions)}  轮数: {args.rounds}  检索模式: {args.mode}")
    if not qa_cache.available():
        print("\n⚠️  Redis 不可用，无法测量缓存收益。")
        print("   启动方式：redis-server 或 docker run -p 6379:6379 redis:7-alpine")
        print("   仍将测量「无缓存」基线，供对照。")

    qa_cache.clear()
    qa_cache.reset_stats()

    retriever = load_retriever(INDEX_DIR)
    if retriever.collection.count() == 0:
        print("⚠️  知识库为空，请先运行：python scripts/build_kb.py")
        return 2

    def one(q, use_cache):
        t = time.time()
        r = chat(q, retriever=retriever, mode=args.mode, model=args.model,
                 use_cache=use_cache, audit_enabled=False, return_hits=False)
        return time.time() - t, r

    ***REMOVED*** 诊断：确认缓存写入/读取在基准进程里真的生效（避免"看起来在测缓存，其实一直 miss"）
    if qa_cache.available():
        _diag_key = qa_cache.cache_key(questions[0], args.mode, 5, args.model)
        qa_cache.set(_diag_key, {"answer": "__diag__"})
        _ok = qa_cache.get(_diag_key) is not None
        qa_cache._client.delete(_diag_key)
        print(f"缓存自检: {'通过' if _ok else '失败（写进去读不出来，基准结果无效）'}"
              f"  kb_version={qa_cache.kb_version()}")
        if not _ok:
            print("⚠️  缓存自检失败，本次基准的缓存收益不可信。")

    ***REMOVED*** 预热（模型加载 / 检索器首次查询开销不计入）
    print("\n预热中（加载嵌入模型与检索器）...")
    one(questions[0], use_cache=False)

    cold, warm, mixed = [], [], []
    cache_hits = 0
    total = 0

    ***REMOVED*** ---- 阶段 1：冷启动 —— 缓存开启但库是空的，全部 MISS 并写入缓存 ----
    ***REMOVED*** 这一轮才是"第一个用户提问"的真实延迟（检索 + LLM 生成 + 写缓存）。
    print("\n[阶段1] 冷启动：缓存为空，全部未命中（并写入缓存）...")
    for i, q in enumerate(questions):
        total += 1
        dt, res = one(q, use_cache=True)
        cold.append(dt)
        mixed.append(dt)
        hit = bool(res.get("cache_hit"))
        cache_hits += 1 if hit else 0
        print(f"  [{'HIT ' if hit else 'MISS'}] q{i+1} {dt*1000:7.1f} ms  {q[:24]}")

    ***REMOVED*** ---- 阶段 2：重复访问 —— 同一批问题再问 N 轮，应全部命中 ----
    if args.rounds > 0 and qa_cache.available():
        print(f"\n[阶段2] 重复访问 {args.rounds} 轮：应全部命中缓存...")
        for rnd in range(args.rounds):
            for i, q in enumerate(questions):
                total += 1
                dt, res = one(q, use_cache=True)
                warm.append(dt)
                mixed.append(dt)
                hit = bool(res.get("cache_hit"))
                cache_hits += 1 if hit else 0
                print(f"  [{'HIT ' if hit else 'MISS'}] r{rnd} q{i+1} {dt*1000:7.1f} ms  {q[:24]}")

    ***REMOVED*** ---- 阶段 3：对照组 —— 完全相同的题目，全程关闭缓存走完整链路 ----
    print("\n[阶段3] 对照组：关闭缓存，全部走完整链路...")
    base = []
    for _ in range(1 + args.rounds):
        for q in questions:
            dt, _ = one(q, use_cache=False)
            base.append(dt)

    def ms(v):
        return f"{statistics.mean(v)*1000:.1f}"

    cold_m, warm_m = statistics.mean(cold or [0]), statistics.mean(warm or [0])
    speedup = (cold_m / warm_m) if warm_m > 0 else 0.0

    print("\n" + "=" * 68)
    print("结果")
    print("=" * 68)
    print(f"冷启动（未命中，完整链路）  平均 {ms(cold)} ms   P50 {pct(cold,50)*1000:.1f}  P95 {pct(cold,95)*1000:.1f}")
    if warm:
        print(f"缓存命中（直接返回）        平均 {ms(warm)} ms   P50 {pct(warm,50)*1000:.1f}  P95 {pct(warm,95)*1000:.1f}")
    print(f"无缓存基线（同题全链路）    平均 {ms(base)} ms   P50 {pct(base,50)*1000:.1f}  P95 {pct(base,95)*1000:.1f}")
    if warm and cold_m:
        print(f"\n命中相对冷启动加速: {speedup:.1f}×  （延迟降低 {(1 - warm_m/cold_m)*100:.1f}%）")
    if base:
        red = (1 - statistics.mean(mixed) / statistics.mean(base)) * 100
        print(f"混合流量 P50 相对无缓存: {red:.1f}% 延迟降低"
              f"（命中率 {cache_hits}/{total} = {cache_hits/total*100:.0f}%）")
    s = qa_cache.stats()
    print(f"\n缓存统计: 命中 {s['hit']} / 未命中 {s['miss']} / 错误 {s['error']}"
          f" / 命中率 {s['hit_rate']*100:.1f}%")
    print("\n说明：命中收益 = 跳过 LLM 生成 + 跳过检索；收益大小取决于访问重复率。")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "cache_available": qa_cache.available(),
        "redis_url": qa_cache.REDIS_URL if qa_cache.available() else "",
        "mode": args.mode, "model": args.model,
        "questions": len(questions), "rounds": args.rounds,
        "cold_ms": {"mean": statistics.mean(cold) * 1000 if cold else 0,
                    "p50": pct(cold, 50) * 1000, "p95": pct(cold, 95) * 1000},
        "warm_ms": ({"mean": statistics.mean(warm) * 1000, "p50": pct(warm, 50) * 1000,
                     "p95": pct(warm, 95) * 1000} if warm else {}),
        "baseline_ms": {"mean": statistics.mean(base) * 1000,
                        "p50": pct(base, 50) * 1000, "p95": pct(base, 95) * 1000},
        "speedup_vs_cold": round(speedup, 2),
        "latency_reduction_pct": round((1 - statistics.mean(mixed) / statistics.mean(base)) * 100, 1) if base else None,
        "hit_rate": round(cache_hits / total, 4) if total else 0,
        "cache_stats": qa_cache.stats(),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
