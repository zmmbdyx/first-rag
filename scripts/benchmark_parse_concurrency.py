#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解析阶段并发模型基准（纯解析/切分，不含嵌入）。

为什么单独测解析：
    入库总耗时里嵌入向量化往往占 95%+，而嵌入是**批处理**、不随解析并发模型变化。
    把两者混在一起测，会把真正的差异稀释到看不见。本脚本只测「解析+切分」这段
    真正受并发模型影响的代码，从而给出可信的对比。

对比三种模型：
    serial  逐文件串行解析
    thread  ThreadPoolExecutor 线程池
    async   asyncio 事件循环 + asyncio.to_thread（本项目入库默认路径）

用法：
    python scripts/benchmark_parse_concurrency.py --dir .bench_big --workers 8 --repeat 3
    python scripts/benchmark_parse_concurrency.py --dir data/corpus
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.pipeline import _gather_parse, _parse_and_chunk, collect_files # noqa: E402


def run_serial(files, chunker):
    return [_parse_and_chunk(f, chunker) for f in files]


def run_thread(files, chunker, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda f: _parse_and_chunk(f, chunker), files))


def run_async(files, chunker, workers):
    todo = [(f, "") for f in files]
    return asyncio.run(_gather_parse(todo, chunker, workers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--chunker", default="smart", choices=["smart", "naive"])
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "benchmark_parse_concurrency.json"))
    args = ap.parse_args()

    files = collect_files([Path(args.dir)])
    if not files:
        print(f"未找到文档：{args.dir}")
        return 2
    total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024

    print("=" * 72)
    print("解析阶段并发模型基准（不含嵌入）")
    print("=" * 72)
    print(f"目录: {args.dir}")
    print(f"文档数: {len(files)}   总大小: {total_mb:.2f} MB   并发度: {args.workers}   重复: {args.repeat}")

    timings: dict[str, list[float]] = {"serial": [], "thread": [], "async": []}
    chunk_counts = {}

    for name, fn in (("serial", lambda: run_serial(files, args.chunker)),
                     ("thread", lambda: run_thread(files, args.chunker, args.workers)),
                     ("async", lambda: run_async(files, args.chunker, args.workers))):
        # 预热一次（排除首次导入/懒加载开销）
        fn()
        for i in range(args.repeat):
            t = time.time()
            res = fn()
            dt = time.time() - t
            timings[name].append(dt)
            chunk_counts[name] = sum(len(c) for _, c in res)
            print(f"  [{name:<7}] 第{i+1}次  {dt:7.3f}s   chunks={chunk_counts[name]}")

    med = {k: statistics.median(v) for k, v in timings.items()}
    print("\n" + "=" * 72)
    print("结果（中位数）")
    print("=" * 72)
    for k in ("serial", "thread", "async"):
        print(f"  {k:<8} {med[k]:7.3f}s   加速比 vs 串行: {med['serial']/med[k]:5.2f}×")

    def gain(base, new):
        return (base - new) / base * 100 if base else 0.0

    print(f"\n线程池 vs 串行:    {gain(med['serial'], med['thread']):+.1f}% 耗时"
          f"（{med['serial']:.3f}s → {med['thread']:.3f}s）")
    print(f"asyncio vs 串行:   {gain(med['serial'], med['async']):+.1f}% 耗时"
          f"（{med['serial']:.3f}s → {med['async']:.3f}s）")
    print(f"asyncio vs 线程池: {gain(med['thread'], med['async']):+.1f}% 耗时")
    print("\n说明：解析库为同步阻塞实现，并发收益来自并行等待 IO 与释放 GIL 的 C 扩展计算；")
    print("      纯 Python CPU 计算受 GIL 限制不会因此加速。")

    out = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "dir": str(args.dir), "docs": len(files), "total_mb": round(total_mb, 2),
        "workers": args.workers, "repeat": args.repeat, "chunker": args.chunker,
        "median_s": {k: round(v, 4) for k, v in med.items()},
        "speedup_vs_serial": {k: round(med["serial"] / v, 3) for k, v in med.items() if v},
        "chunks": chunk_counts,
        "thread_vs_serial_pct": round(gain(med["serial"], med["thread"]), 1),
        "async_vs_serial_pct": round(gain(med["serial"], med["async"]), 1),
        "async_vs_thread_pct": round(gain(med["thread"], med["async"]), 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
