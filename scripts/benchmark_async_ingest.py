#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""异步入库基准：对比「串行 / 线程池 / asyncio+线程池」三种解析并发模型的入库耗时。

方法学（避免数字注水）：
1. 三种模式跑**同一批文档**、同一个空索引目录，唯一变量是解析并发模型；
2. 分开记录「解析阶段」与「总耗时」——嵌入向量化是批处理且不随并发模型变化，
   混在一起会稀释差异，因此单独报 parse_elapsed；
3. 每个模式重复 N 次取中位数，减少单次抖动（磁盘缓存 / OS 调度）；
4. 如实说明：解析库（PyMuPDF/python-docx）是同步阻塞的，且底层计算会释放 GIL，
   所以线程/异步并发的收益来自并行等待 IO 与 C 扩展计算，而非绕过 GIL。

用法：
    python scripts/benchmark_async_ingest.py # 默认 data/corpus
    python scripts/benchmark_async_ingest.py --dir data/samples --repeat 3
"""
import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.pipeline import collect_files, ingest # noqa: E402


def _run_once(files_dir: Path, mode: str, workers: int, tmp_root: Path) -> dict:
    """在一个干净索引目录里跑一次入库，返回耗时统计。"""
    idx = tmp_root / f"idx_{mode}_{workers}_{time.time_ns()}"
    idx.mkdir(parents=True, exist_ok=True)

    if mode == "serial":
        # 串行：workers=1 且关闭异步
        stats = ingest([files_dir], index_dir=idx, incremental=False,
                       workers=1, async_parse=False, quiet=True)
    elif mode == "thread":
        stats = ingest([files_dir], index_dir=idx, incremental=False,
                       workers=workers, async_parse=False, quiet=True)
    elif mode == "async":
        stats = ingest([files_dir], index_dir=idx, incremental=False,
                       workers=workers, async_parse=True, quiet=True)
    else:
        raise ValueError(mode)

    shutil.rmtree(idx, ignore_errors=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "corpus"), help="待入库文档目录")
    ap.add_argument("--workers", type=int, default=8, help="并发度")
    ap.add_argument("--repeat", type=int, default=3, help="每模式重复次数（取中位数）")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "benchmark_async_ingest.json"))
    args = ap.parse_args()

    files_dir = Path(args.dir)
    files = collect_files([files_dir])
    if not files:
        print(f"未找到文档：{files_dir}")
        return 2

    print("=" * 70)
    print("异步入库基准")
    print("=" * 70)
    print(f"文档目录: {files_dir}")
    print(f"文档数: {len(files)}   并发度: {args.workers}   重复: {args.repeat}")

    tmp_root = Path(tempfile.mkdtemp(prefix="ingest_bench_"))
    results: dict[str, list[dict]] = {"serial": [], "thread": [], "async": []}

    try:
        for mode in ("serial", "thread", "async"):
            label = {"serial": "串行解析", "thread": f"线程池解析({args.workers})",
                     "async": f"asyncio+线程池({args.workers})"}[mode]
            for i in range(args.repeat):
                t = time.time()
                st = _run_once(files_dir, mode, args.workers, tmp_root)
                wall = time.time() - t
                results[mode].append(st)
                print(f"  [{label}] 第{i+1}次  解析 {st['parse_elapsed']:6.2f}s  "
                      f"入库总计 {st['elapsed']:6.2f}s  chunk={st['new_chunks']}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    def med(mode, key):
        vals = [r[key] for r in results[mode] if key in r]
        return statistics.median(vals) if vals else 0.0

    print("\n" + "=" * 70)
    print("结果（中位数）")
    print("=" * 70)
    print(f"{'模式':<24}{'解析(s)':>10}{'入库总计(s)':>14}")
    for mode, label in (("serial", "串行解析"), ("thread", "线程池解析"),
                        ("async", "asyncio+线程池")):
        print(f"{label:<24}{med(mode,'parse_elapsed'):>10.2f}{med(mode,'elapsed'):>14.2f}")

    s_parse, t_parse, a_parse = med("serial", "parse_elapsed"), med("thread", "parse_elapsed"), med("async", "parse_elapsed")
    s_all, a_all = med("serial", "elapsed"), med("async", "elapsed")

    def gain(base, new):
        return (base - new) / base * 100 if base else 0.0

    print("\n解析阶段：")
    if t_parse:
        print(f"  线程池 vs 串行:      {gain(s_parse, t_parse):+.1f}% 耗时"
              f"（{s_parse:.2f}s → {t_parse:.2f}s，加速 {s_parse/t_parse:.2f}×）")
    if a_parse:
        print(f"  asyncio vs 串行:     {gain(s_parse, a_parse):+.1f}% 耗时"
              f"（{s_parse:.2f}s → {a_parse:.2f}s，加速 {s_parse/a_parse:.2f}×）")
    if a_parse and t_parse:
        print(f"  asyncio vs 线程池:   {gain(t_parse, a_parse):+.1f}% 耗时")
    print("\n入库总耗时（含嵌入，嵌入为批处理且不随并发模型变化）：")
    print(f"  asyncio vs 串行:     {gain(s_all, a_all):+.1f}% 耗时"
          f"（{s_all:.2f}s → {a_all:.2f}s）")
    print("\n说明：收益来自并发等待阻塞式解析库的 IO 与释放 GIL 的 C 扩展计算；"
          "纯 Python CPU 计算因 GIL 不会因此加速。")

    out = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "dir": str(files_dir), "docs": len(files), "workers": args.workers,
        "repeat": args.repeat,
        "median": {m: {"parse_elapsed": med(m, "parse_elapsed"),
                       "elapsed": med(m, "elapsed"),
                       "chunks": med(m, "new_chunks")} for m in results},
        "parse_gain_thread_vs_serial_pct": round(gain(s_parse, t_parse), 1),
        "parse_gain_async_vs_serial_pct": round(gain(s_parse, a_parse), 1),
        "total_gain_async_vs_serial_pct": round(gain(s_all, a_all), 1),
        "parse_speedup_async_vs_serial": round(s_parse / a_parse, 2) if a_parse else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
