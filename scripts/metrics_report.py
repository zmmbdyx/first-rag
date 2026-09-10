"""运行指标报告：P50/P95/P99 延迟、错误率、Token 消耗（读取 logs/metrics.db）。

用法：python scripts/metrics_report.py [--last 200]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.metrics import aggregate # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=500)
    args = ap.parse_args()

    stats = aggregate(last_n=args.last)
    if not stats:
        print("暂无指标数据（logs/metrics.db 不存在或为空）——先通过 chat() 跑一些问答。")
        return
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"\n最近 {stats['n']} 次问答："
          f"\n  检索 P50 {stats['p50_retrieval_ms']}ms｜生成 P50/P95/P99 = "
          f"{stats['p50_gen_ms']}/{stats['p95_gen_ms']}/{stats['p99_gen_ms']}ms"
          f"\n  错误率 {stats['error_rate']:.1%}｜Token 合计 "
          f"{stats['prompt_tokens_total']}+{stats['completion_tokens_total']}"
          f"（均值 {stats['avg_tokens_per_query']}/次）")


if __name__ == "__main__":
    main()
