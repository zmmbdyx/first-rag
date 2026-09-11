"""量化评测语料的冗余度与「答案可区分性」。

两个会显著抬高召回率的隐患：
  1. 语料存在模板化重复 —— 同一句话在多篇文档里逐字出现时，检索只要命中
     任一篇即算成功，题目变得远比真实场景容易；
  2. 金标答案片段在语料里出现多次（非唯一），命中"任意一份"就算召回。

本脚本不改动任何东西，只统计事实，供判断指标可信度。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "data" / "corpus"
SAMPLES_DIR = ROOT / "data" / "samples"
QUESTIONS = ROOT / "eval" / "questions_v2.jsonl"

_norm = lambda s: re.sub(r"[^\w\u4e00-\u9fff]+", "", s.lower())  # noqa: E731


def read_corpus() -> dict[str, str]:
    """返回 {文档名: 规范化全文}。只用文本类文档，二进制格式无法直接读。"""
    docs: dict[str, str] = {}
    for d in (CORPUS_DIR, SAMPLES_DIR):
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".txt", ".md"):
                docs[p.name] = p.read_text(encoding="utf-8", errors="ignore")
    return docs


def main() -> int:
    docs = read_corpus()
    print(f"=== 可读语料：{len(docs)} 篇文本类文档 ===")
    for d in (CORPUS_DIR, SAMPLES_DIR):
        n = len(list(d.iterdir())) if d.exists() else 0
        kinds = Counter(p.suffix.lower() for p in d.iterdir()) if d.exists() else {}
        print(f"  {d.relative_to(ROOT)}: {n} 个文件  {dict(kinds)}")

    # ---- 1) 句子级重复：同一句话出现在多少篇文档里 ----
    sent_owners: dict[str, set[str]] = defaultdict(set)
    for name, text in docs.items():
        for raw in re.split(r"[。！？\n]", text):
            s = _norm(raw)
            if len(s) >= 12:  # 忽略过短的碎片
                sent_owners[s].add(name)

    dup = {s: o for s, o in sent_owners.items() if len(o) > 1}
    total_sent = len(sent_owners)
    print(f"\n=== 1) 句子级跨文档重复 ===")
    print(f"  去重后句子总数: {total_sent}")
    print(f"  出现在多篇文档中的句子: {len(dup)}  ({len(dup)/max(total_sent,1):.1%})")
    if dup:
        worst = sorted(dup.items(), key=lambda kv: -len(kv[1]))[:3]
        for s, owners in worst:
            print(f"    · 出现在 {len(owners)} 篇: {s[:60]}…")

    # ---- 2) 文档两两相似度（3-gram Jaccard） ----
    def grams(text: str, n: int = 3) -> set[str]:
        s = _norm(text)
        return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}

    names = sorted(docs)
    gsets = {n: grams(docs[n]) for n in names}
    pairs: list[tuple[float, str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga, gb = gsets[a], gsets[b]
            if not ga or not gb:
                continue
            j = len(ga & gb) / len(ga | gb)
            pairs.append((j, a, b))
    pairs.sort(reverse=True)
    print(f"\n=== 2) 文档两两相似度（字符 3-gram Jaccard）===")
    if pairs:
        import statistics

        vals = [p[0] for p in pairs]
        print(f"  配对数: {len(pairs)}  均值 {statistics.mean(vals):.3f}  "
              f"中位 {statistics.median(vals):.3f}  最大 {max(vals):.3f}")
        print("  最相似的 5 对:")
        for j, a, b in pairs[:5]:
            print(f"    {j:.3f}  {a}  ↔  {b}")

    # ---- 3) 金标答案片段的唯一性 ----
    print(f"\n=== 3) 金标答案的落库与唯一性（对照真实索引块）===")
    if not QUESTIONS.exists():
        print("  （未找到 questions_v2.jsonl）")
        return 0
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = [q for q in qs if q.get("answerable")]
    print(f"  可回答题: {len(answerable)} / {len(qs)}")

    # 用评测索引里的**实际块文本**来核对，而不是只比 .txt/.md 源文件。
    # 源文件里 7 篇 .docx + 4 篇 .pdf 无法直接读，上一版因此误报"92 题找不到"，
    # 那个数字毫无意义——检索面对的是解析后的块，不是原始文件。
    chunk_texts: list[tuple[str, str]] = []  # (doc_name, normalized_text)
    idx_root = ROOT / "eval" / ".eval_db" / "hybrid"
    try:
        sys.path.insert(0, str(ROOT))
        from rag.retriever import Retriever

        r = Retriever(idx_root, "rag_chunks")
        data = r.collection.get(include=["documents", "metadatas"])
        for text, meta in zip(data["documents"], data["metadatas"]):
            chunk_texts.append((str((meta or {}).get("doc_name", "")), _norm(text or "")))
        print(f"  已载入索引块: {len(chunk_texts)} 条（来自 eval/.eval_db/hybrid）")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 无法载入索引块（{type(e).__name__}: {e}）")
        print("     先跑 `python scripts/run_eval.py --retrieval-only` 生成索引后再核对。")

    if chunk_texts:
        # 判定必须与 run_eval.py 的 eval_retrieval 保持一致，否则会得出错误结论：
        #   * 单跳题：一个块里同时包含 gold_answer 的**全部**分片（且文档名匹配）；
        #   * 多跳题：gold 是 [{doc, spans}...]，**每个金标项各自**被某个块满足即可，
        #     不同项允许落在不同文档 —— 这正是"多跳"的含义。
        # 曾把多跳题也按"全部分片须在同一文档"来查，结果误报 10 道题"金标不存在"，
        # 实际它们的分片分散在不同文档里，完全正常。
        miss: list[int] = []
        multi_doc: list[int] = []
        per_doc: dict[str, set[int]] = defaultdict(set)

        def _item_ok(doc: str, spans: list[str]) -> bool:
            return any(d == doc and all(sp in t for sp in spans) for d, t in chunk_texts)

        for q in answerable:
            if q.get("qtype") == "multi_hop" and isinstance(q.get("gold"), list):
                items = [(g.get("doc", ""),
                          [s for s in (_norm(x) for x in g.get("spans", [])) if s])
                         for g in q["gold"]]
                if all(_item_ok(d, sp) for d, sp in items if sp):
                    per_doc[q["gold_doc"]].add(q["id"])
                    if len({d for d, _ in items}) > 1:
                        multi_doc.append(q["id"])
                else:
                    miss.append(q["id"])
                continue

            spans = [s for s in (_norm(x) for x in re.split(r"[；;\n]", q.get("gold_answer", ""))) if s]
            if not spans:
                continue
            if _item_ok(q["gold_doc"], spans):
                per_doc[q["gold_doc"]].add(q["id"])
            else:
                miss.append(q["id"])

        print(f"  金标在索引中【无法命中】的题: {len(miss)} 题 "
              f"({miss[:10]}{'…' if len(miss) > 10 else ''})")
        print("     → 应当为 0；不为 0 说明题集与语料不一致（或金标非原文）。")
        print(f"  跨文档多跳题（金标分布在多篇）: {len(multi_doc)} 题 {multi_doc[:8]}")
        print("     → 这类题必须靠真实的跨文档召回才能命中，区分度最高。")
        print("  单篇文档承载题量 Top5: "
              f"{sorted(((len(v), k) for k, v in per_doc.items()), reverse=True)[:5]}")
        print("     → 某篇承载过多时，该文档的检索质量会主导整体指标。")

    # ---- 4) 重复句式：同一 15 字窗口出现在多篇 ----
    win_owners: dict[str, set[str]] = defaultdict(set)
    for name, text in docs.items():
        s = _norm(text)
        for i in range(0, max(len(s) - 15, 1), 5):
            win_owners[s[i:i + 15]].add(name)
    rep = sum(1 for o in win_owners.values() if len(o) > 1)
    print(f"\n=== 4) 15 字滑窗跨文档重复 ===")
    print(f"  去重窗口数: {len(win_owners)}")
    print(f"  跨多篇重复的窗口: {rep} ({rep/max(len(win_owners),1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
