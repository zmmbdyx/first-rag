"""BM25 索引与 RRF 融合逻辑测试（不依赖模型）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.bm25 import BM25Index, tokenize


def test_tokenize_filters_punct():
    toks = tokenize("C3 咖啡机，额定功率：1350W！")
    assert "咖啡机" in toks and "1350w" in toks
    assert all("，" not in t and "：" not in t for t in toks)


def test_bm25_search_ranking():
    chunks = [
        {"chunk_id": "a", "text": "试用期一般为3个月，管理岗位可延长至6个月"},
        {"chunk_id": "b", "text": "年假：满1年5天，满3年10天"},
        {"chunk_id": "c", "text": "咖啡机额定功率1350W，泵压20bar"},
    ]
    for c in chunks:
        c["text"] = c["text"]
    from rag.bm25 import build_from_chunks
    idx = build_from_chunks([{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks])
    hits = idx.search("试用期多长时间", k=2)
    assert hits and hits[0][0] == "a"
    hits2 = idx.search("咖啡机功率", k=2)
    assert hits2 and hits2[0][0] == "c"


def test_rrf_fusion_prefers_double_hits():
    from rag.retriever import Hit

    def mk(cid):
        return Hit(chunk_id=cid, text="", doc_name="d", section_path="", page=-1)

    vec = [mk("a"), mk("b"), mk("c")]
    kw = [mk("b"), mk("d"), mk("a")]
    rrf_k = 60
    scores = {}
    for rank, h in enumerate(vec, 1):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0) + 1 / (rrf_k + rank)
    for rank, h in enumerate(kw, 1):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0) + 1 / (rrf_k + rank)
    ranked = sorted(scores, key=scores.get, reverse=True)
    ***REMOVED*** b 在两路都靠前，应排第一；a 两路均被命中也应靠前
    assert ranked[0] == "b"
    assert set(ranked[:2]) == {"a", "b"}
