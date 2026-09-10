"""查询改写模块测试（纯规则部分，离线可跑）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.rewrite import needs_rewrite, rewrite_query, rule_rewrite # noqa: E402

HIST = [{"role": "user", "content": "试用期一般是多长时间？"},
        {"role": "assistant", "content": "试用期一般为3个月。"}]


def test_no_history_never_rewrites():
    assert not needs_rewrite("试用期多长时间？", [])
    assert rewrite_query("试用期多长时间？", [])["method"] == "none"


def test_needs_rewrite_signals():
    assert needs_rewrite("那转正后呢？", HIST)
    assert needs_rewrite("它保修几年？", HIST)
    assert needs_rewrite("上面说的补贴呢", HIST)
    assert not needs_rewrite("停车费怎么报销？", HIST) # 无指代信号的完整问题


def test_rule_rewrite_merges_topic():
    out = rule_rewrite("那转正后呢？", HIST)
    assert "试用期一般是多长时间" in out and "转正后" in out


def test_rewrite_query_fallback_to_rule(monkeypatch):
    # LLM 不可用时降级为规则改写
    import rag.rewrite as rw

    monkeypatch.setattr(rw, "llm_rewrite", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")))
    out = rewrite_query("那年假呢？", HIST, use_llm=True)
    assert out["method"] == "rule"
    assert "年假" in out["query"]
