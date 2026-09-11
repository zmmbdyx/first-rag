"""验证本次治理改造的四项能力（离线可跑，不需要大模型）：

  1. **文档级 ACL**：不同用户组检索到的文档集合受权限约束，越权文档为 0；
  2. **置信度门限**：低相关问题被系统直接拒答，且**不调用大模型**；
  3. **审计脱敏**：落盘日志中的手机号/邮箱/身份证/API Key/IP 全部被掩码；
  4. **明细指标**：每次问答可按 request_id 追溯，并支持按用户归因 token。

    python scripts/check_governance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def main() -> int:  # noqa: C901
    print("=== 1) 文档级 ACL：检索结果受用户组约束 ===")
    from rag.chunking import Chunk
    from rag.retriever import is_visible
    from rag.vector_store import meta_to_tags, tags_to_meta

    # 标签在 Chroma 里以逗号分隔字符串存储，往返必须无损
    check("标签序列化往返", meta_to_tags(tags_to_meta(["hr", "admin"])) == ["hr", "admin"])
    check("标签去重保序", tags_to_meta(["hr", "hr", "admin"]) == "hr,admin")

    cases = [
        (["public"], ["hr"], True, "公开文档对任意组可见"),
        (["hr"], ["hr"], True, "hr 文档对 hr 组可见"),
        (["hr"], ["it"], False, "hr 文档对 it 组不可见 ← 核心断言"),
        (["finance"], ["hr"], False, "finance 文档对 hr 组不可见"),
        ([], ["hr"], True, "无标签(宽松模式)可见（兼容历史索引）"),
        (["hr"], ["admin"], True, "管理员可见全部"),
        (["hr"], [], True, "无用户上下文不限制（内部/CLI）"),
    ]
    for tags, groups, want, desc in cases:
        got = is_visible(tags, groups)
        check(desc, got == want, f"tags={tags} groups={groups} -> {got}")

    # 入库的 Chunk 默认带 public 标签
    c = Chunk(chunk_id="x", doc_name="d", section_path="", page=1, text="t")
    check("Chunk 默认权限标签为 public", c.perm_tags == ["public"], str(c.perm_tags))

    print("\n=== 2) 置信度门限：低置信直接拒答（不调用大模型）===")
    from rag import confidence as conf
    from rag.config import ANSWER_THRESHOLD, CAUTION_THRESHOLD

    class H:
        def __init__(self, rerank_score=None, text="", score=0.0):
            self.rerank_score = rerank_score
            self.text = text
            self.score = score

    check("配置已设阈值", ANSWER_THRESHOLD is not None and CAUTION_THRESHOLD is not None,
          f"answer={ANSWER_THRESHOLD} caution={CAUTION_THRESHOLD}")

    # 高相关：rerank logit 很大 -> sigmoid 接近 1
    high = conf.assess([H(rerank_score=6.0, text="试用期一般为3个月")], "试用期多久")
    check("高相关 -> high 且不拒答", high.tier == conf.TIER_HIGH and not high.refuse,
          f"tier={high.tier} score={high.score:.3f}")

    # 中等相关：logit=-1.0 -> sigmoid≈0.269，落在 [caution, answer) 区间
    mid = conf.assess([H(rerank_score=-1.0, text="部分相关")], "试用期多久")
    check("中等相关 -> medium 且有提示", mid.tier == conf.TIER_MEDIUM and bool(mid.hint),
          f"tier={mid.tier} score={mid.score:.3f}")

    # 低相关：极负 logit -> sigmoid 接近 0
    low = conf.assess([H(rerank_score=-8.0, text="无关内容")], "量子计算机的退相干时间")
    check("低相关 -> low 且拒答", low.tier == conf.TIER_LOW and low.refuse,
          f"tier={low.tier} score={low.score:.4f}")

    # 空结果必须拒答
    empty = conf.assess([], "任意")
    check("空结果 -> 拒答", empty.refuse and empty.tier == conf.TIER_LOW, empty.basis)

    payload = conf.refusal_payload(low)
    check("拒答结果带 refused_by 标记", payload.get("refused_by") == "confidence")
    check("拒答话术固定可判定",
          payload["answer"] == conf.REFUSAL_TEXT, payload["answer"])

    print("\n=== 3) 审计脱敏：PII 不得原样落盘 ===")
    from rag import audit as audit_mod

    raw = ("手机 13800000000 邮箱 test@example.com 身份证 110101199003077777 "
           "key sk-abcdefghijklmnop1234qrst ip 192.168.1.10")
    cleaned, counts = audit_mod.redact_text(raw)
    for token in ("13800000000", "test@example.com", "110101199003077777",
                  "sk-abcdefghijklmnop1234qrst", "192.168.1.10"):
        check(f"已掩码 {token[:24]}", token not in cleaned)
    check("命中计数完整", len(counts) >= 5, json.dumps(counts, ensure_ascii=False))

    print("\n=== 4) 明细指标：可按 request_id 追溯、可按用户归因 ===")
    from rag import metrics as M

    rid = "selftest-governance-0001"
    M.record_request(
        rid, conversation_id="c1", message_id=1, user_hash="u-hash-1",
        model="m", mode="hybrid", query_used="q", retrieval_ms=12.5, gen_ms=800.0,
        total_ms=812.5, prompt_tokens=600, completion_tokens=60, n_hits=5,
        top_score=0.81, confidence_tier="high", confidence_basis="rerank",
        refused=False, acl_groups=["hr"], cache_hit=False,
    )
    row = M.get_request(rid)
    check("可按 request_id 取回明细", row is not None)
    if row:
        check("明细含置信度分档", row.get("confidence_tier") == "high", row.get("confidence_tier"))
        check("明细含权限组（审计越权用）", row.get("acl_groups") == "hr", row.get("acl_groups"))
        check("明细含 token（成本归因）",
              row.get("prompt_tokens") == 600, str(row.get("prompt_tokens")))

    by_user = M.cost_by_user(last_n_days=1)
    check("成本可按用户聚合", any(r.get("user_hash") == "u-hash-1" for r in by_user),
          f"{len(by_user)} 个用户")

    print("\n=== 5) 会话归属隔离 ===")
    from backend.models import init_db, session_scope
    from backend.services import conversation_service as cs

    init_db()
    with session_scope() as db:
        a = cs.create_conversation(db, title="A的会话", owner_id="user-a")
        b = cs.create_conversation(db, title="B的会话", owner_id="user-b")
        a_list = cs.list_conversations(db, owner_id="user-a")
        ids_a = {c.id for c in a_list}
        check("A 只看到自己的会话", a.id in ids_a and b.id not in ids_a,
              f"A 可见 {len(ids_a)} 个")
        check("owned_by 正确识别归属",
              cs.owned_by(a, "user-a") and not cs.owned_by(a, "user-b"))
        check("空身份（单用户模式）放行",
              cs.owned_by(cs.create_conversation(db, title="无主"), ""))
        cs.delete_conversation(db, a.id)
        cs.delete_conversation(db, b.id)

    print()
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 治理改造四项能力验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
