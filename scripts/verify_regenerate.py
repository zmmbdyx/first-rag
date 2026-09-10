"""验证「重新生成」是覆盖而不是追加。

流程：
  1. 新建会话，问一个问题 → 库里应是 2 条消息（user + assistant）；
  2. 调 regenerate=true → 库里仍应是 2 条，且**只有一条**用户提问；
  3. 再问第二个问题 → 4 条；对第二轮 regenerate → 仍是 4 条。

如果实现错误（把 regenerate 当成新一轮），消息数会变成 4 并出现两条相同提问。

    python scripts/verify_regenerate.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def post(path: str, payload: dict) -> dict:
    r = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    r.add_header("Content-Type", "application/json; charset=utf-8")
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sse_chat(payload: dict) -> tuple[str, dict]:
    """跑一次流式问答，返回 (回答正文, done 事件)。"""
    r = urllib.request.Request(
        f"{BASE}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    r.add_header("Content-Type", "application/json; charset=utf-8")
    event, content, done, error = "message", [], {}, ""
    with urllib.request.urlopen(r, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "content":
                    content.append(data.get("delta", ""))
                elif event == "done":
                    done = data
                elif event == "error":
                    error = data
            elif not line:
                event = "message"
    if error:
        raise RuntimeError(error)
    return "".join(content), done


def main() -> int:
    conv = post("/api/conversations", {})
    cid = conv["conversation_id"]
    print(f"会话 ID: {cid}\n")

    print("=== 1) 首次提问 ===")
    a1, _ = sse_chat({"message": "员工手册里试用期是多久？", "conversation_id": cid})
    msgs = get(f"/api/conversations/{cid}/messages")
    check("消息数为 2（user + assistant）", len(msgs) == 2, f"{len(msgs)} 条")
    check("回答非空", len(a1) > 0, f"{len(a1)} 字")

    print("\n=== 2) 重新生成（应覆盖，不追加）===")
    a2, _ = sse_chat({"message": "", "conversation_id": cid, "regenerate": True})
    msgs2 = get(f"/api/conversations/{cid}/messages")
    users = [m for m in msgs2 if m["role"] == "user"]
    check("消息数仍为 2", len(msgs2) == 2, f"{len(msgs2)} 条")
    check("用户提问只有 1 条（没有重复气泡）", len(users) == 1, f"{len(users)} 条")
    check("用户提问内容未变", users and users[0]["content"] == "员工手册里试用期是多久？",
          repr(users[0]["content"] if users else None))
    check("重新生成了回答", len(a2) > 0, f"{len(a2)} 字")
    check("回答已替换（时间戳/内容由服务端落库）",
          msgs2[-1]["role"] == "assistant")

    listed = get("/api/conversations")
    row = next((c for c in listed if c["conversation_id"] == cid), {})
    check("message_count 已重算为 2", row.get("message_count") == 2, str(row.get("message_count")))

    print("\n=== 3) 第二轮：提问 + 重新生成 ===")
    sse_chat({"message": "IT服务台的报修流程是什么？", "conversation_id": cid})
    msgs3 = get(f"/api/conversations/{cid}/messages")
    check("第二轮后 4 条", len(msgs3) == 4, f"{len(msgs3)} 条")

    sse_chat({"message": "", "conversation_id": cid, "regenerate": True})
    msgs4 = get(f"/api/conversations/{cid}/messages")
    users4 = [m for m in msgs4 if m["role"] == "user"]
    check("重新生成后仍 4 条", len(msgs4) == 4, f"{len(msgs4)} 条")
    check("用户提问 2 条（每轮各一条）", len(users4) == 2, f"{len(users4)} 条")
    check("第二问保持为最后一条提问",
          users4 and users4[-1]["content"] == "IT服务台的报修流程是什么？",
          repr(users4[-1]["content"] if users4 else None))

    print("\n=== 4) 空会话调用 regenerate ===")
    empty = post("/api/conversations", {})
    try:
        sse_chat({"message": "", "conversation_id": empty["conversation_id"], "regenerate": True})
        check("空会话 regenerate 应报错", False, "未返回 error 事件")
    except RuntimeError as e:
        check("空会话 regenerate 返回明确错误", "没有可重新生成" in str(e), str(e))

    # 清理
    req = urllib.request.Request(f"{BASE}/api/conversations/{cid}", method="DELETE")
    urllib.request.urlopen(req, timeout=30).read()
    req = urllib.request.Request(f"{BASE}/api/conversations/{empty['conversation_id']}", method="DELETE")
    urllib.request.urlopen(req, timeout=30).read()

    print()
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 重新生成是覆盖语义，历史中无重复提问")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
