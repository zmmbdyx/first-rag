"""验证 /api/chat 现在会写运行指标（此前新后端一条都不写）。

流程：记录调用前的行数 → 打一次 /api/chat（强制不走缓存）→ 再数行数。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.metrics import DB_PATH  # noqa: E402

BASE = "http://127.0.0.1:8010"


def rows() -> list[tuple]:
    if not DB_PATH.exists():
        return []
    c = sqlite3.connect(DB_PATH)
    try:
        return c.execute("select * from chat_metrics order by rowid").fetchall()
    finally:
        c.close()


def post(path: str, payload: dict) -> dict:
    r = urllib.request.Request(f"{BASE}{path}",
                               data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               method="POST")
    r.add_header("Content-Type", "application/json; charset=utf-8")
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sse(path: str, payload: dict) -> str:
    r = urllib.request.Request(f"{BASE}{path}",
                               data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               method="POST")
    r.add_header("Content-Type", "application/json; charset=utf-8")
    event, content = "message", []
    with urllib.request.urlopen(r, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: ") and event == "content":
                content.append(json.loads(line[6:]).get("delta", ""))
            elif not line:
                event = "message"
    return "".join(content)


before = rows()
print(f"调用前 metrics 行数: {len(before)}")
if before:
    print("  最后一行:", before[-1])

conv = post("/api/conversations", {})
cid = conv["conversation_id"]
answer = sse("/api/chat", {"message": "员工手册里试用期是多久？",
                           "conversation_id": cid, "use_cache": False})
print(f"\n问答完成，回答 {len(answer)} 字")

after = rows()
print(f"调用后 metrics 行数: {len(after)}")
added = len(after) - len(before)
print(f"新增: {added} 行")
if added:
    print("  新增内容:", after[-1])
else:
    print("  ❌ 没有新增 —— 指标链路仍未接通")

req = urllib.request.Request(f"{BASE}/api/conversations/{cid}", method="DELETE")
urllib.request.urlopen(req, timeout=30).read()

print("\n结论:", "✅ 新后端已写入运行指标" if added else "❌ 仍未写入")
