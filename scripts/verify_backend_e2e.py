"""端到端验证后端 API：会话 CRUD + 上传 + SSE 流式问答。

    python scripts/verify_backend_e2e.py [base_url]

只依赖标准库 urllib，避免额外装 requests。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
ROOT = Path(__file__).resolve().parent.parent


def req(method: str, path: str, payload: dict | None = None, timeout: int = 180):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data:
        r.add_header("Content-Type", "application/json; charset=utf-8")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def upload(path: Path) -> dict:
    """multipart/form-data 手工拼装（标准库没有现成封装）。"""
    boundary = "----ragboundary7f3a"
    content = path.read_bytes()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode("utf-8"),
        b"Content-Type: application/octet-stream\r\n\r\n",
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    r = urllib.request.Request(f"{BASE}/api/upload", data=body, method="POST")
    r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sse(path: str, payload: dict, timeout: int = 240) -> list[tuple[str, object]]:
    """读取 SSE 流，返回 [(event, parsed_data)]。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    r = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    r.add_header("Content-Type", "application/json; charset=utf-8")
    r.add_header("Accept", "text/event-stream")

    events: list[tuple[str, object]] = []
    event = "message"
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        assert resp.headers.get("content-type", "").startswith("text/event-stream"), \
            f"content-type 不是 event-stream: {resp.headers.get('content-type')}"
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    events.append((event, json.loads(line[6:])))
                except json.JSONDecodeError as e:
                    events.append((event, {"_parse_error": str(e), "_raw": line}))
            elif not line:
                event = "message"
    return events


def main() -> int:
    failed: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failed.append(label)

    print("=== 1) 健康检查 ===")
    h = req("GET", "/api/health")
    check("status", h.get("status") == "ok", f"chunks={h.get('chunks')} models={len(h.get('models', []))}")

    print("\n=== 2) 会话 CRUD ===")
    conv = req("POST", "/api/conversations", {})
    cid = conv["conversation_id"]
    check("创建会话", bool(cid), f"id={cid}")
    check("默认标题为中文「新对话」", conv["title"] == "新对话", repr(conv["title"]))

    renamed = req("PUT", f"/api/conversations/{cid}/title", {"title": "端到端测试会话"})
    check("重命名回写中文正确", renamed["title"] == "端到端测试会话", repr(renamed["title"]))

    listed = req("GET", "/api/conversations")
    check("列表包含新会话", any(c["conversation_id"] == cid for c in listed), f"{len(listed)} 条")
    check("列表含 message_count 字段", "message_count" in listed[0])

    print("\n=== 3) 文档上传并入库 ===")
    sample = ROOT / "data" / "samples" / "IT服务常见问题FAQ.txt"
    if sample.exists():
        up = upload(sample)
        check("上传返回契约字段",
              all(k in up for k in ("file_id", "filename", "status")), json.dumps(up, ensure_ascii=False))
        check("入库成功", up["status"] in ("done", "processing"), up.get("message", ""))
    else:
        check("示例文件存在", False, str(sample))

    print("\n=== 4) SSE 流式问答 ===")
    events = sse("/api/chat", {"message": "IT服务台的支持时间是什么？", "conversation_id": cid})
    names = [n for n, _ in events]
    check("收到 conversation 事件", "conversation" in names)
    check("收到 sources 事件", "sources" in names)
    check("收到 content 事件", names.count("content") > 0, f"content 帧数={names.count('content')}")
    check("收到 done 事件", "done" in names)
    check("无 error 事件", "error" not in names,
          next((str(d) for n, d in events if n == "error"), ""))

    content = "".join(d.get("delta", "") for n, d in events if n == "content")
    print(f"\n  ── 回答（{len(content)} 字）──")
    print("  " + content[:400].replace("\n", "\n  "))
    check("回答非空", len(content) > 0)

    src = next((d for n, d in events if n == "sources"), [])
    print(f"\n  ── 引用来源 {len(src)} 条 ──")
    for s in src[:3]:
        print(f"  [{s.get('index')}] {s.get('doc_name')} · {s.get('section_path') or '-'} "
              f"· p{s.get('page')} · score={s.get('score')} · sim={s.get('similarity')}")
    check("来源含 doc_name", bool(src) and all(s.get("doc_name") for s in src))
    check("来源含相似度", bool(src) and any(s.get("similarity") is not None for s in src))

    done = next((d for n, d in events if n == "done"), {})
    print(f"\n  ── done ──\n  {json.dumps({k: v for k, v in done.items() if k != 'sources'}, ensure_ascii=False)}")
    check("done 含耗时", float(done.get("latency", 0)) > 0, f"latency={done.get('latency')}")
    check("done 含 message_id", done.get("message_id") is not None)

    print("\n=== 5) 消息持久化与历史还原 ===")
    msgs = req("GET", f"/api/conversations/{cid}/messages")
    check("消息落库 2 条", len(msgs) == 2, f"{len(msgs)} 条")
    if len(msgs) >= 2:
        check("顺序为 user → assistant", msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant")
        check("assistant 保留了引用来源", len(msgs[1].get("sources") or []) == len(src),
              f"{len(msgs[1].get('sources') or [])} 条")
        check("assistant 保留了耗时", float(msgs[1].get("latency", 0)) > 0)
        check("历史正文与流式内容一致", msgs[1]["content"] == content,
              f"库内 {len(msgs[1]['content'])} 字 vs 流式 {len(content)} 字")

    print("\n=== 6) 首条消息自动生成标题 ===")
    conv2 = req("POST", "/api/conversations", {})
    cid2 = conv2["conversation_id"]
    events2 = sse("/api/chat", {"message": "咖啡机保修期是几年？", "conversation_id": cid2})
    done2 = next((d for n, d in events2 if n == "done"), {})
    listed2 = req("GET", "/api/conversations")
    row2 = next((c for c in listed2 if c["conversation_id"] == cid2), {})
    check("标题由首条消息生成", row2.get("title", "").startswith("咖啡机保修期"),
          repr(row2.get("title")))
    check("message_count 已更新", row2.get("message_count") == 2, str(row2.get("message_count")))
    check("done 回传标题", bool(done2.get("title")), repr(done2.get("title")))

    print("\n=== 7) 删除会话级联 ===")
    d1 = req("DELETE", f"/api/conversations/{cid}")
    check("删除返回 ok", d1.get("ok") is True)
    gone = False
    try:
        req("GET", f"/api/conversations/{cid}/messages")
    except urllib.error.HTTPError as e:
        gone = e.code == 404
    check("删除后取消息返回 404", gone)
    req("DELETE", f"/api/conversations/{cid2}")

    print()
    if failed:
        print(f"[失败] {len(failed)} 项未通过：")
        for f in failed:
            print("   -", f)
        return 1
    print("[通过] 后端端到端验证全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
