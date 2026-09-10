"""后端自检：配置 / 建表 / ORM / 路由 是否就绪（不依赖大模型，可离线跑）。

    python scripts/check_backend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        print(f"  [OK]   {label}" + (f" — {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        FAILED.append(f"{label}: {e}")
        print(f"  [FAIL] {label} — {type(e).__name__}: {e}")


def main() -> int:
    print("=== 1) 配置 ===")

    def _settings():
        from backend.config import settings

        return (f"db={settings.database_url} origins={settings.cors_origin_list} "
                f"upload={settings.upload_path}")

    check("加载 Settings", _settings)

    print("=== 2) 建表与 ORM ===")

    def _init():
        from backend.models import init_db

        init_db()
        return "create_all 完成"

    check("初始化数据库", _init)

    def _crud():
        from backend.models import session_scope
        from backend.services import conversation_service as cs

        with session_scope() as db:
            conv = cs.create_conversation(db, title="自检会话")
            cs.add_message(db, conv.id, "user", "这是一条自检消息，用于确认标题自动生成逻辑。")
            cs.add_message(db, conv.id, "assistant", "收到。", latency=0.5)
            db.refresh(conv)
            listed = cs.list_conversations(db, limit=5)
            msgs = cs.list_messages(db, conv.id)
            hist = cs.recent_history(db, conv.id)
            title = conv.title
            count = conv.message_count
            cs.delete_conversation(db, conv.id)
            assert not cs.get_conversation(db, conv.id), "删除后仍能查到会话"
            assert len(msgs) == 2, f"消息数应为 2，实际 {len(msgs)}"
            assert len(hist) == 2, f"历史轮数应为 2，实际 {len(hist)}"
            assert count == 2, f"message_count 应为 2，实际 {count}"
        return f"标题={title!r} 消息={len(msgs)} 列表={len(listed)}"

    check("会话/消息 CRUD 与级联删除", _crud)

    print("=== 3) FastAPI 应用与路由 ===")

    def _routes():
        from backend.main import app

        # 用 OpenAPI schema 校验而不是遍历 app.routes：
        # FastAPI 新版本会把 include_router 的结果包成内部对象，直接遍历拿不到 path。
        schema = app.openapi()
        paths = set(schema.get("paths", {}))
        expected = {
            "/api/chat",
            "/api/conversations",
            "/api/conversations/{conversation_id}/messages",
            "/api/conversations/{conversation_id}/title",
            "/api/conversations/{conversation_id}",
            "/api/upload",
            "/api/health",
        }
        missing = sorted(expected - paths)
        assert not missing, f"缺少路由 {missing}"

        methods = {
            ("/api/chat", "post"),
            ("/api/conversations", "post"),
            ("/api/conversations", "get"),
            ("/api/conversations/{conversation_id}/messages", "get"),
            ("/api/conversations/{conversation_id}", "delete"),
            ("/api/conversations/{conversation_id}/title", "put"),
            ("/api/upload", "post"),
        }
        bad = [f"{p} {m}" for p, m in sorted(methods) if m not in schema["paths"].get(p, {})]
        assert not bad, f"方法缺失 {bad}"
        return f"{len(paths)} 条路径，契约端点与方法齐全"

    check("路由注册（OpenAPI 契约）", _routes)

    print("=== 4) RAG 核心可导入 ===")

    def _core():
        import rag.pipeline  # noqa: F401
        from backend.core import rag_chain, streaming, vectorstore  # noqa: F401

        return "rag.pipeline + backend.core 导入成功"

    check("核心包导入", _core)

    def _sse_format():
        from backend.api.chat import _sse

        frame = _sse("content", {"delta": "你好"})
        assert frame.startswith("event: content\ndata: "), frame
        assert frame.endswith("\n\n"), frame
        assert "\n" not in frame.split("data: ", 1)[1].rstrip("\n"), "data 必须是单行 JSON"
        return repr(frame)

    check("SSE 帧格式", _sse_format)

    print()
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 后端自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
