"""清理开发期产生的测试会话（保留数据库文件本身）。

    python scripts/clear_dev_conversations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import init_db, session_scope  # noqa: E402
from backend.services import conversation_service as cs  # noqa: E402

if __name__ == "__main__":
    init_db()
    with session_scope() as db:
        n = cs.clear_all(db)
    print(f"[完成] 已删除 {n} 个测试会话")
