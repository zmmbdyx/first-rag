"""验证可选鉴权：未配置 API_KEYS 时放行，配置后强制生效。

用 TestClient 直接打应用，不依赖真实大模型：
鉴权是**请求进入路由之前**就完成的，所以 401 的用例不需要模型可用。

    python scripts/check_auth.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def build_app(api_keys: str):
    """以指定 API_KEYS 重新构建应用（settings 是 lru_cache 单例，必须重建）。"""
    os.environ["API_KEYS"] = api_keys
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            del sys.modules[name]
    from backend.main import app

    return app


def main() -> int:
    from fastapi.testclient import TestClient

    print("=== 1) 未配置 API_KEYS：应当放行 ===")
    app = build_app("")
    with TestClient(app) as client:
        r = client.get("/api/health")
        check("health 可访问", r.status_code == 200, f"HTTP {r.status_code}")
        check("health 标记 auth_required=false", r.json().get("auth_required") is False)

        r = client.get("/api/conversations")
        check("会话列表无需密钥", r.status_code == 200, f"HTTP {r.status_code}")

        r = client.post("/api/conversations", json={})
        check("创建会话无需密钥", r.status_code == 201, f"HTTP {r.status_code}")

    print("\n=== 2) 配置 API_KEYS：应当强制鉴权 ===")
    # 故意用一个明显是占位符的串：它匹配 audit_secrets.py 的 ALLOW 白名单，
    # 不会让"提交前脱敏审计"误报（审计脚本无法区分"测试用假密钥"与"真密钥"）。
    SECRET = "sk-xxx-auth-test-placeholder"
    app = build_app(SECRET)
    with TestClient(app) as client:
        r = client.get("/api/health")
        check("health 仍然豁免（健康探针需要）", r.status_code == 200, f"HTTP {r.status_code}")
        check("health 标记 auth_required=true", r.json().get("auth_required") is True)

        for path, method in (
            ("/api/conversations", "GET"),
            ("/api/upload/supported", "GET"),
        ):
            r = client.request(method, path)
            check(f"{method} {path} 无密钥 → 401", r.status_code == 401, f"HTTP {r.status_code}")

        r = client.post("/api/chat", json={"message": "hi", "conversation_id": "x" * 32})
        check("POST /api/chat 无密钥 → 401", r.status_code == 401, f"HTTP {r.status_code}")

        r = client.post("/api/conversations", json={})
        check("POST /api/conversations 无密钥 → 401", r.status_code == 401, f"HTTP {r.status_code}")

        # 错误密钥
        r = client.get("/api/conversations", headers={"X-API-Key": "wrong-key"})
        check("错误密钥 → 401", r.status_code == 401, f"HTTP {r.status_code}")

        r = client.get(
            "/api/conversations", headers={"Authorization": "Bearer wrong-key"}
        )
        check("错误 Bearer → 401", r.status_code == 401, f"HTTP {r.status_code}")

        # 正确密钥：两种头都要能用
        r = client.get("/api/conversations", headers={"X-API-Key": SECRET})
        check("X-API-Key 正确 → 200", r.status_code == 200, f"HTTP {r.status_code}")

        r = client.get("/api/conversations", headers={"Authorization": f"Bearer {SECRET}"})
        check("Authorization Bearer 正确 → 200", r.status_code == 200, f"HTTP {r.status_code}")

        # 大小写不敏感的 scheme
        r = client.get("/api/conversations", headers={"Authorization": f"bearer {SECRET}"})
        check("scheme 大小写不敏感 → 200", r.status_code == 200, f"HTTP {r.status_code}")

        # 401 响应不应回显密钥
        r = client.get("/api/conversations", headers={"X-API-Key": "wrong-key"})
        check("401 响应体不含提交的密钥", "wrong-key" not in r.text, r.text[:80])

    print("\n=== 3) 多密钥逗号分隔 ===")
    app = build_app(f"{SECRET},sk-yyy-second-placeholder")
    with TestClient(app) as client:
        for k in (SECRET, "sk-yyy-second-placeholder"):
            r = client.get("/api/conversations", headers={"X-API-Key": k})
            check(f"密钥 {k[:12]}… 可用", r.status_code == 200, f"HTTP {r.status_code}")

    print()
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 鉴权行为符合预期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
