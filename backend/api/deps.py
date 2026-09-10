"""可选的服务鉴权。

设计取舍
--------
* **默认不鉴权**：``API_KEYS`` 为空时直接放行，本地开发与内网演示零配置。
  这一点与旧版 ``api_server.py`` 的行为保持一致，不制造迁移摩擦。
* **一旦配置就强制生效**：设置了 ``API_KEYS`` 后，除健康检查外的所有
  ``/api/*`` 端点都要求携带密钥，避免"配了 key 以为有保护、实际没挂上"的
  假安全感。
* **只比对摘要，且用常量时间比较**：日志/异常里不会出现明文密钥，
  ``secrets.compare_digest`` 避免按字符提前返回带来的时序侧信道。
* **健康检查故意豁免**：容器 HEALTHCHECK、负载均衡探针与前端"知识库是否
  就绪"都要能无凭据访问；它不返回任何业务数据，只暴露切片数与模型名。
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Header, HTTPException, status

from backend.config import settings

# 允许的请求头：Authorization: Bearer <key>  或  X-API-Key: <key>
_BEARER_PREFIX = "bearer "


def _digest(value: str) -> str:
    """把密钥转成摘要用于比较，避免明文出现在任何日志或报错里。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configured_digests() -> set[str]:
    return {_digest(k) for k in settings.api_key_list}


def auth_enabled() -> bool:
    """是否启用了鉴权（供启动日志与 /api/health 展示）。"""
    return bool(settings.api_key_list)


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """FastAPI 依赖：校验 Bearer / X-API-Key。

    未配置 ``API_KEYS`` 时直接放行（本地开发）；配置了则必须匹配其中之一。
    """
    allowed = _configured_digests()
    if not allowed:
        return

    token = ""
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        token = authorization[len(_BEARER_PREFIX):].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key：请在 Authorization: Bearer <key> 或 X-API-Key 请求头中提供",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 逐个常量时间比较，任一匹配即通过
    if not any(secrets.compare_digest(_digest(token), d) for d in allowed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
