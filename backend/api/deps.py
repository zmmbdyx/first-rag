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
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request, status

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


# ---------------------------------------------------------------- 用户上下文
@dataclass
class UserContext:
    """调用方身份：用于**文档级权限过滤**、成本归因与审计。

    ``groups`` 决定能检索到哪些文档（见 rag.retriever.is_visible）。
    真实部署应由网关/SSO 注入该头部；应用层只做读取，不做信任提升。
    """

    groups: list[str] = field(default_factory=list)
    user_id: str = ""
    # 供日志与指标使用（不存明文）
    user_hash: str = ""

    @property
    def is_admin(self) -> bool:
        from rag.config import ACL_ADMIN_TAG

        return bool(ACL_ADMIN_TAG) and ACL_ADMIN_TAG in self.groups


def get_user_context(request: Request) -> UserContext:
    """从请求头解析用户上下文。

    头部名可配置（``ACL_GROUPS_HEADER`` / ``ACL_USER_HEADER``），默认
    ``X-User-Groups`` 与 ``X-User-Id``。

    ⚠️ **安全前提**：这些头部必须由可信网关注入并**覆盖**客户端传入的同名头。
    若服务直接暴露在公网且未做覆盖，客户端可以伪造成任意组。生产部署请：
    ① 用 API 网关剥掉外部同名头再注入；或 ② 关闭该机制，改由网关注入
    ``Authorization`` 并把组信息编码进签名的 JWT 中由本服务校验。
    """
    raw_groups = request.headers.get(settings.acl_groups_header, "")
    groups = [g.strip() for g in raw_groups.split(",") if g.strip()]
    user_id = (request.headers.get(settings.acl_user_header, "") or "").strip()
    return UserContext(
        groups=groups,
        user_id=user_id,
        user_hash=hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16] if user_id else "",
    )


def require_admin_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """管理接口（删除文档、修改文档密级）专用鉴权。

    与 ``require_api_key`` 分开是刻意的：**问答密钥不应具备管理权限**。
    未配置 ``ADMIN_KEYS`` 时不校验（本地开发）；配置后必须匹配其一。
    """
    allowed = {_digest(k) for k in settings.admin_key_list}
    if not allowed:
        return

    token = ""
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        token = authorization[len(_BEARER_PREFIX):].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token or not any(secrets.compare_digest(_digest(token), d) for d in allowed):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员密钥（ADMIN_KEYS）才能执行该操作",
        )
