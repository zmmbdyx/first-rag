"""后端服务配置（Pydantic Settings）。

设计原则
--------
* **不重复定义密钥**：大模型与向量库相关配置一律复用 ``rag/config.py`` 已有的
  环境变量契约（``API_KEY`` / ``BASE_URL`` / ``MODEL_OPTIONS`` / ``INDEX_DIR`` ...），
  这样 Streamlit、CLI、FastAPI 三条链路共用同一份 ``.env``，不会出现"三处配置三份值"。
* 这里只补充**前后端分离架构新增**的配置：CORS 白名单、会话数据库、上传目录等。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（backend/ 的上一级）。rag/ 核心包在根目录下，需要能被导入。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Settings(BaseSettings):
    """后端专属配置；敏感值全部来自环境变量 / .env，代码里不留默认密钥。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # 允许 .env 里存在 rag 核心包使用的其他变量
    )

    # ---------- 服务 ----------
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")
    reload: bool = Field(default=False, description="开发热重载")
    log_level: str = Field(default="info")

    # ---------- CORS ----------
    # 逗号分隔；默认放行 Vite 开发服务器。生产请收敛为真实域名。
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="允许跨域的前端地址，逗号分隔",
    )

    # ---------- 鉴权 ----------
    # 逗号分隔的多把密钥；**留空表示不鉴权**（本地开发零配置）。
    # 与旧版 api_server.py 的 API_KEYS 同名，迁移时无需改 .env。
    # 一旦配置，除 /api/health 外的所有 /api/* 都要求 Authorization: Bearer
    # 或 X-API-Key 请求头。
    api_keys: str = Field(default="", description="服务端密钥，逗号分隔；空=不鉴权")

    # ---------- 权限（ACL）----------
    # 用户组来源请求头。真实部署应由网关/SSO 注入（如 OIDC 的 groups claim），
    # 前端**不应**自行填写；保留可配置是为了适配不同网关的命名习惯。
    acl_groups_header: str = Field(
        default="X-User-Groups",
        description="承载用户组的请求头名，逗号分隔，如 'hr,finance'",
    )
    # 用户标识头（成本归因与审计用）。只取哈希入库，不留明文。
    acl_user_header: str = Field(default="X-User-Id", description="用户标识请求头名")

    # ---------- 管理接口密钥 ----------
    # 故意与 API_KEYS 分开：普通问答密钥不应具备删除文档、修改文档密级的能力。
    # 留空表示不校验（仅建议本地开发）。
    admin_keys: str = Field(default="", description="管理接口专用密钥；空=不限制")

    # ---------- 会话数据库 ----------
    # 默认 SQLite；生产可换成 Postgres，连接串形如
    #   postgresql+psycopg://<db_user>:<db_password>@<db_host>:5432/rag
    # 真实口令只放在 .env（已 gitignore），代码与模板里一律用占位符。
    database_url: str = Field(
        default="sqlite:///./backend/data/conversations.db",
        description="SQLAlchemy 连接串",
    )
    db_echo: bool = Field(default=False, description="打印 SQL，调试用")

    # ---------- 上传 ----------
    upload_dir: str = Field(default="backend/uploads", description="上传文件落盘目录")
    max_upload_mb: int = Field(default=50, description="单文件大小上限（MB）")
    # 上传后是否自动向量化入库。入库会重建 BM25 索引，大文件下较慢。
    ingest_on_upload: bool = Field(default=True)

    # ---------- 会话行为 ----------
    # 参与改写/生成的最近对话轮数（1 轮 = 一问一答）
    history_rounds: int = Field(default=3, ge=0, le=20)
    title_max_len: int = Field(default=20, description="自动标题取首条消息的前 N 字")
    default_top_k: int = Field(default=5, ge=1, le=20)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def admin_key_list(self) -> list[str]:
        return [k.strip() for k in self.admin_keys.split(",") if k.strip()]

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """进程级单例，避免每次请求重复解析 .env。"""
    return Settings()


settings = get_settings()
