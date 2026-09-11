"""SQLAlchemy 引擎 / 会话工厂 / Base。

数据库地址完全来自环境变量（``DATABASE_URL``），默认 SQLite。
默认走 SQLite 时自动开启 ``check_same_thread=False`` 与外键约束。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def _build_engine():
    url = settings.database_url
    kwargs: dict = {"echo": settings.db_echo, "future": True}
    if settings.is_sqlite:
        # SQLite 需要允许跨线程使用（FastAPI 线程池里跑同步 ORM）
        kwargs["connect_args"] = {"check_same_thread": False}
        # 确保 sqlite 文件所在目录存在
        raw = url.split("sqlite:///", 1)[-1]
        if raw and raw != ":memory:":
            Path(raw).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    else:
        # 连接池：请求级会话，预检失效连接
        kwargs["pool_pre_ping"] = True
        kwargs["pool_recycle"] = 1800
    return create_engine(url, **kwargs)


engine = _build_engine()

if settings.is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - 驱动级钩子
        """外键约束 + WAL（并发读写下更稳）。"""
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """建表 + 轻量迁移（幂等）。在应用 lifespan 启动时调用一次。"""
    from backend.models import conversation, message  # noqa: F401  触发模型注册

    Base.metadata.create_all(bind=engine)
    _run_light_migrations()


# 轻量迁移：为**已存在的表**补上后加的列。
#
# 为什么需要：``create_all`` 只创建缺失的表，**不会**给已有表加列。
# 前后端分离后新增了 conversations.owner_id（会话归属，用于按用户隔离），
# 老部署升级上来时该列不存在，任何插入会话的请求都会 500 —— 实测确实如此。
# 这里用"查列 -> 缺则 ADD COLUMN"补齐，避免为此引入 Alembic 的迁移链
# （本项目仍是单文件 SQLite 起步，收益不抵成本）。
# 生产使用 PostgreSQL 时建议改用 Alembic 管理迁移。
_LIGHT_MIGRATIONS: list[tuple[str, str, str]] = [
    # (表名, 列名, 列定义)
    ("conversations", "owner_id", "VARCHAR(128) NOT NULL DEFAULT ''"),
]


def _run_light_migrations() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl in _LIGHT_MIGRATIONS:
            if table not in existing_tables:
                continue
            cols = {c["name"] for c in inspector.get_columns(table)}
            if column in cols:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
            print(f"[backend] 已迁移：{table}.{column} 补齐完成")
        # 给新列补索引（SQLite/PG 都支持 IF NOT EXISTS）
        for table, column, _ in _LIGHT_MIGRATIONS:
            if table not in existing_tables:
                continue
            try:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS ix_{table}_{column} ON {table} ({column})"))
            except Exception:  # noqa: BLE001 — 索引失败不影响功能
                pass


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每请求一个会话，结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """在非请求上下文（后台线程、脚本）里安全使用会话。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
