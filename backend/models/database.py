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
    """建表（幂等）。在应用 lifespan 启动时调用一次。"""
    from backend.models import conversation, message  # noqa: F401  触发模型注册

    Base.metadata.create_all(bind=engine)


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
