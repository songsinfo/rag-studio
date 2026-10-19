import threading
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, Connection
from app.config import settings

_sqlite_connection_lock = threading.RLock()

engine = create_engine(
    settings.sqlalchemy_database_uri,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args={"charset": "utf8mb4"}
)


@contextmanager
def get_db_connection() -> Generator[Connection, None, None]:
    # 测试夹具使用 SQLite StaticPool，因此 LangGraph checkpoint 线程与应用代码会共享同一个 DB-API 连接。
    # SQLite 的同一连接不适合并发访问，所以这里只对测试环境中的共享连接做串行化保护；
    # 生产 MySQL 仍保持正常并发，避免为了兼容测试而降低真实环境吞吐。
    lock = _sqlite_connection_lock if engine.dialect.name == "sqlite" else None
    if lock is None:
        with engine.connect() as connection:
            yield connection
        return
    with lock:
        with engine.connect() as connection:
            yield connection


def get_db() -> Generator[Connection, None, None]:
    lock = _sqlite_connection_lock if engine.dialect.name == "sqlite" else None
    if lock is None:
        with engine.connect() as connection:
            yield connection
        return
    with lock:
        with engine.connect() as connection:
            yield connection
