"""
backend/database/connection.py
------------------------------
Async SQLAlchemy engine and session factory.

Two hardenings happen here:
  1. Every new connection runs `SET default_transaction_read_only = on`
     and `SET statement_timeout = <ms>` at the PostgreSQL session level.
     Even if the SQL validator is bypassed, the database will refuse any
     write or DDL operation.
  2. Sessions are async, pooled, and yielded as a FastAPI dependency.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_timeout=settings.database_pool_timeout,
    pool_pre_ping=True,
    echo=settings.debug,
)


@event.listens_for(_engine.sync_engine, "connect")
def _set_session_defaults(dbapi_connection, _connection_record):
    """
    Defense-in-depth: every physical connection is pinned to read-only
    and given a statement timeout. PostgreSQL will abort any write or
    long-running query at the engine level.
    """
    with dbapi_connection.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(f"SET statement_timeout = {settings.statement_timeout_ms}")
        cur.execute("SET idle_in_transaction_session_timeout = 30000")
    logger.debug("PostgreSQL session hardened: read-only + timeouts")


AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an async session, rolls back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Manual context manager for use outside the FastAPI request cycle."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def healthcheck() -> bool:
    """Used by /health endpoint to verify DB reachability."""
    try:
        async with session_scope() as s:
            await s.execute(text("SELECT 1"))
        return True
    except Exception as e:  # pragma: no cover
        logger.error("DB healthcheck failed: %s", e)
        return False


async def dispose_engine() -> None:
    await _engine.dispose()
