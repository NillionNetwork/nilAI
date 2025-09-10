from contextlib import asynccontextmanager
import logging
import functools
import sqlalchemy

from typing import Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import AsyncAdaptedQueuePool, Column as _Column
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from nilai_api.config import CONFIG

_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None
_SessionLocal: Optional[sessionmaker] = None

# Create base and engine with improved configuration
Base = sqlalchemy.orm.declarative_base()

logger = logging.getLogger(__name__)


@functools.wraps(_Column)  # type: ignore[reportUnknownVariableType]
def Column(*args: Any, **kwargs: Any):  # ruff: disable=invalid-name
    return _Column(*args, **kwargs)


@dataclass
class DatabaseConfig:
    database_url: sqlalchemy.engine.url.URL
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: timedelta = timedelta(seconds=30)
    pool_recycle: timedelta = timedelta(hours=1)

    @staticmethod
    def from_env() -> "DatabaseConfig":
        database_url = sqlalchemy.engine.url.URL.create(
            drivername="postgresql+asyncpg",  # Use asyncpg driver
            username=CONFIG.database.user,
            password=CONFIG.database.password,
            host=CONFIG.database.host,
            port=CONFIG.database.port,
            database=CONFIG.database.db,
        )
        return DatabaseConfig(database_url)


def get_engine() -> sqlalchemy.ext.asyncio.AsyncEngine:
    global _engine
    if _engine is None:
        config = DatabaseConfig.from_env()
        _engine = create_async_engine(
            config.database_url,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout.total_seconds(),
            pool_recycle=config.pool_recycle.total_seconds(),
            echo=False,  # Set to True for SQL logging during development
        )
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


# Async context manager for database sessions
@asynccontextmanager
async def get_db_session() -> "AsyncGenerator[AsyncSession, Any]":
    """Provide a transactional scope for database operations."""
    session = get_sessionmaker()()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        await session.close()
