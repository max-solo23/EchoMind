"""
Database session management for async SQLAlchemy.

This module provides:
- Async engine creation
- Session factory for dependency injection
- Context manager for session lifecycle

Usage:
    async with get_session() as session:
        repo = SQLAlchemyConversationRepository(session)
        await repo.create_session(...)
"""

from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from config import Config


# Global engine and session factory (initialized on first use)
_engine = None
_async_session_factory = None


def get_engine(config: Config):
    """
    Get or create the async SQLAlchemy engine.

    Uses singleton pattern - engine is created once and reused.
    """
    global _engine

    if _engine is None:
        if not config.database_url:
            raise RuntimeError(
                "Database URL not configured. "
                "Set POSTGRES_URL or DATABASE_URL environment variable."
            )

        _engine = create_async_engine(
            config.database_url,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            echo=config.db_echo,
            pool_pre_ping=True,  # Verify connections before use
        )

    return _engine


def get_session_factory(config: Config) -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session factory.

    Returns:
        Async session factory for creating database sessions
    """
    global _async_session_factory

    if _async_session_factory is None:
        engine = get_engine(config)
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _async_session_factory


@asynccontextmanager
async def get_session(config: Config) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.

    Usage:
        async with get_session(config) as session:
            # Use session
            pass
        # Session is automatically closed

    Yields:
        AsyncSession: Database session
    """
    factory = get_session_factory(config)
    session = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def close_database():
    """
    Close database connections.

    Call this on application shutdown.
    """
    global _engine, _async_session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
