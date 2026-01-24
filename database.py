from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import Config


_engine = None
_async_session_factory = None


def get_engine(config: Config):
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
    global _engine, _async_session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
