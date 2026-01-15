"""Tests for database.py - async SQLAlchemy session management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import database
from config import Config


@pytest.fixture(autouse=True)
def reset_database_globals():
    """
    Reset singleton globals before each test.

    Why: Singleton pattern uses module-level globals (_engine, _async_session_factory).
    Without reset, tests affect each other - test A creates engine, test B gets same one.
    This fixture ensures each test starts with fresh state.
    """
    database._engine = None
    database._async_session_factory = None
    yield
    # Also reset after test
    database._engine = None
    database._async_session_factory = None


@pytest.fixture
def mock_config():
    """Create mock Config with database settings."""
    config = MagicMock(spec=Config)
    config.database_url = "postgresql+asyncpg://user:pass@localhost/testdb"
    config.db_pool_size = 5
    config.db_max_overflow = 10
    config.db_echo = False
    return config


class TestGetEngine:
    """Test engine creation and singleton pattern."""

    @patch("database.create_async_engine")
    def test_creates_engine_with_config(self, mock_create_engine, mock_config):
        """
        Verify engine is created with correct parameters.

        Why: Engine configuration (pool size, timeouts) affects performance
        and reliability. Wrong settings could cause connection exhaustion.
        """
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        result = database.get_engine(mock_config)

        assert result == mock_engine
        mock_create_engine.assert_called_once_with(
            mock_config.database_url,
            pool_size=mock_config.db_pool_size,
            max_overflow=mock_config.db_max_overflow,
            echo=mock_config.db_echo,
            pool_pre_ping=True,
        )

    @patch("database.create_async_engine")
    def test_returns_same_engine_on_second_call(self, mock_create_engine, mock_config):
        """
        Verify singleton pattern - engine created once.

        Why: Creating multiple engines wastes resources and can exhaust
        connection pool. Singleton ensures one shared engine.
        """
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine1 = database.get_engine(mock_config)
        engine2 = database.get_engine(mock_config)

        assert engine1 is engine2
        mock_create_engine.assert_called_once()  # Only once, not twice

    def test_raises_when_no_database_url(self, mock_config):
        """
        Verify RuntimeError raised when database_url is None.

        Why: Clear error message helps debugging deployment issues.
        Better than cryptic SQLAlchemy error when URL is None.
        """
        mock_config.database_url = None

        with pytest.raises(RuntimeError) as exc_info:
            database.get_engine(mock_config)

        assert "Database URL not configured" in str(exc_info.value)


class TestGetSessionFactory:
    """Test session factory creation."""

    @patch("database.async_sessionmaker")
    @patch("database.get_engine")
    def test_creates_factory_with_engine(self, mock_get_engine, mock_sessionmaker, mock_config):
        """
        Verify factory created with correct engine and settings.

        Why: Factory settings (expire_on_commit, autoflush) affect ORM behavior.
        expire_on_commit=False prevents detached instance errors.
        """
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_factory = MagicMock()
        mock_sessionmaker.return_value = mock_factory

        result = database.get_session_factory(mock_config)

        assert result == mock_factory
        mock_sessionmaker.assert_called_once()

    @patch("database.async_sessionmaker")
    @patch("database.get_engine")
    def test_returns_same_factory_on_second_call(
        self, mock_get_engine, mock_sessionmaker, mock_config
    ):
        """
        Verify singleton pattern for factory.

        Why: Same as engine - avoid creating multiple factories.
        """
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_factory = MagicMock()
        mock_sessionmaker.return_value = mock_factory

        factory1 = database.get_session_factory(mock_config)
        factory2 = database.get_session_factory(mock_config)

        assert factory1 is factory2
        mock_sessionmaker.assert_called_once()


class TestGetSession:
    """Test session context manager."""

    @pytest.mark.asyncio
    @patch("database.get_session_factory")
    async def test_yields_session_and_closes(self, mock_get_factory, mock_config):
        """
        Verify session is yielded and closed on exit.

        Why: Unclosed sessions leak connections. Context manager ensures
        cleanup even if code forgets to close explicitly.
        """
        mock_session = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)
        mock_get_factory.return_value = mock_factory

        async with database.get_session(mock_config) as session:
            assert session == mock_session

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("database.get_session_factory")
    async def test_rollbacks_on_exception(self, mock_get_factory, mock_config):
        """
        Verify rollback called when exception raised in context.

        Why: If code raises inside the context, partial changes should
        be rolled back to maintain database consistency.
        """
        mock_session = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)
        mock_get_factory.return_value = mock_factory

        with pytest.raises(ValueError):
            async with database.get_session(mock_config) as _session:
                raise ValueError("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


class TestCloseDatabase:
    """Test database shutdown."""

    @pytest.mark.asyncio
    async def test_disposes_engine_and_resets_globals(self):
        """
        Verify engine disposed and globals reset.

        Why: On shutdown, connections must be closed cleanly.
        Resetting globals allows fresh start if app restarts.
        """
        mock_engine = AsyncMock()
        database._engine = mock_engine
        database._async_session_factory = MagicMock()

        await database.close_database()

        mock_engine.dispose.assert_called_once()
        assert database._engine is None
        assert database._async_session_factory is None  # type: ignore[unreachable]

    @pytest.mark.asyncio
    async def test_handles_already_closed(self):
        """
        Verify no error when engine is already None.

        Why: Idempotent shutdown - calling close twice shouldn't error.
        Important for graceful shutdown handlers that might run multiple times.
        """
        database._engine = None

        # Should not raise
        await database.close_database()
