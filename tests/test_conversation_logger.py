from unittest.mock import AsyncMock

import pytest

from services.conversation_logger import ConversationLogger


@pytest.fixture
def mock_conversation_repo():
    return AsyncMock()


@pytest.fixture
def mock_cache_service():
    return AsyncMock()


@pytest.fixture
def logger(mock_conversation_repo, mock_cache_service):
    return ConversationLogger(
        conversation_repo=mock_conversation_repo,
        cache_service=mock_cache_service,
        enable_caching=True,
    )


class TestConversationLoggerInit:
    def test_stores_dependencies(self, mock_conversation_repo, mock_cache_service):
        logger = ConversationLogger(
            conversation_repo=mock_conversation_repo, cache_service=mock_cache_service
        )

        assert logger.conversation_repo == mock_conversation_repo
        assert logger.cache_service == mock_cache_service
        assert logger.enable_caching is True

    def test_caching_can_be_disabled(self, mock_conversation_repo, mock_cache_service):
        logger = ConversationLogger(
            conversation_repo=mock_conversation_repo,
            cache_service=mock_cache_service,
            enable_caching=False,
        )

        assert logger.enable_caching is False


class TestGetOrCreateSession:
    @pytest.mark.asyncio
    async def test_delegates_to_repo(self, logger, mock_conversation_repo):
        mock_conversation_repo.create_session.return_value = 42

        result = await logger.get_or_create_session("sess_123", "192.168.1.1")

        assert result == 42
        mock_conversation_repo.create_session.assert_called_once_with("sess_123", "192.168.1.1")


class TestCheckCache:
    @pytest.mark.asyncio
    async def test_returns_cached_answer(self, logger, mock_cache_service):
        mock_cache_service.get_cached_answer.return_value = "Cached response"

        result = await logger.check_cache("What is Python?", None, False)

        assert result == "Cached response"

    @pytest.mark.asyncio
    async def test_returns_none_when_caching_disabled(
        self, mock_conversation_repo, mock_cache_service
    ):
        logger = ConversationLogger(
            conversation_repo=mock_conversation_repo,
            cache_service=mock_cache_service,
            enable_caching=False,
        )

        result = await logger.check_cache("Question", None, False)

        assert result is None
        mock_cache_service.get_cached_answer.assert_not_called()


class TestLogAndCache:
    @pytest.mark.asyncio
    async def test_logs_conversation(self, logger, mock_conversation_repo):
        mock_conversation_repo.log_conversation.return_value = 99

        result = await logger.log_and_cache(
            session_db_id=1, user_message="Hello", bot_response="Hi there"
        )

        assert result == 99
        mock_conversation_repo.log_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_caches_response_when_enabled(
        self, logger, mock_cache_service, mock_conversation_repo
    ):
        mock_conversation_repo.log_conversation.return_value = 1

        await logger.log_and_cache(
            session_db_id=1,
            user_message="What is Python?",
            bot_response="A language",
            cache_response=True,
        )

        mock_cache_service.cache_answer.assert_called_once()


class TestDelegationMethods:
    @pytest.mark.asyncio
    async def test_get_session_history(self, logger, mock_conversation_repo):
        mock_conversation_repo.get_session_by_id.return_value = {"id": 1}

        result = await logger.get_session_history("sess_123")

        assert result == {"id": 1}

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, logger, mock_cache_service):
        mock_cache_service.get_cache_stats.return_value = {"total": 10}

        result = await logger.get_cache_stats()

        assert result == {"total": 10}

    @pytest.mark.asyncio
    async def test_clear_cache(self, logger, mock_cache_service):
        mock_cache_service.clear_cache.return_value = 5

        result = await logger.clear_cache()

        assert result == 5

    @pytest.mark.asyncio
    async def test_cleanup_expired_cache(self, logger, mock_cache_service):
        mock_cache_service.cleanup_expired.return_value = 3

        result = await logger.cleanup_expired_cache()

        assert result == 3


class TestAdminMethods:
    @pytest.mark.asyncio
    async def test_list_sessions(self, logger, mock_conversation_repo):
        mock_conversation_repo.list_sessions.return_value = {"sessions": [], "total": 0}

        result = await logger.list_sessions(page=1, limit=10)

        assert result == {"sessions": [], "total": 0}
        mock_conversation_repo.list_sessions.assert_called_once_with(1, 10, "created_at", "desc")

    @pytest.mark.asyncio
    async def test_list_cache_entries(self, logger, mock_cache_service):
        mock_cache_service.list_cache_entries.return_value = {"entries": []}

        result = await logger.list_cache_entries(page=2, limit=15)

        assert result == {"entries": []}

    @pytest.mark.asyncio
    async def test_get_cache_entry(self, logger, mock_cache_service):
        mock_cache_service.get_cache_by_id.return_value = {"id": 1, "question": "Test"}

        result = await logger.get_cache_entry(1)

        assert result == {"id": 1, "question": "Test"}

    @pytest.mark.asyncio
    async def test_delete_cache_entry(self, logger, mock_cache_service):
        mock_cache_service.delete_cache_by_id.return_value = True

        result = await logger.delete_cache_entry(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_update_cache_entry(self, logger, mock_cache_service):
        mock_cache_service.update_cache_variations.return_value = True

        result = await logger.update_cache_entry(1, ["var1", "var2"])

        assert result is True
        mock_cache_service.update_cache_variations.assert_called_once_with(1, ["var1", "var2"])

    @pytest.mark.asyncio
    async def test_search_cache(self, logger, mock_cache_service):
        mock_cache_service.search_cache.return_value = [{"question": "Python"}]

        result = await logger.search_cache("python", limit=10)

        assert result == [{"question": "Python"}]

    @pytest.mark.asyncio
    async def test_delete_session(self, logger, mock_conversation_repo):
        mock_conversation_repo.delete_session.return_value = True

        result = await logger.delete_session("sess_123")

        assert result is True

    @pytest.mark.asyncio
    async def test_clear_all_sessions(self, logger, mock_conversation_repo):
        mock_conversation_repo.clear_all_sessions.return_value = 10

        result = await logger.clear_all_sessions()

        assert result == 10
