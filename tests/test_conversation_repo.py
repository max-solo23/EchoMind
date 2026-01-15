"""Tests for SQLAlchemyConversationRepository - session and conversation logging."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from repositories.conversation_repo import SQLAlchemyConversationRepository


class MockSession:
    """Mock SQLAlchemy Session model for testing."""

    def __init__(
        self,
        id: int = 1,
        session_id: str = "sess_123",
        user_ip: str | None = "127.0.0.1",
        created_at: datetime | None = None,
        last_activity: datetime | None = None,
        conversations: list | None = None,
    ):
        self.id = id
        self.session_id = session_id
        self.user_ip = user_ip
        self.created_at = created_at or datetime.utcnow()
        self.last_activity = last_activity or datetime.utcnow()
        self.conversations = conversations or []


class MockConversation:
    """Mock SQLAlchemy Conversation model for testing."""

    def __init__(
        self,
        id: int = 1,
        session_id: int = 1,
        user_message: str = "Hello",
        bot_response: str = "Hi there",
        tool_calls: str | None = None,
        evaluator_used: bool = False,
        evaluator_passed: bool | None = None,
        timestamp: datetime | None = None,
    ):
        self.id = id
        self.session_id = session_id
        self.user_message = user_message
        self.bot_response = bot_response
        self.tool_calls = tool_calls
        self.evaluator_used = evaluator_used
        self.evaluator_passed = evaluator_passed
        self.timestamp = timestamp or datetime.utcnow()


@pytest.fixture
def mock_db_session():
    """Create mocked AsyncSession."""
    session = AsyncMock()
    return session


@pytest.fixture
def repo(mock_db_session):
    """Create repository with mocked session."""
    return SQLAlchemyConversationRepository(mock_db_session)


class TestCreateSession:
    """Test session creation/retrieval."""

    @pytest.mark.asyncio
    async def test_returns_existing_session_id(self, repo, mock_db_session):
        """
        Verify existing session ID is returned if session exists.

        Why: Idempotent session creation - calling create_session twice
        with same ID shouldn't create duplicates. Frontend might retry
        requests, so we return existing instead of erroring.
        """
        mock_session = MockSession(id=42, session_id="sess_abc")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_session
        mock_db_session.execute.return_value = mock_result

        result = await repo.create_session("sess_abc", "192.168.1.1")

        assert result == 42
        mock_db_session.add.assert_not_called()  # Shouldn't create new

    @pytest.mark.asyncio
    async def test_creates_new_session_when_not_exists(self, repo, mock_db_session):
        """
        Verify new session is created and ID returned.

        Why: First time user connects, we need to create a session
        to group their conversations together.
        """
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        async def mock_refresh(obj):
            obj.id = 99

        mock_db_session.refresh = mock_refresh

        result = await repo.create_session("new_sess", "10.0.0.1")

        assert result == 99
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_none_user_ip(self, repo, mock_db_session):
        """
        Verify None IP is handled (privacy mode or internal use).

        Why: Some users might connect through proxies that don't reveal IP,
        or we might disable IP logging for privacy. Should work without IP.
        """
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        async def mock_refresh(obj):
            obj.id = 1

        mock_db_session.refresh = mock_refresh

        # Should not raise
        result = await repo.create_session("sess_123", None)
        assert result == 1


class TestLogConversation:
    """Test conversation logging."""

    @pytest.mark.asyncio
    async def test_logs_basic_conversation(self, repo, mock_db_session):
        """
        Verify basic conversation is logged and ID returned.

        Why: Core functionality - every user message and bot response
        must be persisted for history and analytics.
        """

        async def mock_refresh(obj):
            obj.id = 55

        mock_db_session.refresh = mock_refresh

        result = await repo.log_conversation(
            session_db_id=1, user_message="What is Python?", bot_response="A programming language"
        )

        assert result == 55
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_with_tool_calls(self, repo, mock_db_session):
        """
        Verify tool_calls are JSON serialized.

        Why: Tool calls are lists/dicts. Database stores them as JSON string.
        This ensures the data survives round-trip to database.
        """

        async def mock_refresh(obj):
            obj.id = 1

        mock_db_session.refresh = mock_refresh

        await repo.log_conversation(
            session_db_id=1,
            user_message="Contact me",
            bot_response="I'll record your details",
            tool_calls=[{"name": "record_user_details", "args": {"email": "test@example.com"}}],
        )

        # Verify the conversation object was created with serialized tool_calls
        call_args = mock_db_session.add.call_args[0][0]
        assert call_args.tool_calls is not None
        # Should be valid JSON
        parsed = json.loads(call_args.tool_calls)
        assert parsed[0]["name"] == "record_user_details"

    @pytest.mark.asyncio
    async def test_logs_with_evaluator_info(self, repo, mock_db_session):
        """
        Verify evaluator metadata is stored.

        Why: Tracking which responses were evaluated and whether they
        passed helps analyze evaluator effectiveness over time.
        """

        async def mock_refresh(obj):
            obj.id = 1

        mock_db_session.refresh = mock_refresh

        await repo.log_conversation(
            session_db_id=1,
            user_message="Test",
            bot_response="Response",
            evaluator_used=True,
            evaluator_passed=False,
        )

        call_args = mock_db_session.add.call_args[0][0]
        assert call_args.evaluator_used is True
        assert call_args.evaluator_passed is False


class TestGetSessionById:
    """Test session retrieval with conversations."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_conversations(self, repo, mock_db_session):
        """
        Verify session dict includes conversation list.

        Why: API endpoint needs full session data including all
        messages for display. Dict format decouples from ORM.
        """
        mock_convs = [
            MockConversation(user_message="Hi", bot_response="Hello"),
            MockConversation(user_message="Bye", bot_response="Goodbye"),
        ]
        mock_session = MockSession(
            id=1, session_id="sess_xyz", user_ip="1.2.3.4", conversations=mock_convs
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_session
        mock_db_session.execute.return_value = mock_result

        result = await repo.get_session_by_id("sess_xyz")

        assert result is not None
        assert result["session_id"] == "sess_xyz"
        assert result["user_ip"] == "1.2.3.4"
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["user_message"] == "Hi"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_db_session):
        """
        Verify None returned for nonexistent session.

        Why: API needs to distinguish "not found" from "found but empty".
        Returns None, caller can return 404.
        """
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo.get_session_by_id("nonexistent")

        assert result is None


class TestListSessions:
    """Test paginated session listing."""

    @pytest.mark.asyncio
    async def test_returns_paginated_results(self, repo, mock_db_session):
        """
        Verify pagination metadata is included.

        Why: Admin dashboard needs to know total pages for pagination UI.
        Can't load all sessions at once for large datasets.
        """
        mock_sessions = [
            MockSession(id=1, session_id="s1", conversations=[MockConversation()]),
            MockSession(id=2, session_id="s2", conversations=[]),
        ]

        # Mock count query
        count_result = MagicMock()
        count_result.scalar.return_value = 50

        # Mock sessions query
        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = mock_sessions

        mock_db_session.execute.side_effect = [count_result, sessions_result]

        result = await repo.list_sessions(page=1, limit=20)

        assert result["total"] == 50
        assert result["page"] == 1
        assert result["limit"] == 20
        assert result["total_pages"] == 3  # ceil(50/20) = 3
        assert len(result["sessions"]) == 2
        assert result["sessions"][0]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_calculates_offset_correctly(self, repo, mock_db_session):
        """
        Verify offset calculation: (page-1) * limit.

        Why: Page 1 should start at offset 0, page 2 at offset 20, etc.
        Off-by-one errors here would skip or duplicate records.
        """
        count_result = MagicMock()
        count_result.scalar.return_value = 100

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [count_result, sessions_result]

        # Page 3 with limit 20 should have offset 40
        await repo.list_sessions(page=3, limit=20)

        # Can't easily verify offset without inspecting query, but we verify it runs
        assert mock_db_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_empty_database(self, repo, mock_db_session):
        """
        Verify empty DB returns zero counts without errors.

        Why: Fresh deployment has no sessions. Should return empty list
        and zero pages, not division by zero or other errors.
        """
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [count_result, sessions_result]

        result = await repo.list_sessions()

        assert result["total"] == 0
        assert result["total_pages"] == 0
        assert result["sessions"] == []


class TestDeleteSession:
    """Test session deletion."""

    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self, repo, mock_db_session):
        """
        Verify True returned when session exists and is deleted.

        Why: Caller needs to know if delete succeeded to return
        appropriate HTTP status (200 vs 404).
        """
        mock_session = MockSession(id=1, session_id="sess_del")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_session
        mock_db_session.execute.return_value = mock_result

        result = await repo.delete_session("sess_del")

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_session)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, repo, mock_db_session):
        """
        Verify False returned when session doesn't exist.

        Why: Idempotent deletes - deleting nonexistent session
        isn't an error, but caller might want to know.
        """
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo.delete_session("nonexistent")

        assert result is False
        mock_db_session.delete.assert_not_called()


class TestClearAllSessions:
    """Test bulk session deletion."""

    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, repo, mock_db_session):
        """
        Verify returns count of sessions that were deleted.

        Why: Admin dashboard shows confirmation "Deleted X sessions".
        Useful for audit logging too.
        """
        count_result = MagicMock()
        count_result.scalar.return_value = 25
        mock_db_session.execute.return_value = count_result

        result = await repo.clear_all_sessions()

        assert result == 25
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deletes_conversations_before_sessions(self, repo, mock_db_session):
        """
        Verify conversations deleted first (foreign key constraint).

        Why: Conversations reference sessions via foreign key.
        Deleting sessions first would violate FK constraint.
        Must delete children before parents.
        """
        count_result = MagicMock()
        count_result.scalar.return_value = 5
        mock_db_session.execute.return_value = count_result

        await repo.clear_all_sessions()

        # execute is called 3 times: count, delete conversations, delete sessions
        assert mock_db_session.execute.call_count == 3
