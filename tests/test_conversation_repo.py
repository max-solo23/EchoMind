import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from repositories.conversation_repo import SQLAlchemyConversationRepository


class MockSession:
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
    session = AsyncMock()
    return session


@pytest.fixture
def repo(mock_db_session):
    return SQLAlchemyConversationRepository(mock_db_session)


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_returns_existing_session_id(self, repo, mock_db_session):
        mock_session = MockSession(id=42, session_id="sess_abc")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_session
        mock_db_session.execute.return_value = mock_result

        result = await repo.create_session("sess_abc", "192.168.1.1")

        assert result == 42
        mock_db_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_new_session_when_not_exists(self, repo, mock_db_session):
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
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        async def mock_refresh(obj):
            obj.id = 1

        mock_db_session.refresh = mock_refresh

        result = await repo.create_session("sess_123", None)
        assert result == 1


class TestLogConversation:
    @pytest.mark.asyncio
    async def test_logs_basic_conversation(self, repo, mock_db_session):
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
        async def mock_refresh(obj):
            obj.id = 1

        mock_db_session.refresh = mock_refresh

        await repo.log_conversation(
            session_db_id=1,
            user_message="Contact me",
            bot_response="I'll record your details",
            tool_calls=[{"name": "record_user_details", "args": {"email": "test@example.com"}}],
        )

        call_args = mock_db_session.add.call_args[0][0]
        assert call_args.tool_calls is not None
        parsed = json.loads(call_args.tool_calls)
        assert parsed[0]["name"] == "record_user_details"

    @pytest.mark.asyncio
    async def test_logs_with_evaluator_info(self, repo, mock_db_session):
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
    @pytest.mark.asyncio
    async def test_returns_dict_with_conversations(self, repo, mock_db_session):
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
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo.get_session_by_id("nonexistent")

        assert result is None


class TestListSessions:
    @pytest.mark.asyncio
    async def test_returns_paginated_results(self, repo, mock_db_session):
        mock_sessions = [
            MockSession(id=1, session_id="s1", conversations=[MockConversation()]),
            MockSession(id=2, session_id="s2", conversations=[]),
        ]

        count_result = MagicMock()
        count_result.scalar.return_value = 50

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = mock_sessions

        mock_db_session.execute.side_effect = [count_result, sessions_result]

        result = await repo.list_sessions(page=1, limit=20)

        assert result["total"] == 50
        assert result["page"] == 1
        assert result["limit"] == 20
        assert result["total_pages"] == 3
        assert len(result["sessions"]) == 2
        assert result["sessions"][0]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_calculates_offset_correctly(self, repo, mock_db_session):
        count_result = MagicMock()
        count_result.scalar.return_value = 100

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [count_result, sessions_result]

        await repo.list_sessions(page=3, limit=20)

        assert mock_db_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_empty_database(self, repo, mock_db_session):
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
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self, repo, mock_db_session):
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
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo.delete_session("nonexistent")

        assert result is False
        mock_db_session.delete.assert_not_called()


class TestClearAllSessions:
    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, repo, mock_db_session):
        count_result = MagicMock()
        count_result.scalar.return_value = 25
        mock_db_session.execute.return_value = count_result

        result = await repo.clear_all_sessions()

        assert result == 25
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deletes_conversations_before_sessions(self, repo, mock_db_session):
        count_result = MagicMock()
        count_result.scalar.return_value = 5
        mock_db_session.execute.return_value = count_result

        await repo.clear_all_sessions()

        assert mock_db_session.execute.call_count == 3
