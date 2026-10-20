import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.main import app
from api.middleware.rate_limit import check_rate_limit
from api.routes.chat import (
    _chat_with_logging,
    _stream_with_logging,
    _Turn,
    drain_background_tasks,
    extract_last_assistant_message,
)
from core.chat import InvalidMessageError
from services.cache_service import CacheLookup


KICKSTART = b":" + b" " * 2048 + b"\n\n"


@asynccontextmanager
async def session_context(session: object) -> AsyncIterator[object]:
    yield session


def session_factory(*sessions: object):
    iterator = iter(sessions)

    def get_test_session(_config: object):
        return session_context(next(iterator))

    return get_test_session


@pytest.mark.asyncio
async def test_cached_chat_logs_cached_reply_without_calling_service():
    chat_service = AsyncMock()
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup(entry_id=4, answer="Cached reply")
    conversation_logger.get_or_create_session.return_value = 7

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object(), object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        reply = await _chat_with_logging(
            chat_service, _Turn("Question", [], "session-id", "127.0.0.1")
        )
        conversation_logger.log_and_cache.assert_not_awaited()  # off the critical path
        await drain_background_tasks()

    assert reply == "Cached reply"
    chat_service.chat.assert_not_awaited()
    conversation_logger.log_and_cache.assert_awaited_once_with(
        session_db_id=7,
        user_message="Question",
        bot_response="Cached reply",
        cache_response=False,
        last_assistant_message=None,
        is_continuation=False,
    )
    conversation_logger.record_cache_hit.assert_awaited_once_with(4)


@pytest.mark.asyncio
async def test_uncached_chat_calls_service_then_persists_reply():
    chat_service = AsyncMock()
    chat_service.chat.return_value = "Generated reply"
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup()
    conversation_logger.get_or_create_session.return_value = 9

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object(), object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        reply = await _chat_with_logging(
            chat_service, _Turn("Question", [], "session-id", "127.0.0.1")
        )
        await drain_background_tasks()

    assert reply == "Generated reply"
    chat_service.chat.assert_awaited_once_with("Question", [])
    conversation_logger.log_and_cache.assert_awaited_once_with(
        session_db_id=9,
        user_message="Question",
        bot_response="Generated reply",
        cache_response=True,
        last_assistant_message=None,
        is_continuation=False,
    )


@pytest.mark.asyncio
async def test_streamed_chat_persists_accumulated_sse_reply():
    async def stream() -> AsyncIterator[bytes]:
        yield b'data: {"delta": "Generated "}\n\n'
        yield b'data: {"delta": "reply"}\n\n'
        yield b'data: {"delta": null, "metadata": {"done": true}}\n\n'

    chat_service = MagicMock()
    chat_service.chat_stream.return_value = stream()
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup()
    conversation_logger.get_or_create_session.return_value = 11

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object(), object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        chunks = [
            chunk
            async for chunk in _stream_with_logging(
                chat_service, _Turn("Question", [], "session-id", "127.0.0.1")
            )
        ]
        await drain_background_tasks()

    assert chunks == [
        KICKSTART,
        b'data: {"delta": "Generated "}\n\n',
        b'data: {"delta": "reply"}\n\n',
        b'data: {"delta": null, "metadata": {"done": true}}\n\n',
    ]
    conversation_logger.log_and_cache.assert_awaited_once_with(
        session_db_id=11,
        user_message="Question",
        bot_response="Generated reply",
        cache_response=True,
        last_assistant_message=None,
        is_continuation=False,
    )


class TestExtractLastAssistantMessage:
    def test_returns_none_for_empty_history(self):
        assert extract_last_assistant_message([]) is None

    def test_returns_none_when_no_assistant_turn(self):
        assert extract_last_assistant_message([{"role": "user", "content": "hi"}]) is None

    def test_returns_most_recent_assistant_content(self):
        history = [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "last"},
        ]

        assert extract_last_assistant_message(history) == "second"

    def test_truncates_long_content(self):
        history = [{"role": "assistant", "content": "x" * 600}]

        assert extract_last_assistant_message(history) == "x" * 500

    def test_returns_none_for_empty_assistant_content(self):
        assert extract_last_assistant_message([{"role": "assistant", "content": ""}]) is None


@pytest.mark.asyncio
async def test_chat_without_database_skips_logging():
    chat_service = AsyncMock()
    chat_service.chat.return_value = "Plain reply"

    with (
        patch("api.routes.chat.is_database_configured", return_value=False),
        patch("api.routes.chat.get_config", side_effect=AssertionError("config must not be read")),
    ):
        reply = await _chat_with_logging(chat_service, _Turn("Question", [], "session-id", None))

    assert reply == "Plain reply"
    chat_service.chat.assert_awaited_once_with("Question", [])


@pytest.mark.asyncio
async def test_stream_without_database_passes_chunks_through():
    async def stream() -> AsyncIterator[bytes]:
        yield b'data: {"delta": "a"}\n\n'

    chat_service = MagicMock()
    chat_service.chat_stream.return_value = stream()

    with patch("api.routes.chat.is_database_configured", return_value=False):
        chunks = [
            c async for c in _stream_with_logging(chat_service, _Turn("Question", [], "s", None))
        ]

    assert chunks == [KICKSTART, b'data: {"delta": "a"}\n\n']


@pytest.mark.asyncio
async def test_cached_stream_replays_answer_in_chunks_without_calling_service():
    cached = "x" * 45
    chat_service = MagicMock()
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup(entry_id=4, answer=cached)
    conversation_logger.get_or_create_session.return_value = 3

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object(), object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        chunks = [
            c async for c in _stream_with_logging(chat_service, _Turn("Question", [], "s", None))
        ]
        await drain_background_tasks()

    assert chunks[0] == KICKSTART
    assert [json.loads(c[6:]) for c in chunks[1:]] == [
        {"delta": "x" * 20, "metadata": {"cached": True}},
        {"delta": "x" * 20, "metadata": {"cached": True}},
        {"delta": "x" * 5, "metadata": {"cached": True}},
        {"delta": None, "metadata": {"done": True, "cached": True}},
    ]
    chat_service.chat_stream.assert_not_called()
    conversation_logger.log_and_cache.assert_awaited_once_with(
        session_db_id=3,
        user_message="Question",
        bot_response=cached,
        cache_response=False,
        last_assistant_message=None,
        is_continuation=False,
    )
    conversation_logger.record_cache_hit.assert_awaited_once_with(4)


@pytest.mark.asyncio
async def test_stream_ignores_non_json_chunks_and_records_conversation_context():
    async def stream() -> AsyncIterator[bytes]:
        yield b": keepalive\n\n"
        yield b"data: not json\n\n"
        yield b'data: {"delta": "ok"}\n\n'

    history = [{"role": "assistant", "content": "earlier"}]
    chat_service = MagicMock()
    chat_service.chat_stream.return_value = stream()
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup()
    conversation_logger.get_or_create_session.return_value = 5

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object(), object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        chunks = [
            c async for c in _stream_with_logging(chat_service, _Turn("Q", history, "s", None))
        ]
        await drain_background_tasks()

    assert len(chunks) == 4
    conversation_logger.check_cache.assert_awaited_once_with(
        question="Q", last_assistant_message="earlier", is_continuation=True
    )
    conversation_logger.log_and_cache.assert_awaited_once_with(
        session_db_id=5,
        user_message="Q",
        bot_response="ok",
        cache_response=True,
        last_assistant_message="earlier",
        is_continuation=True,
    )


@pytest.mark.asyncio
async def test_stream_with_empty_reply_is_not_persisted():
    async def stream() -> AsyncIterator[bytes]:
        yield b'data: {"delta": null, "metadata": {"done": true}}\n\n'

    chat_service = MagicMock()
    chat_service.chat_stream.return_value = stream()
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup()

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        chunks = [c async for c in _stream_with_logging(chat_service, _Turn("Q", [], "s", None))]
        await drain_background_tasks()

    assert len(chunks) == 2
    conversation_logger.log_and_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_persist_failure_does_not_break_stream():
    async def stream() -> AsyncIterator[bytes]:
        yield b'data: {"delta": "ok"}\n\n'

    chat_service = MagicMock()
    chat_service.chat_stream.return_value = stream()
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup()
    conversation_logger.log_and_cache.side_effect = RuntimeError("db down")

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object(), object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        chunks = [c async for c in _stream_with_logging(chat_service, _Turn("Q", [], "s", None))]
        await drain_background_tasks()

    assert chunks == [KICKSTART, b'data: {"delta": "ok"}\n\n']
    conversation_logger.log_and_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_sends_first_byte_before_touching_the_database():
    conversation_logger = AsyncMock()
    conversation_logger.check_cache.return_value = CacheLookup()

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_config", return_value=object()),
        patch("api.routes.chat.get_session", session_factory(object())),
        patch("api.routes.chat.get_conversation_logger", return_value=conversation_logger),
    ):
        stream = _stream_with_logging(MagicMock(), _Turn("Q", [], "s", None))
        first = await anext(stream)
        conversation_logger.check_cache.assert_not_awaited()
        await stream.aclose()

    assert first == KICKSTART


@pytest.fixture
def no_rate_limit():
    app.dependency_overrides[check_rate_limit] = lambda: None
    yield
    app.dependency_overrides.pop(check_rate_limit, None)


HEADERS = {"X-API-Key": "test-api-key"}


class TestChatEndpoint:
    def test_returns_reply(self, client, mock_chat_service, no_rate_limit):
        history = [{"role": "user", "content": "hi"}]

        response = client.post(
            "/api/v1/chat", json={"message": "hello there", "history": history}, headers=HEADERS
        )

        assert response.status_code == 200
        assert response.json() == {"reply": "Test response"}
        mock_chat_service.chat.assert_awaited_once_with("hello there", history)

    def test_invalid_message_returns_400(self, client, mock_chat_service, no_rate_limit):
        mock_chat_service.chat.side_effect = InvalidMessageError("Your message is incomplete")

        response = client.post("/api/v1/chat", json={"message": "??"}, headers=HEADERS)

        assert response.status_code == 400
        assert "incomplete" in response.json()["detail"]

    def test_unexpected_error_returns_500(self, client, mock_chat_service, no_rate_limit):
        mock_chat_service.chat.side_effect = RuntimeError("boom")

        response = client.post("/api/v1/chat", json={"message": "hello there"}, headers=HEADERS)

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to process chat request. Please try again."

    def test_empty_message_rejected_by_validation(self, client, no_rate_limit):
        response = client.post("/api/v1/chat", json={"message": ""}, headers=HEADERS)

        assert response.status_code == 422

    def test_streaming_response_sets_sse_headers(self, client, mock_chat_service, no_rate_limit):
        async def stream() -> AsyncIterator[bytes]:
            yield b'data: {"delta": "hi"}\n\n'
            yield b'data: {"delta": null, "metadata": {"done": true}}\n\n'

        mock_chat_service.chat_stream.return_value = stream()

        response = client.post(
            "/api/v1/chat?stream=true",
            json={"message": "hello there"},
            headers={**HEADERS, "X-Session-ID": "session-42"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["X-Session-ID"] == "session-42"
        assert response.headers["Cache-Control"] == "no-cache, no-transform"
        assert b'"delta": "hi"' in response.content
        mock_chat_service.chat_stream.assert_called_once_with("hello there", [])

