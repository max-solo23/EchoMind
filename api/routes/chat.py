import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from api.dependencies import (
    get_app_settings,
    get_chat_service,
    get_config,
    get_conversation_logger,
    is_database_configured,
    refresh_app_settings,
)
from api.middleware.auth import verify_api_key
from api.middleware.rate_limit import check_rate_limit
from core.chat import Chat, InvalidMessageError, SSEEvent
from models.requests import ChatRequest
from models.responses import ChatResponse
from repositories.connection import get_session


logger = logging.getLogger(__name__)

CONTEXT_TRUNCATE_LENGTH = 500
STREAMING_CHUNK_SIZE = 20
# A 2 KB comment line makes proxies and browsers flush headers and open the EventSource at once.
SSE_KICKSTART = (":" + " " * 2048 + "\n\n").encode()

router = APIRouter(prefix="/api/v1", tags=["chat"])

_background_tasks: set[asyncio.Task] = set()


def _run_in_background(coro) -> None:
    """Run bookkeeping off the request's critical path. The set holds a strong reference until done."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def drain_background_tasks() -> None:
    await asyncio.gather(*_background_tasks, return_exceptions=True)


def extract_last_assistant_message(history: list[dict]) -> str | None:
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content[:CONTEXT_TRUNCATE_LENGTH] if content else None
    return None


@dataclass(frozen=True)
class _Turn:
    """One user message plus the context the cache and the conversation log need."""

    message: str
    history: list[dict]
    session_id: str
    client_ip: str | None

    @property
    def last_assistant_message(self) -> str | None:
        return extract_last_assistant_message(self.history)

    @property
    def is_continuation(self) -> bool:
        return bool(self.history)


@router.post("/chat", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    chat_service: Annotated[Chat, Depends(get_chat_service)],
    stream: bool = Query(False, description="Enable streaming response (SSE)"),
    x_session_id: str | None = Header(None, description="Session ID for conversation tracking"),
):
    turn = _Turn(
        message=chat_request.message,
        history=chat_request.history,
        session_id=x_session_id or str(uuid.uuid4()),
        client_ip=request.client.host if request.client else None,
    )
    logger.info(
        f"Chat request - Session: {turn.session_id[:8]}..., "
        f"Message length: {len(turn.message)}, "
        f"Has history: {turn.is_continuation}, "
        f"IP: {turn.client_ip}, "
        f"Streaming: {stream}"
    )

    try:
        if stream:
            return StreamingResponse(
                _stream_with_logging(chat_service, turn),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Session-ID": turn.session_id,
                },
            )
        return ChatResponse(reply=await _chat_with_logging(chat_service, turn))

    except InvalidMessageError as e:
        logger.warning(f"Invalid message from {turn.client_ip}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request. Please try again.",
        ) from None


async def _chat_with_logging(chat_service: Chat, turn: _Turn) -> str:
    if not is_database_configured():
        return await chat_service.chat(turn.message, turn.history)

    await refresh_app_settings()
    cached_answer = await _get_cached_answer(turn) if get_app_settings().cache_enabled else None
    if cached_answer:
        logger.info(f"Cache hit for session {turn.session_id}")
        return cached_answer

    reply = await chat_service.chat(turn.message, turn.history)
    _run_in_background(_persist_in_background(turn, reply))
    return reply


async def _stream_with_logging(chat_service: Chat, turn: _Turn) -> AsyncGenerator[bytes, None]:
    # The first byte leaves before any database work so the client's EventSource opens immediately.
    yield SSE_KICKSTART

    if not is_database_configured():
        async for chunk in chat_service.chat_stream(turn.message, turn.history):
            yield chunk
        return

    await refresh_app_settings()
    cached_answer = await _get_cached_answer(turn) if get_app_settings().cache_enabled else None
    if cached_answer:
        logger.info(f"Cache hit (streaming) for session {turn.session_id}")
        for i in range(0, len(cached_answer), STREAMING_CHUNK_SIZE):
            text_chunk = cached_answer[i : i + STREAMING_CHUNK_SIZE]
            yield SSEEvent(delta=text_chunk, metadata={"cached": True}).encode()
        yield SSEEvent(metadata={"done": True, "cached": True}).encode()
        return

    deltas: list[str] = []
    async for chunk in chat_service.chat_stream(turn.message, turn.history):
        yield chunk
        if delta := _extract_delta(chunk):
            deltas.append(delta)

    if deltas:
        _run_in_background(_persist_in_background(turn, "".join(deltas)))


def _extract_delta(chunk: bytes) -> str | None:
    if not chunk.startswith(b"data: "):
        return None
    try:
        return json.loads(chunk[6:]).get("delta")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


async def _persist_in_background(turn: _Turn, reply: str) -> None:
    try:
        await _persist_reply(turn, reply)
        logger.info(f"Logged conversation for session {turn.session_id}")
    except Exception:
        logger.exception(f"Failed to log conversation for session {turn.session_id}")


async def _get_cached_answer(turn: _Turn) -> str | None:
    async with get_session(get_config()) as session:
        conversation_logger = await get_conversation_logger(session)
        cached_answer = await conversation_logger.check_cache(
            question=turn.message,
            last_assistant_message=turn.last_assistant_message,
            is_continuation=turn.is_continuation,
        )
        if cached_answer:
            await _log_reply(conversation_logger, turn, cached_answer, cache_response=False)
        return cached_answer


async def _persist_reply(turn: _Turn, reply: str) -> None:
    async with get_session(get_config()) as session:
        conversation_logger = await get_conversation_logger(session)
        await _log_reply(conversation_logger, turn, reply, cache_response=True)


async def _log_reply(conversation_logger, turn: _Turn, reply: str, *, cache_response: bool) -> None:
    session_db_id = await conversation_logger.get_or_create_session(turn.session_id, turn.client_ip)
    await conversation_logger.log_and_cache(
        session_db_id=session_db_id,
        user_message=turn.message,
        bot_response=reply,
        cache_response=cache_response,
        last_assistant_message=turn.last_assistant_message,
        is_continuation=turn.is_continuation,
    )
