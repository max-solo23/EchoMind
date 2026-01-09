import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from api.dependencies import (
    get_chat_service,
    get_config,
    get_conversation_logger,
    is_database_configured,
)
from api.middleware.auth import verify_api_key
from api.middleware.rate_limit import limiter
from api.middleware.rate_limit_state import rate_limit_state
from Chat import Chat, InvalidMessageError
from models.requests import ChatRequest
from models.responses import ChatResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def extract_last_assistant_message(history: list[dict]) -> str | None:
    """
    Extract the last assistant message from conversation history.

    Used for context-aware cache keys to prevent cross-conversation contamination.

    Args:
        history: List of conversation turns [{"role": "user/assistant", "content": "..."}]

    Returns:
        Last assistant message content (truncated to 500 chars) or None
    """
    if not history:
        return None

    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Truncate to 500 chars for cache key efficiency
            return content[:500] if content else None

    return None


def get_rate_limit() -> str:
    """
    Returns the rate limit string based on configuration.

    If rate limiting is disabled, returns a very high limit effectively disabling it.
    """
    if rate_limit_state.enabled:
        return f"{rate_limit_state.rate_per_hour}/hour"
    return "1000000/hour"  # Effectively unlimited when disabled


@router.post("/chat", dependencies=[Depends(verify_api_key)])
@limiter.limit(get_rate_limit)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    chat_service: Annotated[Chat, Depends(get_chat_service)],
    stream: bool = Query(False, description="Enable streaming response (SSE)"),
    x_session_id: str | None = Header(None, description="Session ID for conversation tracking")
):
    """
    Chat endpoint with optional streaming support and conversation logging.

    - Without stream parameter: Returns complete response (with evaluator if enabled)
    - With stream=true: Returns SSE stream (no evaluator)
    - X-Session-ID header: Optional session ID for conversation tracking

    When database is configured:
    - Checks cache for similar questions (context-aware, 90% TF-IDF match)
    - Skips cache for acknowledgements like "ok", "thanks" in continuations
    - Logs all conversations
    - Caches responses with TTL (24h conversational, 30d knowledge)
    """
    try:
        # Generate session ID if not provided
        session_id = x_session_id or str(uuid.uuid4())

        # Get client IP for logging
        client_ip = request.client.host if request.client else None

        # Log request details
        logger.info(
            f"Chat request - Session: {session_id[:8]}..., "
            f"Message length: {len(chat_request.message)}, "
            f"Has history: {len(chat_request.history) > 0}, "
            f"IP: {client_ip}, "
            f"Streaming: {stream}"
        )

        if stream:
            # Streaming mode with caching support
            return StreamingResponse(
                _stream_with_logging(
                    chat_service=chat_service,
                    message=chat_request.message,
                    history=chat_request.history,
                    session_id=session_id,
                    client_ip=client_ip
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Session-ID": session_id,
                },
            )
        else:
            # Non-streaming mode with caching support
            reply = await _chat_with_logging(
                chat_service=chat_service,
                message=chat_request.message,
                history=chat_request.history,
                session_id=session_id,
                client_ip=client_ip
            )

            return ChatResponse(reply=reply)

    except InvalidMessageError as e:
        logger.warning(f"Invalid message from {client_ip}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from None
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        ) from None


async def _chat_with_logging(
    chat_service: Chat,
    message: str,
    history: list[dict],
    session_id: str,
    client_ip: str | None
) -> str:
    """
    Handle chat with context-aware caching and logging.

    Flow:
    1. Extract context (last assistant message, is_continuation)
    2. If DB configured: check cache with context awareness
    3. If cache hit: return cached answer
    4. If cache miss: call LLM
    5. If DB configured: log conversation and cache with TTL
    """
    if not is_database_configured():
        # No database - just call LLM directly
        return chat_service.chat(message, history)

    # Extract context for cache key
    last_assistant_msg = extract_last_assistant_message(history)
    is_continuation = len(history) > 0

    # Database is configured - use caching and logging
    from api.dependencies import get_config
    from database import get_session

    config = get_config()

    async with get_session(config) as session:
        conversation_logger = await get_conversation_logger(session)

        # Check cache with context awareness
        cached_answer = await conversation_logger.check_cache(
            question=message,
            last_assistant_message=last_assistant_msg,
            is_continuation=is_continuation
        )
        if cached_answer:
            logger.info(f"Cache hit for session {session_id}")
            # Log the cached response too
            session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
            await conversation_logger.log_and_cache(
                session_db_id=session_db_id,
                user_message=message,
                bot_response=cached_answer,
                cache_response=False,  # Don't re-cache
                last_assistant_message=last_assistant_msg,
                is_continuation=is_continuation
            )
            return cached_answer

        # Cache miss - call LLM
        reply = chat_service.chat(message, history)

        # Log and cache the response with context
        session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
        await conversation_logger.log_and_cache(
            session_db_id=session_db_id,
            user_message=message,
            bot_response=reply,
            evaluator_used=chat_service.evaluator_llm is not None,
            cache_response=True,
            last_assistant_message=last_assistant_msg,
            is_continuation=is_continuation
        )

        logger.info(f"Logged conversation for session {session_id}")
        return reply


async def _stream_with_logging(
    chat_service: Chat,
    message: str,
    history: list[dict],
    session_id: str,
    client_ip: str | None
) -> AsyncGenerator[bytes, None]:
    """
    Stream chat response with context-aware caching and logging.

    Flow:
    1. Extract context (last assistant message, is_continuation)
    2. Check cache with context - if hit, stream cached answer
    3. If cache miss, stream from LLM while accumulating response
    4. After streaming completes, log and cache with TTL
    """
    # Extract context for cache key
    last_assistant_msg = extract_last_assistant_message(history)
    is_continuation = len(history) > 0

    if not is_database_configured():
        # No database - just stream directly
        async for chunk in chat_service.chat_stream(message, history):
            yield chunk
        return

    # Database is configured - check cache first
    from database import get_session

    config = get_config()

    async with get_session(config) as session:
        conversation_logger = await get_conversation_logger(session)

        # Check cache with context awareness
        cached_answer = await conversation_logger.check_cache(
            question=message,
            last_assistant_message=last_assistant_msg,
            is_continuation=is_continuation
        )
        if cached_answer:
            logger.info(f"Cache hit (streaming) for session {session_id}")

            # Log the cached response
            session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
            await conversation_logger.log_and_cache(
                session_db_id=session_db_id,
                user_message=message,
                bot_response=cached_answer,
                cache_response=False,
                last_assistant_message=last_assistant_msg,
                is_continuation=is_continuation
            )

            # Stream the cached answer as SSE events
            yield (":" + (" " * 2048) + "\n\n").encode("utf-8")  # Kick-start for buffering

            # Stream cached answer in chunks for natural feel
            chunk_size = 20
            for i in range(0, len(cached_answer), chunk_size):
                chunk = cached_answer[i:i + chunk_size]
                event = {"delta": chunk, "metadata": {"cached": True}}
                yield f"data: {json.dumps(event)}\n\n".encode()

            # Completion event
            event = {"delta": None, "metadata": {"done": True, "cached": True}}
            yield f"data: {json.dumps(event)}\n\n".encode()
            return

    # Cache miss - stream from LLM and accumulate response
    accumulated_response = []

    async for chunk in chat_service.chat_stream(message, history):
        yield chunk

        # Try to extract delta content from the chunk
        try:
            chunk_str = chunk.decode("utf-8")
            if chunk_str.startswith("data: "):
                data = json.loads(chunk_str[6:].strip())
                if data.get("delta"):
                    accumulated_response.append(data["delta"])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # After streaming completes, log and cache with context
    full_response = "".join(accumulated_response)

    if full_response and is_database_configured():
        try:
            async with get_session(config) as session:
                conversation_logger = await get_conversation_logger(session)
                session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
                await conversation_logger.log_and_cache(
                    session_db_id=session_db_id,
                    user_message=message,
                    bot_response=full_response,
                    evaluator_used=False,  # Evaluator not used in streaming
                    cache_response=True,
                    last_assistant_message=last_assistant_msg,
                    is_continuation=is_continuation
                )
                logger.info(f"Logged streaming conversation for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log streaming conversation: {e}")
