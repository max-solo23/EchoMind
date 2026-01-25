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
from api.middleware.rate_limit import check_rate_limit
from Chat import Chat, InvalidMessageError
from database import get_session
from models.requests import ChatRequest
from models.responses import ChatResponse


logger = logging.getLogger(__name__)

CONTEXT_TRUNCATE_LENGTH = 500
SSE_KICKSTART_BUFFER_SIZE = 2048
STREAMING_CHUNK_SIZE = 20

router = APIRouter(prefix="/api/v1", tags=["chat"])


def extract_last_assistant_message(history: list[dict]) -> str | None:
    if not history:
        return None

    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content[:CONTEXT_TRUNCATE_LENGTH] if content else None

    return None


@router.post("/chat", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    chat_service: Annotated[Chat, Depends(get_chat_service)],
    stream: bool = Query(False, description="Enable streaming response (SSE)"),
    x_session_id: str | None = Header(None, description="Session ID for conversation tracking"),
):
    try:
        session_id = x_session_id or str(uuid.uuid4())

        client_ip = request.client.host if request.client else None

        logger.info(
            f"Chat request - Session: {session_id[:8]}..., "
            f"Message length: {len(chat_request.message)}, "
            f"Has history: {len(chat_request.history) > 0}, "
            f"IP: {client_ip}, "
            f"Streaming: {stream}"
        )

        if stream:
            return StreamingResponse(
                _stream_with_logging(
                    chat_service=chat_service,
                    message=chat_request.message,
                    history=chat_request.history,
                    session_id=session_id,
                    client_ip=client_ip,
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
            reply = await _chat_with_logging(
                chat_service=chat_service,
                message=chat_request.message,
                history=chat_request.history,
                session_id=session_id,
                client_ip=client_ip,
            )

            return ChatResponse(reply=reply)

    except InvalidMessageError as e:
        logger.warning(f"Invalid message from {client_ip}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}",
        ) from None


async def _chat_with_logging(
    chat_service: Chat, message: str, history: list[dict], session_id: str, client_ip: str | None
) -> str:
    if not is_database_configured():
        return chat_service.chat(message, history)

    last_assistant_msg = extract_last_assistant_message(history)
    is_continuation = len(history) > 0

    config = get_config()

    async with get_session(config) as session:
        conversation_logger = await get_conversation_logger(session)

        cached_answer = await conversation_logger.check_cache(
            question=message,
            last_assistant_message=last_assistant_msg,
            is_continuation=is_continuation,
        )
        if cached_answer:
            logger.info(f"Cache hit for session {session_id}")
            session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
            await conversation_logger.log_and_cache(
                session_db_id=session_db_id,
                user_message=message,
                bot_response=cached_answer,
                cache_response=False,
                last_assistant_message=last_assistant_msg,
                is_continuation=is_continuation,
            )
            return cached_answer

        reply = chat_service.chat(message, history)

        session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
        await conversation_logger.log_and_cache(
            session_db_id=session_db_id,
            user_message=message,
            bot_response=reply,
            evaluator_used=False,
            cache_response=True,
            last_assistant_message=last_assistant_msg,
            is_continuation=is_continuation,
        )

        logger.info(f"Logged conversation for session {session_id}")
        return reply


async def _stream_with_logging(
    chat_service: Chat, message: str, history: list[dict], session_id: str, client_ip: str | None
) -> AsyncGenerator[bytes, None]:
    last_assistant_msg = extract_last_assistant_message(history)
    is_continuation = len(history) > 0

    if not is_database_configured():
        async for chunk in chat_service.chat_stream(message, history):
            yield chunk
        return

    config = get_config()

    async with get_session(config) as session:
        conversation_logger = await get_conversation_logger(session)

        cached_answer = await conversation_logger.check_cache(
            question=message,
            last_assistant_message=last_assistant_msg,
            is_continuation=is_continuation,
        )
        if cached_answer:
            logger.info(f"Cache hit (streaming) for session {session_id}")

            session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
            await conversation_logger.log_and_cache(
                session_db_id=session_db_id,
                user_message=message,
                bot_response=cached_answer,
                cache_response=False,
                last_assistant_message=last_assistant_msg,
                is_continuation=is_continuation,
            )

            yield (":" + (" " * SSE_KICKSTART_BUFFER_SIZE) + "\n\n").encode(
                "utf-8"
            )

            for i in range(0, len(cached_answer), STREAMING_CHUNK_SIZE):
                text_chunk = cached_answer[i : i + STREAMING_CHUNK_SIZE]
                event: dict[str, str | dict[str, bool]] = {
                    "delta": text_chunk,
                    "metadata": {"cached": True},
                }
                yield f"data: {json.dumps(event)}\n\n".encode()

            done_event: dict[str, str | None | dict[str, bool]] = {
                "delta": None,
                "metadata": {"done": True, "cached": True},
            }
            yield f"data: {json.dumps(done_event)}\n\n".encode()
            return

    accumulated_response = []

    async for chunk in chat_service.chat_stream(message, history):
        yield chunk

        try:
            chunk_str = chunk.decode("utf-8")
            if chunk_str.startswith("data: "):
                data = json.loads(chunk_str[6:].strip())
                if data.get("delta"):
                    accumulated_response.append(data["delta"])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    full_response = "".join(accumulated_response)

    if full_response and is_database_configured():
        try:
            async with get_session(config) as session:
                conversation_logger = await get_conversation_logger(session)
                session_db_id = await conversation_logger.get_or_create_session(
                    session_id, client_ip
                )
                await conversation_logger.log_and_cache(
                    session_db_id=session_db_id,
                    user_message=message,
                    bot_response=full_response,
                    evaluator_used=False,
                    cache_response=True,
                    last_assistant_message=last_assistant_msg,
                    is_continuation=is_continuation,
                )
                logger.info(f"Logged streaming conversation for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log streaming conversation: {e}")
