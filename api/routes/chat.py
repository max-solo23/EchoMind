from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, Optional
import uuid
import logging

from models.requests import ChatRequest
from models.responses import ChatResponse
from api.middleware.auth import verify_api_key
from api.middleware.rate_limit import limiter
from api.dependencies import (
    get_chat_service,
    get_db_session,
    get_conversation_logger,
    is_database_configured,
)
from Chat import Chat
from services.conversation_logger import ConversationLogger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])




@router.post("/chat", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/hour")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    chat_service: Annotated[Chat, Depends(get_chat_service)],
    stream: bool = Query(False, description="Enable streaming response (SSE)"),
    x_session_id: Optional[str] = Header(None, description="Session ID for conversation tracking")
):
    """
    Chat endpoint with optional streaming support and conversation logging.

    - Without stream parameter: Returns complete response (with evaluator if enabled)
    - With stream=true: Returns SSE stream (no evaluator)
    - X-Session-ID header: Optional session ID for conversation tracking

    When database is configured:
    - Checks cache for similar questions (90% TF-IDF match)
    - Logs all conversations
    - Caches responses for future use
    """
    try:
        # Generate session ID if not provided
        session_id = x_session_id or str(uuid.uuid4())

        # Get client IP for logging
        client_ip = request.client.host if request.client else None

        if stream:
            # Streaming mode - no caching (response built incrementally)
            return StreamingResponse(
                chat_service.chat_stream(chat_request.message, chat_request.history),
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

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        )


async def _chat_with_logging(
    chat_service: Chat,
    message: str,
    history: list[dict],
    session_id: str,
    client_ip: Optional[str]
) -> str:
    """
    Handle chat with optional caching and logging.

    Flow:
    1. If DB configured: check cache for similar question
    2. If cache hit: return cached answer
    3. If cache miss: call LLM
    4. If DB configured: log conversation and cache response
    """
    if not is_database_configured():
        # No database - just call LLM directly
        return chat_service.chat(message, history)

    # Database is configured - use caching and logging
    from database import get_session
    from api.dependencies import get_config

    config = get_config()

    async with get_session(config) as session:
        conversation_logger = await get_conversation_logger(session)

        # Check cache first
        cached_answer = await conversation_logger.check_cache(message)
        if cached_answer:
            logger.info(f"Cache hit for session {session_id}")
            # Log the cached response too
            session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
            await conversation_logger.log_and_cache(
                session_db_id=session_db_id,
                user_message=message,
                bot_response=cached_answer,
                cache_response=False  # Don't re-cache
            )
            return cached_answer

        # Cache miss - call LLM
        reply = chat_service.chat(message, history)

        # Log and cache the response
        session_db_id = await conversation_logger.get_or_create_session(session_id, client_ip)
        await conversation_logger.log_and_cache(
            session_db_id=session_db_id,
            user_message=message,
            bot_response=reply,
            evaluator_used=chat_service.evaluator_llm is not None,
            cache_response=True
        )

        logger.info(f"Logged conversation for session {session_id}")
        return reply
    
