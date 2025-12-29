"""
Admin API routes for EchoMind.

Provides endpoints for:
- Cache management (stats, clear)
- Session history viewing
- System health with database status

All admin endpoints require API key authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from api.middleware.auth import verify_api_key
from api.dependencies import (
    get_db_session,
    get_conversation_logger,
    is_database_configured,
)
from services.conversation_logger import ConversationLogger


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(verify_api_key)]
)


# Response models
class CacheStats(BaseModel):
    total_questions: int
    total_variations: int
    avg_variations_per_question: float


class ClearCacheResponse(BaseModel):
    success: bool
    deleted_count: int


class ConversationItem(BaseModel):
    user_message: str
    bot_response: str
    timestamp: datetime


class SessionHistory(BaseModel):
    id: int
    session_id: str
    user_ip: Optional[str]
    created_at: datetime
    conversations: list[ConversationItem]


class DatabaseStatus(BaseModel):
    configured: bool
    connected: bool
    error: Optional[str] = None


class AdminHealthResponse(BaseModel):
    status: str
    database: DatabaseStatus
    cache: Optional[CacheStats] = None


def require_database():
    """Dependency that ensures database is configured."""
    if not is_database_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured. Set POSTGRES_URL environment variable."
        )


@router.get("/health", response_model=AdminHealthResponse)
async def admin_health():
    """
    Extended health check with database status.

    Returns system status including database connectivity and cache stats.
    """
    db_status = DatabaseStatus(
        configured=is_database_configured(),
        connected=False
    )

    cache_stats = None

    if is_database_configured():
        try:
            from api.dependencies import get_config
            from database import get_session

            config = get_config()
            async with get_session(config) as session:
                logger = await get_conversation_logger(session)
                stats = await logger.get_cache_stats()
                cache_stats = CacheStats(**stats)
                db_status.connected = True
        except Exception as e:
            db_status.error = str(e)

    return AdminHealthResponse(
        status="healthy" if db_status.connected else "degraded",
        database=db_status,
        cache=cache_stats
    )


@router.get(
    "/cache/stats",
    response_model=CacheStats,
    dependencies=[Depends(require_database)]
)
async def get_cache_stats(
    session: AsyncSession = Depends(get_db_session)
):
    """
    Get cache statistics.

    Returns:
    - total_questions: Number of cached questions
    - total_variations: Total answer variations stored
    - avg_variations_per_question: Average variations per question
    """
    logger = await get_conversation_logger(session)
    stats = await logger.get_cache_stats()
    return CacheStats(**stats)


@router.delete(
    "/cache",
    response_model=ClearCacheResponse,
    dependencies=[Depends(require_database)]
)
async def clear_cache(
    session: AsyncSession = Depends(get_db_session)
):
    """
    Clear all cached answers.

    Use with caution - this will delete all cached question/answer pairs.
    """
    logger = await get_conversation_logger(session)
    deleted = await logger.clear_cache()
    return ClearCacheResponse(success=True, deleted_count=deleted)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistory,
    dependencies=[Depends(require_database)]
)
async def get_session_history(
    session_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Get conversation history for a session.

    Args:
        session_id: The session identifier string

    Returns:
        Session details with all conversations
    """
    logger = await get_conversation_logger(session)
    history = await logger.get_session_history(session_id)

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )

    return SessionHistory(
        id=history["id"],
        session_id=history["session_id"],
        user_ip=history["user_ip"],
        created_at=history["created_at"],
        conversations=[
            ConversationItem(**conv)
            for conv in history["conversations"]
        ]
    )
