from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import (
    get_conversation_logger,
    get_db_session,
    is_database_configured,
)
from api.middleware.auth import verify_api_key
from api.middleware.rate_limit_state import rate_limit_state


router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(verify_api_key)])


class CacheStats(BaseModel):
    total_questions: int
    total_variations: int
    avg_variations_per_question: float
    knowledge_entries: int = 0
    conversational_entries: int = 0
    expired_entries: int = 0


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
    user_ip: str | None
    created_at: datetime
    conversations: list[ConversationItem]


class DatabaseStatus(BaseModel):
    configured: bool
    connected: bool
    error: str | None = None


class AdminHealthResponse(BaseModel):
    status: str
    database: DatabaseStatus
    cache: CacheStats | None = None


class SessionSummary(BaseModel):
    id: int
    session_id: str
    user_ip: str | None
    created_at: datetime
    last_activity: datetime | None
    message_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
    total: int
    page: int
    limit: int
    total_pages: int


class CacheEntry(BaseModel):
    id: int
    cache_key: str | None = None
    question: str
    context_preview: str | None = None
    variations: list[str]
    variation_index: int
    cache_type: str = "knowledge"
    expires_at: datetime | None = None
    hit_count: int
    created_at: datetime | None
    last_used: datetime | None


class CacheEntryDetail(BaseModel):
    id: int
    cache_key: str | None = None
    question: str
    context_preview: str | None = None
    tfidf_vector: str | None
    variations: list[str]
    variation_index: int
    cache_type: str = "knowledge"
    expires_at: datetime | None = None
    hit_count: int
    created_at: datetime | None
    last_used: datetime | None


class CacheListResponse(BaseModel):
    entries: list[CacheEntry]
    total: int
    page: int
    limit: int
    total_pages: int


class CacheSearchResult(BaseModel):
    id: int
    cache_key: str | None = None
    question: str
    context_preview: str | None = None
    cache_type: str = "knowledge"
    expires_at: datetime | None = None
    hit_count: int
    last_used: datetime | None


class CacheSearchResponse(BaseModel):
    results: list[CacheSearchResult]
    total: int


class UpdateCacheRequest(BaseModel):
    variations: list[str] = Field(..., min_length=1, max_length=3)


class DeleteCacheResponse(BaseModel):
    success: bool
    deleted_id: int


class UpdateCacheResponse(BaseModel):
    success: bool
    updated_at: datetime


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


class SessionSortBy(str, Enum):
    created_at = "created_at"
    last_activity = "last_activity"


class CacheSortBy(str, Enum):
    hit_count = "hit_count"
    created_at = "created_at"
    last_used = "last_used"
    expires_at = "expires_at"
    cache_type = "cache_type"


class CleanupExpiredResponse(BaseModel):
    success: bool
    deleted_count: int


class DeleteSessionResponse(BaseModel):
    success: bool
    session_id: str


class ClearSessionsResponse(BaseModel):
    success: bool
    deleted_count: int


class RateLimitSettings(BaseModel):
    enabled: bool
    rate_per_hour: int


class UpdateRateLimitRequest(BaseModel):
    enabled: bool | None = None
    rate_per_hour: int | None = Field(None, ge=1)


def require_database():
    if not is_database_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured. Set POSTGRES_URL environment variable.",
        )


@router.get("/health", response_model=AdminHealthResponse)
async def admin_health():
    db_status = DatabaseStatus(configured=is_database_configured(), connected=False)

    cache_stats = None

    if is_database_configured():
        try:
            from api.dependencies import get_config
            from repositories.connection import get_session

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
        cache=cache_stats,
    )


@router.get("/cache/stats", response_model=CacheStats, dependencies=[Depends(require_database)])
async def get_cache_stats(session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    stats = await logger.get_cache_stats()
    return CacheStats(**stats)


@router.delete(
    "/cache", response_model=ClearCacheResponse, dependencies=[Depends(require_database)]
)
async def clear_cache(session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    deleted = await logger.clear_cache()
    return ClearCacheResponse(success=True, deleted_count=deleted)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistory,
    dependencies=[Depends(require_database)],
)
async def get_session_history(session_id: str, session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    history = await logger.get_session_history(session_id)

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found"
        )

    return SessionHistory(
        id=history["id"],
        session_id=history["session_id"],
        user_ip=history["user_ip"],
        created_at=history["created_at"],
        conversations=[ConversationItem(**conv) for conv in history["conversations"]],
    )


@router.get(
    "/sessions", response_model=SessionListResponse, dependencies=[Depends(require_database)]
)
async def list_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: SessionSortBy = Query(SessionSortBy.created_at),
    order: SortOrder = Query(SortOrder.desc),
    session: AsyncSession = Depends(get_db_session),
):
    logger = await get_conversation_logger(session)
    result = await logger.list_sessions(page, limit, sort_by.value, order.value)

    return SessionListResponse(
        sessions=[SessionSummary(**s) for s in result["sessions"]],
        total=result["total"],
        page=result["page"],
        limit=result["limit"],
        total_pages=result["total_pages"],
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    dependencies=[Depends(require_database)],
)
async def delete_session(session_id: str, session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    success = await logger.delete_session(session_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found"
        )

    return DeleteSessionResponse(success=True, session_id=session_id)


@router.delete(
    "/sessions", response_model=ClearSessionsResponse, dependencies=[Depends(require_database)]
)
async def clear_all_sessions(session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    deleted = await logger.clear_all_sessions()
    return ClearSessionsResponse(success=True, deleted_count=deleted)


@router.get(
    "/cache/entries", response_model=CacheListResponse, dependencies=[Depends(require_database)]
)
async def list_cache_entries(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: CacheSortBy = Query(CacheSortBy.last_used),
    order: SortOrder = Query(SortOrder.desc),
    session: AsyncSession = Depends(get_db_session),
):
    logger = await get_conversation_logger(session)
    result = await logger.list_cache_entries(page, limit, sort_by.value, order.value)

    return CacheListResponse(
        entries=[CacheEntry(**e) for e in result["entries"]],
        total=result["total"],
        page=result["page"],
        limit=result["limit"],
        total_pages=result["total_pages"],
    )


@router.get(
    "/cache/search", response_model=CacheSearchResponse, dependencies=[Depends(require_database)]
)
async def search_cache(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    logger = await get_conversation_logger(session)
    results = await logger.search_cache(q, limit)

    return CacheSearchResponse(
        results=[CacheSearchResult(**r) for r in results], total=len(results)
    )


@router.get(
    "/cache/{cache_id}", response_model=CacheEntryDetail, dependencies=[Depends(require_database)]
)
async def get_cache_entry(cache_id: int, session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    entry = await logger.get_cache_entry(cache_id)

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Cache entry {cache_id} not found"
        )

    return CacheEntryDetail(**entry)


@router.put(
    "/cache/{cache_id}",
    response_model=UpdateCacheResponse,
    dependencies=[Depends(require_database)],
)
async def update_cache_entry(
    cache_id: int, request: UpdateCacheRequest, session: AsyncSession = Depends(get_db_session)
):
    logger = await get_conversation_logger(session)
    success = await logger.update_cache_entry(cache_id, request.variations)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Cache entry {cache_id} not found"
        )

    return UpdateCacheResponse(success=True, updated_at=datetime.utcnow())


@router.delete(
    "/cache/{cache_id}",
    response_model=DeleteCacheResponse,
    dependencies=[Depends(require_database)],
)
async def delete_cache_entry(cache_id: int, session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    success = await logger.delete_cache_entry(cache_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Cache entry {cache_id} not found"
        )

    return DeleteCacheResponse(success=True, deleted_id=cache_id)


@router.post(
    "/cache/cleanup",
    response_model=CleanupExpiredResponse,
    dependencies=[Depends(require_database)],
)
async def cleanup_expired_cache(session: AsyncSession = Depends(get_db_session)):
    logger = await get_conversation_logger(session)
    deleted = await logger.cleanup_expired_cache()
    return CleanupExpiredResponse(success=True, deleted_count=deleted)


@router.get("/rate-limit", response_model=RateLimitSettings)
async def get_rate_limit_settings():
    return RateLimitSettings(**rate_limit_state.get_settings())


@router.post("/rate-limit", response_model=RateLimitSettings)
async def update_rate_limit_settings(request: UpdateRateLimitRequest):
    try:
        rate_limit_state.update_settings(
            enabled=request.enabled, rate_per_hour=request.rate_per_hour
        )
        return RateLimitSettings(**rate_limit_state.get_settings())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request data"
        ) from None
