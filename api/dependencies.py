from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from Chat import Chat
from config import Config
from core.llm import create_llm_provider
from database import get_session
from EvaluatorAgent import EvaluatorAgent
from Me import Me
from PushOver import PushOver
from repositories.cache_repo import SQLAlchemyCacheRepository
from repositories.conversation_repo import SQLAlchemyConversationRepository
from services.cache_service import CacheService
from services.conversation_logger import ConversationLogger
from services.similarity_service import SimilarityService
from Tools import Tools


@lru_cache
def get_config() -> Config:
    """
    Get singleton Config instance.
    Uses lru_cache to ensure config is loaded once and reused.
    """
    return Config.from_env()


@lru_cache
def get_chat_service() -> Chat:
    """
    Get singleton Chat service with all dependencies wired up.

    Uses lru_cache to ensure service is created once and reused.
    Follows same dependency injection pattern as app.py.
    """
    config = get_config()
    llm_provider = create_llm_provider(config)

    # Warn about provider limitations
    if not llm_provider.capabilities.get("tools", False):
        print(
            f"Warning: {config.llm_provider} does not support tool calling. "
            "record_user_details and record_unknown_question will be disabled.",
            flush=True,
        )

    if config.use_evaluator and not llm_provider.capabilities.get("structured_output", False):
        print(
            f"Warning: {config.llm_provider} does not support structured outputs. "
            "Evaluator will use JSON fallback mode.",
            flush=True,
        )

    me = Me(config.persona_name, config.persona_file)
    evaluator: EvaluatorAgent | None = None

    if config.use_evaluator:
        evaluator = EvaluatorAgent(me, llm_provider, config.llm_model)
    pushover: PushOver | None = None

    if config.pushover_token and config.pushover_user:
        pushover = PushOver(config.pushover_token, config.pushover_user)

    tools = Tools(pushover)

    return Chat(me, llm_provider, config.llm_model, tools, evaluator)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @router.get("/example")
        async def example(session: AsyncSession = Depends(get_db_session)):
            ...

    Yields:
        AsyncSession: Database session (auto-closed after request)
    """
    config = get_config()

    if not config.database_url:
        raise RuntimeError("Database not configured")

    async with get_session(config) as session:
        yield session


async def get_conversation_logger(session: AsyncSession) -> ConversationLogger:
    """
    Create ConversationLogger with all dependencies.

    Args:
        session: Database session from get_db_session

    Returns:
        Configured ConversationLogger
    """
    conversation_repo = SQLAlchemyConversationRepository(session)
    cache_repo = SQLAlchemyCacheRepository(session)
    similarity_service = SimilarityService(threshold=0.90)
    cache_service = CacheService(cache_repo, similarity_service)

    return ConversationLogger(
        conversation_repo=conversation_repo, cache_service=cache_service, enable_caching=True
    )


def is_database_configured() -> bool:
    """Check if database is configured."""
    config = get_config()
    return config.database_url is not None
