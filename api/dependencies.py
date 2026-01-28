from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from config import Config
from core.chat import Chat
from core.llm import create_llm_provider
from core.persona import Persona
from repositories.cache_repo import SQLAlchemyCacheRepository
from repositories.connection import get_session
from repositories.conversation_repo import SQLAlchemyConversationRepository
from services.cache_service import CacheService
from services.conversation_logger import ConversationLogger
from services.push_over import PushOver
from services.similarity_service import SimilarityService
from tools.llm_tools import Tools


@lru_cache
def get_config() -> Config:
    return Config.from_env()

@lru_cache
def get_persona() -> Persona:
    config = get_config()
    return Persona(config.persona_name, config.persona_file)

@lru_cache
def get_chat_service() -> Chat:
    config = get_config()
    llm_provider = create_llm_provider(config)

    if not llm_provider.capabilities.get("tools", False):
        print(
            f"Warning: {config.llm_provider} does not support tool calling. "
            "record_user_details and record_unknown_question will be disabled.",
            flush=True,
        )

    persona = get_persona()
    pushover: PushOver | None = None

    if config.pushover_token and config.pushover_user:
        pushover = PushOver(config.pushover_token, config.pushover_user)

    tools = Tools(pushover)

    return Chat(persona, llm_provider, config.llm_model, tools)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    config = get_config()

    if not config.database_url:
        raise RuntimeError("Database not configured")

    async with get_session(config) as session:
        yield session


async def get_conversation_logger(session: AsyncSession) -> ConversationLogger:
    conversation_repo = SQLAlchemyConversationRepository(session)
    cache_repo = SQLAlchemyCacheRepository(session)
    similarity_service = SimilarityService(threshold=0.80)
    persona_hash = get_persona().content_hash()
    cache_service = CacheService(cache_repo, similarity_service, persona_hash)

    return ConversationLogger(
        conversation_repo=conversation_repo, cache_service=cache_service, enable_caching=True
    )


def is_database_configured() -> bool:
    config = get_config()
    return config.database_url is not None
