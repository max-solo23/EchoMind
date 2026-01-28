from repositories.conversation_repo import SQLAlchemyConversationRepository

from .cache_service import CacheService


class ConversationLogger:
    def __init__(
        self,
        conversation_repo: SQLAlchemyConversationRepository,
        cache_service: CacheService,
        enable_caching: bool = True,
    ):
        self.conversation_repo = conversation_repo
        self.cache_service = cache_service
        self.enable_caching = enable_caching

    async def get_or_create_session(self, session_id: str, user_ip: str | None = None) -> int:
        return await self.conversation_repo.create_session(session_id, user_ip)

    async def check_cache(
        self,
        question: str,
        last_assistant_message: str | None = None,
        is_continuation: bool = False,
    ) -> str | None:
        if not self.enable_caching:
            return None

        return await self.cache_service.get_cached_answer(
            message=question,
            last_assistant_message=last_assistant_message,
            is_continuation=is_continuation,
        )

    async def log_and_cache(
        self,
        session_db_id: int,
        user_message: str,
        bot_response: str,
        tool_calls: list | None = None,
        evaluator_used: bool = False,
        evaluator_passed: bool | None = None,
        cache_response: bool = True,
        last_assistant_message: str | None = None,
        is_continuation: bool = False,
    ) -> int:
        conversation_id = await self.conversation_repo.log_conversation(
            session_db_id=session_db_id,
            user_message=user_message,
            bot_response=bot_response,
            tool_calls=tool_calls,
            evaluator_used=evaluator_used,
            evaluator_passed=evaluator_passed,
        )

        if self.enable_caching and cache_response:
            await self.cache_service.cache_answer(
                message=user_message,
                answer=bot_response,
                last_assistant_message=last_assistant_message,
                is_continuation=is_continuation,
            )

        return conversation_id

    async def get_session_history(self, session_id: str) -> dict | None:
        return await self.conversation_repo.get_session_by_id(session_id)

    async def get_cache_stats(self) -> dict:
        return await self.cache_service.get_cache_stats()

    async def clear_cache(self) -> int:
        return await self.cache_service.clear_cache()

    async def cleanup_expired_cache(self) -> int:
        return await self.cache_service.cleanup_expired()

    async def list_sessions(
        self, page: int = 1, limit: int = 20, sort_by: str = "created_at", order: str = "desc"
    ) -> dict:
        return await self.conversation_repo.list_sessions(page, limit, sort_by, order)

    async def list_cache_entries(
        self, page: int = 1, limit: int = 20, sort_by: str = "last_used", order: str = "desc"
    ) -> dict:
        return await self.cache_service.list_cache_entries(page, limit, sort_by, order)

    async def get_cache_entry(self, cache_id: int) -> dict | None:
        return await self.cache_service.get_cache_by_id(cache_id)

    async def delete_cache_entry(self, cache_id: int) -> bool:
        return await self.cache_service.delete_cache_by_id(cache_id)

    async def update_cache_entry(self, cache_id: int, variations: list[str]) -> bool:
        return await self.cache_service.update_cache_variations(cache_id, variations)

    async def search_cache(self, query: str, limit: int = 20) -> list[dict]:
        return await self.cache_service.search_cache(query, limit)

    async def delete_session(self, session_id: str) -> bool:
        return await self.conversation_repo.delete_session(session_id)

    async def clear_all_sessions(self) -> int:
        return await self.conversation_repo.clear_all_sessions()
