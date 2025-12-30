"""
Conversation Logger Service - Main orchestration for logging and caching.

This is the entry point from Chat.py for:
- Logging all conversations
- Checking cache for existing answers (with context awareness)
- Storing answers in cache (with TTL)

Design:
- Coordinates between cache service and conversation repository
- Handles session management
- Provides clean API for Chat.py integration
- Supports context-aware caching to prevent cross-conversation contamination
"""

from typing import Optional
from repositories.conversation_repo import SQLAlchemyConversationRepository
from .cache_service import CacheService


class ConversationLogger:
    """
    Main service for conversation logging and caching.

    Usage in Chat.py:
    1. Before LLM call: check cache for answer (with context)
    2. After LLM call: log conversation and cache answer (with context + TTL)
    """

    def __init__(
        self,
        conversation_repo: SQLAlchemyConversationRepository,
        cache_service: CacheService,
        enable_caching: bool = True
    ):
        """
        Initialize with dependencies.

        Args:
            conversation_repo: Repository for conversation storage
            cache_service: Service for answer caching
            enable_caching: Whether to use answer caching (default True)
        """
        self.conversation_repo = conversation_repo
        self.cache_service = cache_service
        self.enable_caching = enable_caching

    async def get_or_create_session(
        self,
        session_id: str,
        user_ip: Optional[str] = None
    ) -> int:
        """
        Get existing session or create new one.

        Args:
            session_id: Unique session identifier (from cookie/header)
            user_ip: Optional user IP address

        Returns:
            Database session ID (integer)
        """
        return await self.conversation_repo.create_session(session_id, user_ip)

    async def check_cache(
        self,
        question: str,
        last_assistant_message: Optional[str] = None,
        is_continuation: bool = False
    ) -> Optional[str]:
        """
        Check if we have a cached answer for this question with context.

        Call this BEFORE making LLM request.

        Context-aware caching:
        - Uses last_assistant_message to create unique cache keys
        - Skips cache for acknowledgements/fillers in continuations
        - Respects TTL (24h for conversational, 30d for knowledge)

        Args:
            question: User's question
            last_assistant_message: Previous assistant response (for context key)
            is_continuation: True if conversation has history (affects denylist)

        Returns:
            Cached answer if found and not expired, None otherwise
        """
        if not self.enable_caching:
            return None

        return await self.cache_service.get_cached_answer(
            message=question,
            last_assistant_message=last_assistant_message,
            is_continuation=is_continuation
        )

    async def log_and_cache(
        self,
        session_db_id: int,
        user_message: str,
        bot_response: str,
        tool_calls: Optional[list] = None,
        evaluator_used: bool = False,
        evaluator_passed: Optional[bool] = None,
        cache_response: bool = True,
        last_assistant_message: Optional[str] = None,
        is_continuation: bool = False
    ) -> int:
        """
        Log conversation and optionally cache the response with context.

        Call this AFTER getting LLM response.

        Context-aware caching:
        - Creates unique cache key from (context + question)
        - Applies appropriate TTL based on cache type
        - Skips caching for low-information inputs

        Args:
            session_db_id: Database session ID
            user_message: User's message
            bot_response: Bot's response
            tool_calls: List of tool calls made
            evaluator_used: Whether evaluator was used
            evaluator_passed: Whether evaluation passed
            cache_response: Whether to cache this response
            last_assistant_message: Previous assistant response (for context key)
            is_continuation: True if conversation has history

        Returns:
            Conversation ID
        """
        # Log the conversation (always logged regardless of cache)
        conversation_id = await self.conversation_repo.log_conversation(
            session_db_id=session_db_id,
            user_message=user_message,
            bot_response=bot_response,
            tool_calls=tool_calls,
            evaluator_used=evaluator_used,
            evaluator_passed=evaluator_passed
        )

        # Cache the response if enabled and requested
        if self.enable_caching and cache_response:
            # cache_answer handles all filtering (denylist, min tokens, etc.)
            await self.cache_service.cache_answer(
                message=user_message,
                answer=bot_response,
                last_assistant_message=last_assistant_message,
                is_continuation=is_continuation
            )

        return conversation_id

    async def get_session_history(self, session_id: str) -> Optional[dict]:
        """
        Get all conversations for a session.

        Args:
            session_id: Session identifier string

        Returns:
            Session data with conversations, or None if not found
        """
        return await self.conversation_repo.get_session_by_id(session_id)

    async def get_cache_stats(self) -> dict:
        """
        Get cache statistics including TTL breakdown.

        Returns:
            Dict with cache stats (total, by type, expired count)
        """
        return await self.cache_service.get_cache_stats()

    async def clear_cache(self) -> int:
        """
        Clear all cached answers.

        Returns:
            Number of entries deleted
        """
        return await self.cache_service.clear_cache()

    async def cleanup_expired_cache(self) -> int:
        """
        Delete all expired cache entries.

        Call periodically (e.g., on startup, via cron) to clean up old entries.

        Returns:
            Number of entries deleted
        """
        return await self.cache_service.cleanup_expired()

    # Admin methods for frontend

    async def list_sessions(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> dict:
        """List all sessions with pagination."""
        return await self.conversation_repo.list_sessions(page, limit, sort_by, order)

    async def list_cache_entries(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "last_used",
        order: str = "desc"
    ) -> dict:
        """List cache entries with pagination."""
        return await self.cache_service.list_cache_entries(page, limit, sort_by, order)

    async def get_cache_entry(self, cache_id: int) -> Optional[dict]:
        """Get single cache entry by ID."""
        return await self.cache_service.get_cache_by_id(cache_id)

    async def delete_cache_entry(self, cache_id: int) -> bool:
        """Delete single cache entry by ID."""
        return await self.cache_service.delete_cache_by_id(cache_id)

    async def update_cache_entry(self, cache_id: int, variations: list[str]) -> bool:
        """Update cache entry variations."""
        return await self.cache_service.update_cache_variations(cache_id, variations)

    async def search_cache(self, query: str, limit: int = 20) -> list[dict]:
        """Search cache entries by question text."""
        return await self.cache_service.search_cache(query, limit)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its conversations.

        Args:
            session_id: Session identifier string

        Returns:
            True if deleted, False if not found
        """
        return await self.conversation_repo.delete_session(session_id)

    async def clear_all_sessions(self) -> int:
        """
        Delete all sessions and their conversations.

        Returns:
            Number of sessions deleted
        """
        return await self.conversation_repo.clear_all_sessions()
