"""
Conversation Logger Service - Main orchestration for logging and caching.

This is the entry point from Chat.py for:
- Logging all conversations
- Checking cache for existing answers
- Storing answers in cache

Design:
- Coordinates between cache service and conversation repository
- Handles session management
- Provides clean API for Chat.py integration
"""

from typing import Optional
from repositories.conversation_repo import SQLAlchemyConversationRepository
from .cache_service import CacheService


class ConversationLogger:
    """
    Main service for conversation logging and caching.

    Usage in Chat.py:
    1. Before LLM call: check cache for answer
    2. After LLM call: log conversation and cache answer
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

    async def check_cache(self, question: str) -> Optional[str]:
        """
        Check if we have a cached answer for this question.

        Call this BEFORE making LLM request.

        Args:
            question: User's question

        Returns:
            Cached answer if found, None otherwise
        """
        if not self.enable_caching:
            return None

        return await self.cache_service.get_cached_answer(question)

    async def log_and_cache(
        self,
        session_db_id: int,
        user_message: str,
        bot_response: str,
        tool_calls: Optional[list] = None,
        evaluator_used: bool = False,
        evaluator_passed: Optional[bool] = None,
        cache_response: bool = True
    ) -> int:
        """
        Log conversation and optionally cache the response.

        Call this AFTER getting LLM response.

        Args:
            session_db_id: Database session ID
            user_message: User's message
            bot_response: Bot's response
            tool_calls: List of tool calls made
            evaluator_used: Whether evaluator was used
            evaluator_passed: Whether evaluation passed
            cache_response: Whether to cache this response

        Returns:
            Conversation ID
        """
        # Log the conversation
        conversation_id = await self.conversation_repo.log_conversation(
            session_db_id=session_db_id,
            user_message=user_message,
            bot_response=bot_response,
            tool_calls=tool_calls,
            evaluator_used=evaluator_used,
            evaluator_passed=evaluator_passed
        )

        # Cache the response if enabled
        if self.enable_caching and cache_response:
            # Only cache if no similar question exists
            should_cache = await self.cache_service.should_cache(user_message)
            if should_cache:
                await self.cache_service.cache_answer(user_message, bot_response)

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
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return await self.cache_service.get_cache_stats()

    async def clear_cache(self) -> int:
        """
        Clear all cached answers.

        Returns:
            Number of entries deleted
        """
        return await self.cache_service.clear_cache()
