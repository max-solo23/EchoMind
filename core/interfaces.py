from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """
    Interface for any LLM provider (OpenAI, Anthropic, Google, Deepseek, local models)

    This protocol defines the contract that any LLM provider must fulfill
    to work with our Chat and Evaluator services.
    """

    class ChatCompletions:
        """Nested class for chat completion methods."""

        def create(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
            **kwargs: Any
        ) -> Any:
            """Create a chat completion."""
            ...

        def parse(
            self,
            model: str,
            messages: list[dict],
            response_format: Any,
            **kwargs: Any
        ) -> Any:
            """Create a chat completion with structured output."""
            ...

    class Chat:
        completions: "LLMProvider.ChatCompletions"

    @property
    def chat(self) -> "LLMProvider.Chat":
        """Access to chat completions."""
        ...


@runtime_checkable
class NotificationProvider(Protocol):
    """
    Interface for notification services (Pushover, email, telegram etc.)
    This protocol defines the contract for sending notifications.
    """

    def push(self, text: str) -> None:
        """Send notification with the given text"""
        ...


@runtime_checkable
class ConversationRepository(Protocol):
    """
    Interface for conversation data access.

    Why Protocol?
    - Allows swapping implementations (SQLAlchemy, MogoDB, etc.)
    - Enables testing with mock repositories
    - Follows Dependency Inversion Principle
    """

    async def create_session(self, session_id: str, user_ip: str | None) -> int:
        """Create new session, return session DB ID."""
        ...

    async def log_conversation(
            self,
            session_db_id: int,
            user_message: str,
            bot_response: str,
            tool_calls: list | None = None,
            evaluator_used: bool = False,
            evaluator_passed: bool | None = None
    ) -> int:
        """Log a conversation, return conversation ID."""
        ...

    async def get_session_by_id(self, session_id: str) -> dict | None:
        """Get session by session_id string."""
        ...


@runtime_checkable
class CacheRepository(Protocol):
    """Interface for cache data access."""

    async def get_cache_by_question(self, question: str) -> dict | None:
        """Get cached answer for exact question match."""
        ...

    async def create_cache(self, question: str, tfidf_vector: str, answer: str) -> int:
        """Create new cache entry with first variation."""
        ...

    async def add_variation(self, cache_id: int, answer: str) -> None:
        """Add answer variation to existing cache (max 3)."""
        ...

    async def get_next_variation(self, cache_id: int) -> str:
        """Get next answer variation and increment rotation index."""
        ...

    async def clear_all_cache(self) -> int:
        """Clear all cache answers. Returns count of deleted rows."""
        ...
