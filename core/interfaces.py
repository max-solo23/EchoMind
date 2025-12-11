from typing import Protocol, Any, runtime_checkable


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