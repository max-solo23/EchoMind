from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    class ChatCompletions:
        def create(
            self, model: str, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any
        ) -> Any: ...

        def parse(
            self, model: str, messages: list[dict], response_format: Any, **kwargs: Any
        ) -> Any: ...

    class Chat:
        completions: "LLMProvider.ChatCompletions"

    @property
    def chat(self) -> "LLMProvider.Chat": ...


@runtime_checkable
class NotificationProvider(Protocol):
    def push(self, text: str) -> None: ...


@runtime_checkable
class ConversationRepository(Protocol):
    async def create_session(self, session_id: str, user_ip: str | None) -> int: ...

    async def log_conversation(
        self,
        session_db_id: int,
        user_message: str,
        bot_response: str,
        tool_calls: list | None = None,
        evaluator_used: bool = False,
        evaluator_passed: bool | None = None,
    ) -> int: ...

    async def get_session_by_id(self, session_id: str) -> dict | None: ...


@runtime_checkable
class CacheRepository(Protocol):
    async def get_cache_by_question(self, question: str) -> dict | None: ...

    async def create_cache(self, question: str, tfidf_vector: str, answer: str) -> int: ...

    async def add_variation(self, cache_id: int, answer: str) -> None: ...

    async def get_next_variation(self, cache_id: int) -> str: ...

    async def clear_all_cache(self) -> int: ...
