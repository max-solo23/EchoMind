from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from core.llm.types import CompletionResponse, StreamDelta


@runtime_checkable
class LLMProvider(Protocol):
    """
    Dependency-inversion boundary for LLM access.

    Implementations can wrap OpenAI, OpenAI-compatible APIs (DeepSeek/Grok/Ollama),
    or other vendors (Claude/Gemini) behind a stable interface.
    """

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse: ...

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamDelta]: ...

    def parse(
        self,
        *,
        model: str,
        messages: list[dict],
        response_format: Any,
    ) -> Any:
        """
        Optional structured-output helper used by the evaluator.
        Providers that don't support this should raise NotImplementedError.
        """
        raise NotImplementedError

    @property
    def capabilities(self) -> dict[str, bool]:
        """
        Return provider capabilities.

        Default implementation assumes full support.
        Providers with limitations should override this.

        Returns:
            dict with capability flags:
            - "tools": Supports function/tool calling
            - "streaming": Supports streaming responses
            - "structured_output": Supports structured output via parse()
        """
        return {
            "tools": True,
            "streaming": True,
            "structured_output": True,
        }

