from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from core.llm.types import CompletionResponse, StreamDelta


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse: ...

    def stream(
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
        raise NotImplementedError

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "tools": True,
            "streaming": True,
            "structured_output": True,
        }
