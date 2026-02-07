from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.llm.provider import LLMProvider
from core.llm.types import CompletionMessage, CompletionResponse, StreamDelta, ToolCallDelta


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)

class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        self._async_client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, default_headers=default_headers
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM call after {retry_state.attempt_number}/{3}: "
        ),
    )
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )
        choice = response.choices[0]
        msg = choice.message
        return CompletionResponse(
            finish_reason=choice.finish_reason,
            message=CompletionMessage(
                role=msg.role,
                content=msg.content,
                tool_calls=getattr(msg, "tool_calls", None),
            ),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM call after {retry_state.attempt_number}/{3}: "
        ),
    )
    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        stream = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            stream=True,
        )
        assert hasattr(stream, "__aiter__")

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            tool_call_deltas: list[ToolCallDelta] | None = None
            if delta.tool_calls:
                tool_call_deltas = []
                for tc in delta.tool_calls:
                    func = getattr(tc, "function", None)
                    tool_call_deltas.append(
                        ToolCallDelta(
                            index=tc.index,
                            id=getattr(tc, "id", None),
                            name=getattr(func, "name", None) if func else None,
                            arguments=getattr(func, "arguments", None) if func else None,
                        )
                    )

            yield StreamDelta(
                content=delta.content,
                tool_calls=tool_call_deltas,
                finish_reason=choice.finish_reason,
            )

    def parse(
        self,
        *,
        model: str,
        messages: list[dict],
        response_format: Any,
    ) -> Any:
        return self._client.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format=response_format,
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "tools": True,
            "streaming": True,
            "structured_output": True,
        }
