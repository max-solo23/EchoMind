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

    from openai.types.chat import ChatCompletionStreamOptionsParam


logger = logging.getLogger(__name__)

INCLUDE_USAGE: ChatCompletionStreamOptionsParam = {"include_usage": True}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        self._async_client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, default_headers=default_headers
        )
        # Sent only when configured: OpenAI-compatible providers without reasoning reject it.
        # extra_body because SDK 2.4 types lack newer values such as "none" and "max".
        self._extra_body: dict[str, Any] | None = (
            {"reasoning_effort": reasoning_effort} if reasoning_effort else None
        )

    @staticmethod
    def _log_usage(model: str, usage: Any) -> None:
        if usage is None:
            return
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        logger.info(
            "LLM usage model=%s prompt=%s cached=%s cache_write=%s completion=%s reasoning=%s",
            model,
            usage.prompt_tokens,
            getattr(prompt_details, "cached_tokens", None),
            getattr(prompt_details, "cache_write_tokens", None),
            usage.completion_tokens,
            getattr(completion_details, "reasoning_tokens", None),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM call after {retry_state.attempt_number}/{3}: "
        ),
    )
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            extra_body=self._extra_body,
        )
        self._log_usage(model, getattr(response, "usage", None))
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
        reraise=True,
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
            stream_options=INCLUDE_USAGE,
            extra_body=self._extra_body,
        )
        assert hasattr(stream, "__aiter__")

        async for chunk in stream:
            # With include_usage the last chunk carries usage and no choices.
            if not chunk.choices:
                self._log_usage(model, getattr(chunk, "usage", None))
                continue
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
