from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI, OpenAI

from core.llm.provider import LLMProvider
from core.llm.types import CompletionMessage, CompletionResponse, StreamDelta, ToolCallDelta


class OpenAICompatibleProvider(LLMProvider):
    """
    OpenAI SDK wrapper that also works with OpenAI-compatible APIs by setting base_url.

    Examples:
    - DeepSeek (OpenAI-compatible)
    - xAI Grok (OpenAI-compatible)
    - Ollama (http://localhost:11434/v1)
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        response = self._client.chat.completions.create(model=model, messages=messages, tools=tools)
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

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        stream = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            tool_call_deltas: list[ToolCallDelta] | None = None
            if delta.tool_calls:
                tool_call_deltas = []
                for tc in delta.tool_calls:
                    tool_call_deltas.append(
                        ToolCallDelta(
                            index=tc.index,
                            id=getattr(tc, "id", None),
                            name=getattr(getattr(tc, "function", None), "name", None),
                            arguments=getattr(getattr(tc, "function", None), "arguments", None),
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
        # Only available for providers that support the OpenAI SDK's structured outputs.
        return self._client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        """OpenAI-compatible providers support all features."""
        return {
            "tools": True,
            "streaming": True,
            "structured_output": True,
        }

