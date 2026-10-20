from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.llm.providers.openai_compatible import OpenAICompatibleProvider
from core.llm.types import CompletionMessage, CompletionResponse, StreamDelta, ToolCallDelta


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


logger = logging.getLogger(__name__)


class ResponsesStreamError(Exception):
    """The stream ended with response.failed or an error event."""


def _field(obj: Any, name: str) -> Any:
    # Tool calls arrive as dicts from the streaming loop and as objects from complete().
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Chat Completions history -> (instructions, Responses input items).

    Function calls are sent without their item id, so no reasoning item has to accompany them.
    """
    instructions: list[str] = []
    items: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            instructions.append(m["content"])
        elif role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]}
            )
        elif role == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m["tool_calls"]:
                function = _field(tc, "function")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": _field(tc, "id"),
                        "name": _field(function, "name"),
                        "arguments": _field(function, "arguments"),
                    }
                )
        else:
            items.append({"role": role, "content": m["content"]})
    return ("\n\n".join(instructions) or None), items


def to_responses_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    # strict=False keeps the non-strict schema behaviour the Chat Completions tools had.
    return [{"type": "function", **t["function"], "strict": False} for t in tools]


def _log_usage(model: str, usage: Any) -> None:
    if usage is None:
        return
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    logger.info(
        "LLM usage model=%s prompt=%s cached=%s completion=%s reasoning=%s",
        model,
        usage.input_tokens,
        getattr(input_details, "cached_tokens", None),
        usage.output_tokens,
        getattr(output_details, "reasoning_tokens", None),
    )


_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying LLM call after {retry_state.attempt_number}/{3}: "
    ),
)


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    """OpenAI via /v1/responses: the only endpoint where gpt-6 models accept function tools
    together with a reasoning effort other than "none". Speaks Chat Completions shapes to Chat."""

    def __init__(
        self,
        *,
        api_key: str,
        reasoning_effort: Callable[[], str | None] = lambda: None,
    ) -> None:
        super().__init__(api_key=api_key)
        # Read on every call, so an admin change applies without a restart.
        self._reasoning_effort = reasoning_effort

    def _request(self, model: str, messages: list[dict], tools: list[dict] | None) -> dict:
        instructions, items = to_responses_input(messages)
        effort = self._reasoning_effort()
        request = {
            "model": model,
            "instructions": instructions,
            "input": items,
            "tools": to_responses_tools(tools),
            "store": False,
            # extra_body because SDK 2.4 types lack newer efforts such as "none" and "max".
            "extra_body": {"reasoning": {"effort": effort}} if effort else None,
        }
        # The SDK would send None as JSON null; leaving the key out means "not set".
        return {key: value for key, value in request.items() if value is not None}

    @_RETRY
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        response = await self._async_client.responses.create(
            **self._request(model, messages, tools)
        )
        _log_usage(model, response.usage)
        tool_calls = [
            SimpleNamespace(
                id=item.call_id,
                type="function",
                function=SimpleNamespace(name=item.name, arguments=item.arguments),
            )
            for item in response.output
            if item.type == "function_call"
        ]
        return CompletionResponse(
            finish_reason="tool_calls" if tool_calls else "stop",
            message=CompletionMessage(
                role="assistant",
                content=response.output_text or None,
                tool_calls=tool_calls or None,
            ),
        )

    @_RETRY
    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        stream = await self._async_client.responses.create(
            **self._request(model, messages, tools), stream=True
        )
        tool_index: dict[int, int] = {}  # Responses output_index -> Chat Completions tool index

        async for event in stream:
            kind = event.type
            if kind == "response.output_text.delta":
                yield StreamDelta(content=event.delta)
            elif kind == "response.output_item.added" and event.item.type == "function_call":
                tool_index[event.output_index] = len(tool_index)
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=tool_index[event.output_index],
                            id=event.item.call_id,
                            name=event.item.name,
                        )
                    ]
                )
            elif kind == "response.function_call_arguments.delta":
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(index=tool_index[event.output_index], arguments=event.delta)
                    ]
                )
            elif kind in ("response.completed", "response.incomplete"):
                _log_usage(model, event.response.usage)
                cut_short = kind == "response.incomplete"
                yield StreamDelta(
                    finish_reason="tool_calls" if tool_index else "length" if cut_short else "stop"
                )
            elif kind in ("response.failed", "error"):
                raise ResponsesStreamError(str(getattr(event, "response", event)))
