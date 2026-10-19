from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import RateLimitError
from tenacity import wait_none

from core.llm.providers.openai_compatible import OpenAICompatibleProvider


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("Rate limit exceeded", response=response, body=None)


def _completion(content="Hello!", tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)])


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def provider():
    # tenacity copies the retry policy per call, so swapping `wait` here skips the real 1s/2s backoff
    retry_policy = OpenAICompatibleProvider.complete.retry
    original_wait = retry_policy.wait
    retry_policy.wait = wait_none()
    yield OpenAICompatibleProvider(api_key="test-key")
    retry_policy.wait = original_wait


async def test_retry_on_rate_limit(provider):
    provider._async_client.chat.completions.create = AsyncMock(
        side_effect=[_rate_limit_error(), _rate_limit_error(), _completion()]
    )

    result = await provider.complete(
        model="gpt-5.2-nano",
        messages=[{"role": "user", "content": "test"}],
    )

    assert result.message.content == "Hello!"
    assert result.finish_reason == "stop"
    assert provider._async_client.chat.completions.create.call_count == 3


async def test_gives_up_after_max_retries(provider):
    provider._async_client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())

    with pytest.raises(RateLimitError):
        await provider.complete(
            model="gpt-5.2-nano",
            messages=[{"role": "user", "content": "test"}],
        )

    assert provider._async_client.chat.completions.create.call_count == 3


async def test_complete_passes_tools_and_returns_tool_calls(provider):
    tool_calls = [SimpleNamespace(id="call_1")]
    create = AsyncMock(return_value=_completion(content=None, tool_calls=tool_calls))
    provider._async_client.chat.completions.create = create
    tools = [{"type": "function", "function": {"name": "t"}}]

    result = await provider.complete(model="m", messages=[], tools=tools)

    assert result.message.content is None
    assert result.message.tool_calls is tool_calls
    assert create.await_args.kwargs["tools"] is tools


async def test_stream_maps_content_and_tool_call_deltas(provider):
    function = SimpleNamespace(name="t", arguments='{"a"')
    chunks = [
        _chunk(content="Hel"),
        _chunk(tool_calls=[SimpleNamespace(index=0, id="call_1", function=function)]),
        _chunk(tool_calls=[SimpleNamespace(index=0, id=None, function=None)]),
        _chunk(finish_reason="tool_calls"),
    ]
    create = AsyncMock(return_value=FakeStream(chunks))
    provider._async_client.chat.completions.create = create

    deltas = [
        d async for d in provider.stream(model="m", messages=[{"role": "user", "content": "hi"}])
    ]

    assert create.await_args.kwargs["stream"] is True
    assert (deltas[0].content, deltas[0].tool_calls) == ("Hel", None)
    (first,) = deltas[1].tool_calls
    assert (first.index, first.id, first.name, first.arguments) == (0, "call_1", "t", '{"a"')
    (second,) = deltas[2].tool_calls
    assert (second.name, second.arguments) == (None, None)
    assert deltas[3].finish_reason == "tool_calls"


def test_parse_delegates_to_sync_client(provider):
    provider._client.chat.completions.parse = MagicMock(return_value="parsed")

    assert provider.parse(model="m", messages=[], response_format=dict) == "parsed"
    provider._client.chat.completions.parse.assert_called_once_with(
        model="m", messages=[], response_format=dict
    )


def test_capabilities(provider):
    assert provider.capabilities == {"tools": True, "streaming": True, "structured_output": True}
