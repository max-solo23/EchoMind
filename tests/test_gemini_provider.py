import json
from contextlib import asynccontextmanager

import httpx
import pytest

from core.llm.providers.gemini import GeminiProvider


HELLO = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://gemini.test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class AsyncResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = HELLO if payload is None else payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _http_error(self.status_code)

    def json(self):
        return self._payload


class StreamResponse:
    def __init__(self, lines, status_code):
        self.lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _http_error(self.status_code)

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class AsyncClientSpy:
    def __init__(self, lines=(), status_code=200):
        self.response = AsyncResponse(status_code=status_code)
        self.lines = list(lines)
        self.status_code = status_code
        self.calls = []

    async def post(self, url, *, params, json):
        self.calls.append((url, params, json))
        return self.response

    @asynccontextmanager
    async def stream(self, method, url, *, params, json):
        self.calls.append((url, params, json))
        yield StreamResponse(self.lines, self.status_code)


@pytest.fixture
def provider():
    return GeminiProvider(api_key="test-key", base_url="https://gemini.test/")


def test_to_gemini_merges_system_parts_and_maps_roles(provider):
    system, contents = provider._to_gemini(
        [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "yo"},
            {"role": "system", "content": "B"},
        ]
    )

    assert system == {"parts": [{"text": "A\n\nB"}]}
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "yo"}]},
    ]


def test_to_gemini_without_system_prompt(provider):
    system, contents = provider._to_gemini([{"role": "user", "content": "hi"}])

    assert system is None
    assert len(contents) == 1


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, ""),
        ({"candidates": []}, ""),
        ({"candidates": [{}]}, ""),
        (
            {
                "candidates": [
                    {"content": {"parts": [{"text": "a"}, {"text": None}, {"x": 1}, {"text": "b"}]}}
                ]
            },
            "ab",
        ),
    ],
)
def test_extract_text(provider, payload, expected):
    assert provider._extract_text(payload) == expected


async def test_gemini_complete_uses_async_client(provider):
    spy = AsyncClientSpy()
    provider._async_client = spy

    response = await provider.complete(
        model="gemini-pro",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Hello"}],
    )

    assert response.message.content == "Hello"
    assert response.finish_reason is None
    url, params, body = spy.calls[0]
    assert url == "https://gemini.test/v1beta/models/gemini-pro:generateContent"
    assert params == {"key": "test-key"}
    assert body == {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "system_instruction": {"parts": [{"text": "S"}]},
    }


async def test_complete_rejects_tools(provider):
    with pytest.raises(NotImplementedError):
        await provider.complete(model="m", messages=[], tools=[{"type": "function"}])


async def test_complete_raises_on_http_error(provider):
    provider._async_client = AsyncClientSpy(status_code=500)

    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete(model="m", messages=[{"role": "user", "content": "x"}])


async def test_stream_yields_text_then_stop(provider):
    lines = [
        "",
        "not json",
        json.dumps({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
        json.dumps({"candidates": []}),
        json.dumps({"candidates": [{"content": {"parts": [{"text": "lo"}]}}]}),
    ]
    spy = AsyncClientSpy(lines=lines)
    provider._async_client = spy

    deltas = [
        d
        async for d in provider.stream(
            model="gemini-pro", messages=[{"role": "user", "content": "Hi"}]
        )
    ]

    assert [d.content for d in deltas] == ["Hel", "lo", None]
    assert deltas[-1].finish_reason == "stop"
    assert spy.calls[0][0] == "https://gemini.test/v1beta/models/gemini-pro:streamGenerateContent"


async def test_stream_raises_on_http_error(provider):
    provider._async_client = AsyncClientSpy(status_code=503)

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in provider.stream(model="m", messages=[{"role": "user", "content": "x"}]):
            pass


async def test_stream_rejects_tools(provider):
    with pytest.raises(NotImplementedError):
        async for _ in provider.stream(model="m", messages=[], tools=[{"type": "function"}]):
            pass


def test_parse_not_supported(provider):
    with pytest.raises(NotImplementedError):
        provider.parse(model="m", messages=[], response_format=dict)


def test_capabilities(provider):
    assert provider.capabilities == {"tools": False, "streaming": True, "structured_output": False}
