import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.llm.providers.openai_responses import (
    OpenAIResponsesProvider,
    ResponsesStreamError,
    to_responses_input,
    to_responses_tools,
)


class FakeStream:
    def __init__(self, events):
        self._events = events

    async def __aiter__(self):
        for event in self._events:
            yield event


def ev(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


USAGE = SimpleNamespace(
    input_tokens=3400,
    output_tokens=20,
    input_tokens_details=SimpleNamespace(cached_tokens=3388),
    output_tokens_details=SimpleNamespace(reasoning_tokens=5),
)


def make_provider(effort="low"):
    provider = OpenAIResponsesProvider(api_key="k", reasoning_effort=lambda: effort)
    provider._async_client.responses.create = AsyncMock()  # type: ignore[method-assign]
    return provider


def test_history_with_tool_round_converts_to_responses_items():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "record_unknown_question", "arguments": '{"question": "q"}'},
    }
    messages: list[dict] = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "favourite team?"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "content": '{"recorded": "ok"}', "tool_call_id": "call_1"},
    ]

    instructions, items = to_responses_input(messages)

    assert instructions == "persona"
    assert items == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "favourite team?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "record_unknown_question",
            "arguments": '{"question": "q"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": '{"recorded": "ok"}'},
    ]


def test_tool_calls_as_objects_convert_like_dicts():
    tool_call = SimpleNamespace(id="c", function=SimpleNamespace(name="n", arguments="{}"))

    _, items = to_responses_input(
        [{"role": "assistant", "content": "checking", "tool_calls": [tool_call]}]
    )

    assert items == [
        {"role": "assistant", "content": "checking"},
        {"type": "function_call", "call_id": "c", "name": "n", "arguments": "{}"},
    ]


def test_tools_are_flattened_and_not_strict():
    tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]

    assert to_responses_tools(tools) == [
        {"type": "function", "name": "t", "parameters": {"type": "object"}, "strict": False}
    ]
    assert to_responses_tools(None) is None


async def test_request_reads_effort_per_call_and_omits_unset_fields():
    effort = {"value": "none"}
    provider = OpenAIResponsesProvider(api_key="k", reasoning_effort=lambda: effort["value"])
    create = AsyncMock(return_value=SimpleNamespace(output=[], output_text="hi", usage=USAGE))
    provider._async_client.responses.create = create  # type: ignore[method-assign]

    await provider.complete(model="m", messages=[{"role": "user", "content": "q"}])
    effort["value"] = "high"
    await provider.complete(model="m", messages=[{"role": "user", "content": "q"}])

    first, second = (call.kwargs for call in create.call_args_list)
    assert first["extra_body"] == {"reasoning": {"effort": "none"}}
    assert second["extra_body"] == {"reasoning": {"effort": "high"}}
    assert first["store"] is False
    assert "tools" not in first
    assert "instructions" not in first


async def test_complete_maps_function_calls_to_chat_completions_shape():
    provider = make_provider()
    output = [
        ev("reasoning"),
        SimpleNamespace(
            type="function_call", call_id="c1", name="record_unknown_question", arguments="{}"
        ),
    ]
    provider._async_client.responses.create.return_value = SimpleNamespace(
        output=output, output_text="", usage=USAGE
    )

    result = await provider.complete(model="m", messages=[], tools=[])

    assert result.finish_reason == "tool_calls"
    assert result.message.content is None
    (call,) = result.message.tool_calls
    assert (call.id, call.function.name, call.function.arguments) == (
        "c1",
        "record_unknown_question",
        "{}",
    )


async def test_stream_maps_text_tool_calls_and_finish(caplog):
    provider = make_provider()
    function_call = SimpleNamespace(type="function_call", call_id="c1", name="record_user_details")
    provider._async_client.responses.create.return_value = FakeStream(
        [
            ev("response.created"),
            ev("response.output_item.added", output_index=0, item=ev("reasoning")),
            ev("response.output_item.added", output_index=1, item=function_call),
            ev("response.function_call_arguments.delta", output_index=1, delta='{"email":'),
            ev("response.function_call_arguments.delta", output_index=1, delta=' "a@b.c"}'),
            ev("response.completed", response=SimpleNamespace(usage=USAGE)),
        ]
    )

    with caplog.at_level("INFO"):
        deltas = [d async for d in provider.stream(model="m", messages=[])]

    calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert (calls[0].index, calls[0].id, calls[0].name) == (0, "c1", "record_user_details")
    assert {c.index for c in calls} == {0}
    assert json.loads("".join(c.arguments or "" for c in calls)) == {"email": "a@b.c"}
    assert deltas[-1].finish_reason == "tool_calls"
    assert "prompt=3400 cached=3388 completion=20 reasoning=5" in caplog.text


async def test_stream_text_only_finishes_with_stop():
    provider = make_provider()
    provider._async_client.responses.create.return_value = FakeStream(
        [
            ev("response.output_text.delta", delta="Hel"),
            ev("response.output_text.delta", delta="lo"),
            ev("response.completed", response=SimpleNamespace(usage=None)),
        ]
    )

    deltas = [d async for d in provider.stream(model="m", messages=[])]

    assert "".join(d.content or "" for d in deltas) == "Hello"
    assert deltas[-1].finish_reason == "stop"


async def test_stream_incomplete_finishes_with_length():
    provider = make_provider()
    provider._async_client.responses.create.return_value = FakeStream(
        [
            ev("response.output_text.delta", delta="cut"),
            ev("response.incomplete", response=SimpleNamespace(usage=None)),
        ]
    )

    deltas = [d async for d in provider.stream(model="m", messages=[])]

    assert deltas[-1].finish_reason == "length"


async def test_stream_failure_raises():
    provider = make_provider()
    provider._async_client.responses.create.return_value = FakeStream(
        [ev("response.failed", response="boom")]
    )

    with pytest.raises(ResponsesStreamError):
        [d async for d in provider.stream(model="m", messages=[])]
