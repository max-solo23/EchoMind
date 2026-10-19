import json

import pytest

from core.chat import TOOL_CALL_LIMIT_MESSAGE, Chat, InvalidMessageError, SSEEvent
from core.llm.types import CompletionMessage, CompletionResponse, StreamDelta, ToolCallDelta
from core.persona import Persona


class TestMessageValidation:
    def test_valid_message(self):
        assert Chat._is_valid_message("Hello, how are you?") is True
        assert Chat._is_valid_message("What's your background?") is True
        assert Chat._is_valid_message("Tell me about yourself") is True

    def test_too_short_message(self):
        assert Chat._is_valid_message("h") is False
        assert Chat._is_valid_message("i") is False
        assert Chat._is_valid_message("") is False

    def test_whitespace_only_message(self):
        assert Chat._is_valid_message("   ") is False
        assert Chat._is_valid_message("\t\n") is False

    def test_gibberish_message(self):
        assert Chat._is_valid_message("111a222b333c444") is False
        assert Chat._is_valid_message("!@#$%^&*()") is False
        assert Chat._is_valid_message("123456789") is False

    def test_low_letter_percentage(self):
        assert Chat._is_valid_message("!!!a!!!") is False
        assert Chat._is_valid_message("123a456") is False


class TestChatBasics:
    @pytest.fixture
    def mock_tools(self):
        class MockTools:
            tools = []

        return MockTools()

    def test_chat_initialization(self, mock_llm_provider, temp_persona_file, mock_tools):
        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        assert chat.llm == mock_llm_provider
        assert chat.llm_model == "test-model"
        assert chat.persona == persona

    async def test_chat_basic_response(self, mock_llm_provider, temp_persona_file, mock_tools):
        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        response = await chat.chat("Hello", [])

        assert response == "Mock LLM response"

    async def test_chat_with_history(
        self, mock_llm_provider, temp_persona_file, sample_chat_history, mock_tools
    ):
        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        response = await chat.chat("Tell me more", sample_chat_history)

        assert response == "Mock LLM response"

    async def test_chat_rejects_invalid_message(self, mock_llm_provider, temp_persona_file, mock_tools):
        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        with pytest.raises(InvalidMessageError) as exc_info:
            await chat.chat("h", [])

        assert "invalid" in str(exc_info.value).lower()

    async def test_chat_rejects_gibberish(self, mock_llm_provider, temp_persona_file, mock_tools):
        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        with pytest.raises(InvalidMessageError):
            await chat.chat("!@#$%^&*()", [])

    async def test_chat_handles_llm_error(self, temp_persona_file, mock_tools):
        class FailingLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            async def complete(self, **kwargs):
                raise Exception("LLM failed")

        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=FailingLLM(), llm_model="test", llm_tools=mock_tools)

        response = await chat.chat("Hello", [])

        assert "unexpected" in response.lower()

    async def test_chat_handles_api_timeout_error(self, temp_persona_file, mock_tools):
        from openai import APITimeoutError

        class TimeoutLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            async def complete(self, **kwargs):
                raise APITimeoutError(request=None)

        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=TimeoutLLM(), llm_model="test", llm_tools=mock_tools)

        response = await chat.chat("Hello", [])

        assert "longer than expected" in response.lower()

    def test_get_tools_returns_tools_when_supported(self, mock_llm_provider, temp_persona_file):
        class ToolsWithList:
            tools = [{"type": "function", "function": {"name": "test"}}]

        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=mock_llm_provider, llm_model="test", llm_tools=ToolsWithList())

        assert chat._get_tools() is not None
        assert len(chat._get_tools()) == 1

    def test_get_tools_returns_none_when_not_supported(self, temp_persona_file):
        class NoToolsLLM:
            @property
            def capabilities(self):
                return {"tools": False}

        class ToolsWithList:
            tools = [{"type": "function", "function": {"name": "test"}}]

        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=NoToolsLLM(), llm_model="test", llm_tools=ToolsWithList())

        assert chat._get_tools() is None

    async def test_chat_handles_connection_error(self, temp_persona_file, mock_tools):
        from openai import APIConnectionError

        class ConnectionLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            async def complete(self, **kwargs):
                raise APIConnectionError(request=None)

        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=ConnectionLLM(), llm_model="test", llm_tools=mock_tools)

        response = await chat.chat("Hello", [])

        assert "trouble connecting" in response.lower()

    async def test_chat_stops_after_tool_call_limit(self, temp_persona_file):
        class ToolLoopLLM:
            def __init__(self):
                self.calls = 0

            @property
            def capabilities(self):
                return {"tools": True}

            async def complete(self, **kwargs):
                self.calls += 1
                return CompletionResponse(
                    finish_reason="tool_calls",
                    message=CompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "test_tool", "arguments": "{}"},
                            }
                        ],
                    ),
                )

        class LoopTools:
            tools = [{"type": "function", "function": {"name": "test_tool"}}]

            def handle_tool_call(self, tool_calls):
                return [{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]

        llm = ToolLoopLLM()
        persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(persona=persona, llm=llm, llm_model="test", llm_tools=LoopTools())

        response = await chat.chat("Hello", [])

        assert response == TOOL_CALL_LIMIT_MESSAGE
        assert llm.calls == 5


class TestSSEEvent:
    def test_encode_with_delta(self):
        event = SSEEvent(delta="Hello")
        encoded = event.encode()

        assert b"data:" in encoded
        assert b"Hello" in encoded

    def test_encode_with_metadata(self):
        event = SSEEvent(metadata={"done": True})
        encoded = event.encode()

        assert b"data:" in encoded
        assert b"done" in encoded


TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "test_tool", "arguments": '{"a": 1}'},
}

TOOL_CALL_ROUND = [
    StreamDelta(
        tool_calls=[ToolCallDelta(index=0, id="call_1", name="test_tool", arguments='{"a"')]
    ),
    StreamDelta(tool_calls=[ToolCallDelta(index=0, arguments=": 1}")]),
    StreamDelta(finish_reason="tool_calls"),
]
ANSWER_ROUND = [
    StreamDelta(content="Do"),
    StreamDelta(content="ne"),
    StreamDelta(finish_reason="stop"),
]


def _decode_events(chunks: list[bytes]) -> list[dict]:
    return [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith(b"data: ")]


def _make_chat(persona_file, llm, tools) -> Chat:
    persona = Persona(name="Test User", persona_yaml_file=persona_file)
    return Chat(persona=persona, llm=llm, llm_model="test", llm_tools=tools)


class RecordingTools:
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.received: list = []

    def handle_tool_call(self, tool_calls):
        self.received.append(tool_calls)
        if self.fail:
            raise RuntimeError("tool exploded")
        return [{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]


class ScriptedStreamLLM:
    """Plays back one list of deltas per LLM round and records the messages it received."""

    def __init__(self, rounds: list[list[StreamDelta]]):
        self.rounds = rounds
        self.seen_messages: list[list[dict]] = []

    @property
    def capabilities(self):
        return {"tools": True, "streaming": True}

    async def stream(self, *, model, messages, tools=None):
        self.seen_messages.append([dict(m) for m in messages])
        for delta in self.rounds[len(self.seen_messages) - 1]:
            yield delta


class TestChatToolCalls:
    async def test_tool_result_is_fed_back_to_llm(self, temp_persona_file):
        class ToolThenAnswerLLM:
            def __init__(self):
                self.seen_messages = []

            @property
            def capabilities(self):
                return {"tools": True}

            async def complete(self, *, model, messages, tools=None):
                self.seen_messages.append(list(messages))
                if len(self.seen_messages) == 1:
                    return CompletionResponse(
                        finish_reason="tool_calls",
                        message=CompletionMessage(
                            role="assistant", content=None, tool_calls=[TOOL_CALL]
                        ),
                    )
                return CompletionResponse(
                    finish_reason="stop",
                    message=CompletionMessage(role="assistant", content="Done", tool_calls=None),
                )

        llm = ToolThenAnswerLLM()
        tools = RecordingTools()
        chat = _make_chat(temp_persona_file, llm, tools)

        response = await chat.chat("Hello", [])

        assert response == "Done"
        assert tools.received == [[TOOL_CALL]]
        second_round = llm.seen_messages[1]
        assert second_round[-2] == {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
        assert second_round[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}

    async def test_empty_content_becomes_empty_string(self, temp_persona_file):
        class EmptyLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            async def complete(self, **kwargs):
                return CompletionResponse(
                    finish_reason="stop", message=CompletionMessage(role="assistant", content=None)
                )

        chat = _make_chat(temp_persona_file, EmptyLLM(), RecordingTools())

        assert await chat.chat("Hello", []) == ""


class TestChatStream:
    async def test_streams_only_sse_data_events(self, temp_persona_file):
        llm = ScriptedStreamLLM([ANSWER_ROUND])
        chat = _make_chat(temp_persona_file, llm, RecordingTools())

        chunks = [chunk async for chunk in chat.chat_stream("Hello", [])]

        assert all(chunk.startswith(b"data: ") for chunk in chunks)
        assert _decode_events(chunks) == [
            {"delta": "Do", "metadata": None},
            {"delta": "ne", "metadata": None},
            {"delta": None, "metadata": {"done": True}},
        ]

    async def test_rejects_invalid_message_before_streaming(self, temp_persona_file):
        llm = ScriptedStreamLLM([ANSWER_ROUND])
        chat = _make_chat(temp_persona_file, llm, RecordingTools())

        with pytest.raises(InvalidMessageError):
            async for _ in chat.chat_stream("h", []):
                pass

        assert llm.seen_messages == []

    async def test_assembles_chunked_tool_call_then_streams_answer(self, temp_persona_file):
        llm = ScriptedStreamLLM([TOOL_CALL_ROUND, ANSWER_ROUND])
        tools = RecordingTools()
        chat = _make_chat(temp_persona_file, llm, tools)

        events = _decode_events([chunk async for chunk in chat.chat_stream("Hello", [])])

        assert [e["metadata"] for e in events if e["metadata"]] == [
            {"tool_call": "test_tool", "status": "executing"},
            {"tool_call": "test_tool", "status": "success"},
            {"done": True},
        ]
        assert "".join(e["delta"] for e in events if e["delta"]) == "Done"

        (call,) = tools.received[0]
        assert call.id == "call_1"
        assert (call.function.name, call.function.arguments) == ("test_tool", '{"a": 1}')

        second_round = llm.seen_messages[1]
        assert second_round[-2]["role"] == "assistant"
        assert second_round[-2]["tool_calls"] == [TOOL_CALL]
        assert second_round[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}

    async def test_accumulates_tool_calls_by_index_in_any_order(self, temp_persona_file):
        round_one = [
            StreamDelta(
                tool_calls=[ToolCallDelta(index=1, id="call_2", name="second", arguments="{}")]
            ),
            StreamDelta(
                tool_calls=[ToolCallDelta(index=0, id="call_1", name="test_tool", arguments="{}")]
            ),
            StreamDelta(finish_reason="tool_calls"),
        ]
        llm = ScriptedStreamLLM([round_one, ANSWER_ROUND])
        tools = RecordingTools()
        chat = _make_chat(temp_persona_file, llm, tools)

        async for _ in chat.chat_stream("Hello", []):
            pass

        assert [calls[0].function.name for calls in tools.received] == ["test_tool", "second"]

    async def test_tool_failure_stops_round_and_still_finishes(self, temp_persona_file):
        llm = ScriptedStreamLLM([TOOL_CALL_ROUND, ANSWER_ROUND])
        chat = _make_chat(temp_persona_file, llm, RecordingTools(fail=True))

        events = _decode_events([chunk async for chunk in chat.chat_stream("Hello", [])])

        assert [e["metadata"] for e in events] == [
            {"tool_call": "test_tool", "status": "executing"},
            {"tool_call": "test_tool", "status": "failed", "error": "Tool execution failed"},
            {"done": True},
        ]
        assert len(llm.seen_messages) == 1

    async def test_stops_after_tool_call_round_limit(self, temp_persona_file):
        llm = ScriptedStreamLLM([TOOL_CALL_ROUND] * 5)
        chat = _make_chat(temp_persona_file, llm, RecordingTools())

        events = _decode_events([chunk async for chunk in chat.chat_stream("Hello", [])])

        assert len(llm.seen_messages) == 5
        assert events[-2]["metadata"] == {
            "error": TOOL_CALL_LIMIT_MESSAGE,
            "code": "tool_call_limit",
        }
        assert events[-1]["metadata"] == {"done": True}

    async def test_llm_error_becomes_error_event_without_done(self, temp_persona_file):
        from openai import APITimeoutError

        class FailingStreamLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            async def stream(self, **kwargs):
                raise APITimeoutError(request=None)
                yield  # makes this an async generator

        chat = _make_chat(temp_persona_file, FailingStreamLLM(), RecordingTools())

        events = _decode_events([chunk async for chunk in chat.chat_stream("Hello", [])])

        assert events == [
            {
                "delta": None,
                "metadata": {
                    "error": "I'm taking longer than expected to respond. Please try again.",
                    "code": "api_timeout",
                },
            }
        ]
