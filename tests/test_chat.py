import pytest

from core.chat import Chat, InvalidMessageError, SSEEvent
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
