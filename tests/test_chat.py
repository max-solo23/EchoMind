"""
Tests for Chat.py - Main conversation handler

Key concepts tested:
1. Using mock_llm_provider fixture for LLM interactions
2. Message validation logic
3. Chat flow with history
4. Exception handling
"""

import pytest

from core.chat import Chat, InvalidMessageError, SSEEvent
from core.persona import Me


class TestMessageValidation:
    """Test the message validation logic."""

    def test_valid_message(self):
        """Test that valid messages pass validation."""
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
        """Test that keyboard mashing is rejected."""
        assert Chat._is_valid_message("111a222b333c444") is False  # Only 3 letters out of 15 = 20%
        assert Chat._is_valid_message("!@#$%^&*()") is False
        assert Chat._is_valid_message("123456789") is False

    def test_low_letter_percentage(self):
        """Test that messages with < 30% letters are rejected."""
        assert Chat._is_valid_message("!!!a!!!") is False
        assert Chat._is_valid_message("123a456") is False


class TestChatBasics:
    """Test basic chat functionality."""

    @pytest.fixture
    def mock_tools(self):
        """Create a mock Tools object."""

        class MockTools:
            tools = []  # Empty tools list

        return MockTools()

    def test_chat_initialization(self, mock_llm_provider, temp_persona_file, mock_tools):
        """Test that Chat can be initialized."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        assert chat.llm == mock_llm_provider
        assert chat.llm_model == "test-model"
        assert chat.person == me

    def test_chat_basic_response(self, mock_llm_provider, temp_persona_file, mock_tools):
        """Test basic chat response without history."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        response = chat.chat("Hello", [])

        # Should return the mock response
        assert response == "Mock LLM response"

    def test_chat_with_history(
        self, mock_llm_provider, temp_persona_file, sample_chat_history, mock_tools
    ):
        """Test chat with conversation history."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        response = chat.chat("Tell me more", sample_chat_history)

        # Should return mock response even with history
        assert response == "Mock LLM response"

    def test_chat_rejects_invalid_message(self, mock_llm_provider, temp_persona_file, mock_tools):
        """Test that invalid messages raise InvalidMessageError."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        # 1-char message should raise exception
        with pytest.raises(InvalidMessageError) as exc_info:
            chat.chat("h", [])

        assert "invalid" in str(exc_info.value).lower()

    def test_chat_rejects_gibberish(self, mock_llm_provider, temp_persona_file, mock_tools):
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=mock_llm_provider, llm_model="test-model", llm_tools=mock_tools)

        with pytest.raises(InvalidMessageError):
            chat.chat("!@#$%^&*()", [])

    def test_chat_handles_llm_error(self, temp_persona_file, mock_tools):
        class FailingLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            def complete(self, **kwargs):
                raise Exception("LLM failed")

        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=FailingLLM(), llm_model="test", llm_tools=mock_tools)

        response = chat.chat("Hello", [])

        assert "unexpected" in response.lower()

    def test_chat_handles_api_timeout_error(self, temp_persona_file, mock_tools):
        from openai import APITimeoutError

        class TimeoutLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            def complete(self, **kwargs):
                raise APITimeoutError(request=None)

        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=TimeoutLLM(), llm_model="test", llm_tools=mock_tools)

        response = chat.chat("Hello", [])

        assert "longer than expected" in response.lower()

    def test_get_tools_returns_tools_when_supported(self, mock_llm_provider, temp_persona_file):
        class ToolsWithList:
            tools = [{"type": "function", "function": {"name": "test"}}]

        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=mock_llm_provider, llm_model="test", llm_tools=ToolsWithList())

        assert chat._get_tools() is not None
        assert len(chat._get_tools()) == 1

    def test_get_tools_returns_none_when_not_supported(self, temp_persona_file):
        class NoToolsLLM:
            @property
            def capabilities(self):
                return {"tools": False}

        class ToolsWithList:
            tools = [{"type": "function", "function": {"name": "test"}}]

        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=NoToolsLLM(), llm_model="test", llm_tools=ToolsWithList())

        assert chat._get_tools() is None

    def test_chat_handles_connection_error(self, temp_persona_file, mock_tools):
        from openai import APIConnectionError

        class ConnectionLLM:
            @property
            def capabilities(self):
                return {"tools": False}

            def complete(self, **kwargs):
                raise APIConnectionError(request=None)

        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(person=me, llm=ConnectionLLM(), llm_model="test", llm_tools=mock_tools)

        response = chat.chat("Hello", [])

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
