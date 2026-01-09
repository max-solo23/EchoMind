"""
Tests for Chat.py - Main conversation handler

Key concepts tested:
1. Using mock_llm_provider fixture for LLM interactions
2. Message validation logic
3. Chat flow with history
4. Exception handling
"""
import pytest

from Chat import Chat, InvalidMessageError
from Me import Me


class TestMessageValidation:
    """Test the message validation logic."""

    def test_valid_message(self):
        """Test that valid messages pass validation."""
        assert Chat._is_valid_message("Hello, how are you?") is True
        assert Chat._is_valid_message("What's your background?") is True
        assert Chat._is_valid_message("Tell me about yourself") is True

    def test_too_short_message(self):
        """Test that messages under 3 characters are rejected."""
        assert Chat._is_valid_message("hi") is False
        assert Chat._is_valid_message("ok") is False
        assert Chat._is_valid_message("") is False

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
        chat = Chat(
            person=me,
            llm=mock_llm_provider,
            llm_model="test-model",
            llm_tools=mock_tools
        )

        assert chat.llm == mock_llm_provider
        assert chat.llm_model == "test-model"
        assert chat.person == me

    def test_chat_basic_response(self, mock_llm_provider, temp_persona_file, mock_tools):
        """Test basic chat response without history."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(
            person=me,
            llm=mock_llm_provider,
            llm_model="test-model",
            llm_tools=mock_tools
        )

        response = chat.chat("Hello", [])

        # Should return the mock response
        assert response == "Mock LLM response"

    def test_chat_with_history(self, mock_llm_provider, temp_persona_file, sample_chat_history, mock_tools):
        """Test chat with conversation history."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(
            person=me,
            llm=mock_llm_provider,
            llm_model="test-model",
            llm_tools=mock_tools
        )

        response = chat.chat("Tell me more", sample_chat_history)

        # Should return mock response even with history
        assert response == "Mock LLM response"

    def test_chat_rejects_invalid_message(self, mock_llm_provider, temp_persona_file, mock_tools):
        """Test that invalid messages raise InvalidMessageError."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(
            person=me,
            llm=mock_llm_provider,
            llm_model="test-model",
            llm_tools=mock_tools
        )

        # Too short message should raise exception
        with pytest.raises(InvalidMessageError) as exc_info:
            chat.chat("hi", [])

        assert "invalid" in str(exc_info.value).lower()

    def test_chat_rejects_gibberish(self, mock_llm_provider, temp_persona_file, mock_tools):
        """Test that gibberish is rejected."""
        me = Me(name="Test User", persona_yaml_file=temp_persona_file)
        chat = Chat(
            person=me,
            llm=mock_llm_provider,
            llm_model="test-model",
            llm_tools=mock_tools
        )

        with pytest.raises(InvalidMessageError):
            chat.chat("!@#$%^&*()", [])
