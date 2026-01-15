"""
Tests for Tools.py - LLM function calling tools

Key concepts tested:
1. Mocking external services (NotificationProvider)
2. Testing with and without dependencies
3. Verifying method calls on mocks
4. Testing tool routing logic
"""

import json
from unittest.mock import MagicMock

from Tools import Tools


def test_tools_initializes_with_schemas():
    """Test that Tools creates tool schemas on init."""
    tools = Tools()

    # Should have 2 tools
    assert len(tools.tools) == 2

    # Each tool should have type and function
    assert tools.tools[0]["type"] == "function"
    assert tools.tools[1]["type"] == "function"

    # Should have the expected tool names
    assert tools.tools[0]["function"]["name"] == "record_user_details"
    assert tools.tools[1]["function"]["name"] == "record_unknown_question"


def test_record_user_details_without_notification():
    """Test record_user_details works without message_app (prints instead)."""
    tools = Tools(message_app=None)  # No notification service

    result = tools.record_user_details(email="test@example.com", name="Test User")

    # Should return success
    assert result == {"recorded": "ok"}


def test_record_user_details_with_notification():
    """Test record_user_details sends notification when message_app exists."""

    # Create a mock notification provider
    class MockNotificationProvider:
        def __init__(self):
            self.messages = []  # Track all messages sent

        def push(self, text: str):
            self.messages.append(text)

    mock_notifier = MockNotificationProvider()
    tools = Tools(message_app=mock_notifier)

    result = tools.record_user_details(
        email="test@example.com", name="Test User", notes="Interested in something"
    )

    # Should return success
    assert result == {"recorded": "ok"}

    # Should have sent notification
    assert len(mock_notifier.messages) == 1
    assert "Test User" in mock_notifier.messages[0]
    assert "test@example.com" in mock_notifier.messages[0]
    assert "Interested in something" in mock_notifier.messages[0]


def test_record_user_details_with_defaults():
    """Test record_user_details uses default values for optional params."""

    class MockNotificationProvider:
        def __init__(self):
            self.messages = []

        def push(self, text: str):
            self.messages.append(text)

    mock_notifier = MockNotificationProvider()
    tools = Tools(message_app=mock_notifier)

    # Only provide email (name and notes should use defaults)
    result = tools.record_user_details(email="test@example.com")

    assert result == {"recorded": "ok"}
    assert len(mock_notifier.messages) == 1
    assert "Name not provided" in mock_notifier.messages[0]
    assert "not provided" in mock_notifier.messages[0]


def test_record_unknown_question_without_notification():
    """Test record_unknown_question works without message_app."""
    tools = Tools(message_app=None)

    result = tools.record_unknown_question(question="What is your favorite color?")

    assert result == {"recorded": "ok"}


def test_record_unknown_question_with_notification():
    """Test record_unknown_question sends notification when message_app exists."""

    class MockNotificationProvider:
        def __init__(self):
            self.messages = []

        def push(self, text: str):
            self.messages.append(text)

    mock_notifier = MockNotificationProvider()
    tools = Tools(message_app=mock_notifier)

    question = "What is your favorite color?"
    result = tools.record_unknown_question(question=question)

    assert result == {"recorded": "ok"}
    assert len(mock_notifier.messages) == 1
    assert question in mock_notifier.messages[0]


# Tests for handle_tool_call - the dispatch method


def test_handle_tool_call_dispatches_correctly():
    """
    Test handle_tool_call routes to correct method.

    Why: LLM returns tool name as string. handle_tool_call must
    dynamically find and call the right method using getattr.
    """

    class MockNotificationProvider:
        def push(self, text: str):
            pass

    tools = Tools(message_app=MockNotificationProvider())

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "record_user_details"
    mock_tool_call.function.arguments = json.dumps(
        {"email": "dispatch@test.com", "name": "Test User"}
    )
    mock_tool_call.id = "call_123"

    results = tools.handle_tool_call([mock_tool_call])

    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "call_123"
    assert json.loads(results[0]["content"]) == {"recorded": "ok"}


def test_handle_tool_call_unknown_tool_returns_empty():
    """
    Test handle_tool_call returns empty dict for unknown tools.

    Why: Malformed LLM response or security probe shouldn't crash.
    getattr returns None for unknown attributes, ternary returns {}.
    """
    tools = Tools()

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "nonexistent_tool"
    mock_tool_call.function.arguments = json.dumps({})
    mock_tool_call.id = "call_456"

    results = tools.handle_tool_call([mock_tool_call])

    assert len(results) == 1
    assert json.loads(results[0]["content"]) == {}


def test_handle_tool_call_multiple_calls():
    """
    Test handle_tool_call processes multiple tool calls.

    Why: LLM can return multiple tool calls in one response.
    Each must be processed and returned with correct tool_call_id.
    """
    tools = Tools(message_app=None)

    call1 = MagicMock()
    call1.function.name = "record_user_details"
    call1.function.arguments = json.dumps({"email": "a@test.com"})
    call1.id = "call_1"

    call2 = MagicMock()
    call2.function.name = "record_unknown_question"
    call2.function.arguments = json.dumps({"question": "Test?"})
    call2.id = "call_2"

    results = tools.handle_tool_call([call1, call2])

    assert len(results) == 2
    assert results[0]["tool_call_id"] == "call_1"
    assert results[1]["tool_call_id"] == "call_2"
