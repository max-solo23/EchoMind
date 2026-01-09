"""
Tests for Tools.py - LLM function calling tools

Key concepts tested:
1. Mocking external services (NotificationProvider)
2. Testing with and without dependencies
3. Verifying method calls on mocks
4. Testing tool routing logic
"""
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
        email="test@example.com",
        name="Test User",
        notes="Interested in something"
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
