import json
from unittest.mock import MagicMock

from tools.llm_tools import Tools


def test_tools_initializes_with_schemas():
    tools = Tools()

    assert len(tools.tools) == 2
    assert tools.tools[0]["type"] == "function"
    assert tools.tools[1]["type"] == "function"
    assert tools.tools[0]["function"]["name"] == "record_user_details"
    assert tools.tools[1]["function"]["name"] == "record_unknown_question"


def test_record_user_details_without_notification():
    tools = Tools(message_app=None)

    result = tools.record_user_details(email="test@example.com", name="Test User")

    assert result == {"recorded": "ok"}


def test_record_user_details_with_notification():
    class MockNotificationProvider:
        def __init__(self):
            self.messages = []

        def push(self, text: str):
            self.messages.append(text)

    mock_notifier = MockNotificationProvider()
    tools = Tools(message_app=mock_notifier)

    result = tools.record_user_details(
        email="test@example.com", name="Test User", notes="Interested in something"
    )

    assert result == {"recorded": "ok"}
    assert len(mock_notifier.messages) == 1
    assert "Test User" in mock_notifier.messages[0]
    assert "test@example.com" in mock_notifier.messages[0]
    assert "Interested in something" in mock_notifier.messages[0]


def test_record_user_details_with_defaults():
    class MockNotificationProvider:
        def __init__(self):
            self.messages = []

        def push(self, text: str):
            self.messages.append(text)

    mock_notifier = MockNotificationProvider()
    tools = Tools(message_app=mock_notifier)

    result = tools.record_user_details(email="test@example.com")

    assert result == {"recorded": "ok"}
    assert len(mock_notifier.messages) == 1
    assert "Name not provided" in mock_notifier.messages[0]
    assert "not provided" in mock_notifier.messages[0]


def test_record_unknown_question_without_notification():
    tools = Tools(message_app=None)

    result = tools.record_unknown_question(question="What is your favorite color?")

    assert result == {"recorded": "ok"}


def test_record_unknown_question_with_notification():
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


def test_handle_tool_call_dispatches_correctly():
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
    tools = Tools()

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "nonexistent_tool"
    mock_tool_call.function.arguments = json.dumps({})
    mock_tool_call.id = "call_456"

    results = tools.handle_tool_call([mock_tool_call])

    assert len(results) == 1
    assert json.loads(results[0]["content"]) == {}


def test_handle_tool_call_multiple_calls():
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
