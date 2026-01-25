import os
import tempfile

import pytest

from config import Config
from core.llm.types import CompletionMessage, CompletionResponse


@pytest.fixture
def mock_env_vars():
    return {
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "test-key-123",
        "LLM_MODEL": "gpt-5-mini",
        "API_KEY": "test-api-key",
        "ALLOWED_ORIGINS": "http://localhost:3000",
    }


@pytest.fixture
def config(mock_env_vars, monkeypatch):
    for key, value in mock_env_vars.items():
        monkeypatch.setenv(key, value)

    return Config.from_env()


@pytest.fixture
def mock_llm_provider():
    class MockLLM:
        def complete(self, *, model: str, messages: list[dict], tools: list[dict] | None = None):
            return CompletionResponse(
                finish_reason="stop",
                message=CompletionMessage(
                    role="assistant", content="Mock LLM response", tool_calls=None
                ),
            )

        @property
        def capabilities(self):
            return {"tools": True, "streaming": True}

    return MockLLM()


@pytest.fixture
def sample_chat_history():
    """Example of chat history for testing conversations."""
    return [
        {"role": "user", "content": "What's your name?"},
        {"role": "assistant", "content": "I'm EchoMind"},
        {"role": "user", "content": "What can you do?"},
        {"role": "assistant", "content": "I can answer questions on User's background."},
    ]


@pytest.fixture
def temp_persona_file():
    """Create a temporary persona YAML file for testing."""

    temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", delete=False, suffix=".yaml"
    )

    yaml_content = """
    name: "Test User"
    role: "Software Engineer"
    background: "Testing expert"
    """
    temp_file.write(yaml_content)
    temp_file.close()

    yield temp_file.name

    os.unlink(temp_file.name)
