from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_chat_service
from api.main import app
from config import Config


@pytest.fixture
def test_config():
    return Config(
        llm_provider="openai",
        llm_api_key="test-key-123",
        llm_model="gpt-4",
        llm_base_url=None,
        persona_name="Test",
        persona_file="persona.yaml",
        pushover_token=None,
        pushover_user=None,
        api_key="test-api-key",
        allowed_origins=["http://localhost:3000"],
        rate_limit_per_hour=15,
    )


@pytest.fixture
def mock_chat_service():
    # FastAPI resolves Depends() against the original function object, so patching the
    # module attribute never reaches the endpoint. dependency_overrides is the only hook.
    service = MagicMock()
    service.chat = AsyncMock(return_value="Test response")
    app.dependency_overrides[get_chat_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_chat_service, None)


@pytest.fixture
def client(test_config, mock_chat_service):
    with (
        patch("api.dependencies.get_config", return_value=test_config),
        patch("api.main.get_config", return_value=test_config),
        patch("api.middleware.auth.get_config", return_value=test_config),
        patch("api.middleware.rate_limit.get_config", return_value=test_config),
    ):
        yield TestClient(app)
