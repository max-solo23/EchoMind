from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.middleware.rate_limit_state import rate_limit_state


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")

    from config import Config

    test_config = Config(
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

    with (
        patch("api.dependencies.get_config", return_value=test_config),
        patch("api.main.get_config", return_value=test_config),
        patch("api.middleware.auth.get_config", return_value=test_config),
        patch("api.middleware.rate_limit.get_config", return_value=test_config),
    ):
        yield TestClient(app)


@pytest.fixture
def mock_chat_service():
    with patch("api.dependencies.get_chat_service") as mock:
        service = MagicMock()
        service.chat.return_value = "Test response"
        mock.return_value = service
        yield service


@pytest.fixture(autouse=True)
def reset_rate_limit():
    rate_limit_state.update_settings(enabled=True, rate_per_hour=15)
    yield


class TestRateLimiting:

    def test_chat_endpoint_enforces_rate_limit(self, client, mock_chat_service):
        counter = {"value": 0}

        async def mock_increment(key: str) -> int:
            counter["value"] += 1
            return counter["value"]

        with patch("api.middleware.rate_limit._increment_counter", side_effect=mock_increment):
            for i in range(15):
                response = client.post(
                    "/api/v1/chat",
                    json={"message": f"test {i}", "history": []},
                    headers={"X-API-Key": "test-api-key"},
                )
                assert response.status_code == 200

            response = client.post(
                "/api/v1/chat",
                json={"message": "test 16", "history": []},
                headers={"X-API-Key": "test-api-key"},
            )
            assert response.status_code == 429

    def test_rate_limit_response_format(self, client, mock_chat_service):
        async def mock_increment(key: str) -> int:
            return 100

        with patch("api.middleware.rate_limit._increment_counter", side_effect=mock_increment):
            response = client.post(
                "/api/v1/chat",
                json={"message": "test", "history": []},
                headers={"X-API-Key": "test-api-key"},
            )
            assert response.status_code == 429
            assert "detail" in response.json()

    def test_retry_after_header_present(self, client, mock_chat_service):
        async def mock_increment(key: str) -> int:
            return 100

        with patch("api.middleware.rate_limit._increment_counter", side_effect=mock_increment):
            response = client.post(
                "/api/v1/chat",
                json={"message": "test", "history": []},
                headers={"X-API-Key": "test-api-key"},
            )
            assert response.status_code == 429
            assert "retry-after" in response.headers

    def test_health_endpoint_not_rate_limited(self, client):
        for _ in range(20):
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limiting_disabled(self, client, mock_chat_service):
        rate_limit_state.update_settings(enabled=False, rate_per_hour=15)

        for i in range(20):
            response = client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={"X-API-Key": "test-api-key"},
            )
            assert response.status_code == 200

    def test_rate_limit_fails_open(self, client, mock_chat_service):
        async def mock_increment_error(key: str) -> int:
            raise Exception("DB connection failed")

        with patch("api.middleware.rate_limit._increment_counter", side_effect=mock_increment_error):
            response = client.post(
                "/api/v1/chat",
                json={"message": "test", "history": []},
                headers={"X-API-Key": "test-api-key"},
            )
            assert response.status_code == 200
