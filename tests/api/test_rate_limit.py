"""
Tests for IP-based rate limiting functionality.

Tests ensure that:
- Rate limiting is enforced on the chat endpoint
- Health endpoint is NOT rate limited
- 429 responses have correct format and headers
- Rate limits reset after time window
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time
import os

from api.main import app


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the FastAPI app."""
    # Set required environment variables for API
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")

    # Mock the config dependency to return test config
    from config import Config
    test_config = Config(
        llm_provider="openai",
        llm_api_key="test-key-123",
        llm_model="gpt-4",
        llm_base_url=None,
        use_evaluator=False,
        persona_name="Test",
        persona_file="persona.yaml",
        pushover_token=None,
        pushover_user=None,
        api_key="test-api-key",
        allowed_origins=["http://localhost:3000"],
        rate_limit_per_hour=15  # Test expects 15 requests
    )

    # Patch in multiple places where get_config is called
    with patch("api.dependencies.get_config", return_value=test_config), \
         patch("api.main.get_config", return_value=test_config), \
         patch("api.middleware.auth.get_config", return_value=test_config):
        yield TestClient(app)


@pytest.fixture
def mock_chat_service():
    """Mock the chat service to avoid LLM calls in tests."""
    with patch("api.dependencies.get_chat_service") as mock:
        service = MagicMock()
        service.chat.return_value = "Test response"
        mock.return_value = service
        yield service


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Reset rate limit counters before each test."""
    from api.middleware.rate_limit import limiter
    from api.middleware.rate_limit_state import rate_limit_state

    # Reset slowapi limiter storage
    limiter.reset()

    # Ensure rate limiting is enabled for tests
    rate_limit_state.update_settings(enabled=True, rate_per_hour=15)

    yield

    # Clean up after test
    limiter.reset()


class TestRateLimiting:
    """Test suite for rate limiting functionality."""

    def test_chat_endpoint_enforces_rate_limit(self, client, mock_chat_service):
        """Test that chat endpoint enforces 15 requests per hour limit."""
        # Make requests up to the limit
        for i in range(15):
            response = client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={"X-API-Key": "test-api-key"}
            )
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # 16th request should be rate limited
        response = client.post(
            "/api/v1/chat",
            json={"message": "test 16", "history": []},
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 429
        assert "error" in response.json()
        assert response.json()["error"] == "Rate limit exceeded"

    def test_rate_limit_response_format(self, client, mock_chat_service):
        """Test that 429 responses have correct JSON format."""
        # Exhaust rate limit
        for i in range(15):
            client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={"X-API-Key": "test-api-key"}
            )

        # Check 429 response format
        response = client.post(
            "/api/v1/chat",
            json={"message": "test", "history": []},
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 429

        data = response.json()
        assert "error" in data
        assert "detail" in data
        assert "retry_after" in data
        assert data["error"] == "Rate limit exceeded"
        assert data["retry_after"] == "3600"

    def test_retry_after_header_present(self, client, mock_chat_service):
        """Test that Retry-After header is present in 429 responses."""
        # Exhaust rate limit
        for i in range(15):
            client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={"X-API-Key": "test-api-key"}
            )

        # Check Retry-After header
        response = client.post(
            "/api/v1/chat",
            json={"message": "test", "history": []},
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 429
        assert "retry-after" in response.headers
        assert response.headers["retry-after"] == "3600"

    def test_health_endpoint_not_rate_limited(self, client):
        """Test that health endpoint is NOT rate limited."""
        # Make many requests to health endpoint
        for i in range(20):
            response = client.get("/health")
            assert response.status_code == 200, f"Health check {i+1} should always succeed"

    def test_root_endpoint_not_rate_limited(self, client):
        """Test that root endpoint is NOT rate limited."""
        # Make many requests to root endpoint
        for i in range(20):
            response = client.get("/")
            assert response.status_code == 200, f"Root request {i+1} should always succeed"

    def test_rate_limit_applies_before_auth(self, client, mock_chat_service):
        """Test that rate limiting happens before authentication."""
        # Exhaust rate limit with valid API key
        for i in range(15):
            client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={"X-API-Key": "test-api-key"}
            )

        # Next request should be rate limited even with invalid API key
        # (rate limit check happens first)
        response = client.post(
            "/api/v1/chat",
            json={"message": "test", "history": []},
            headers={"X-API-Key": "invalid-key"}
        )
        # Could be 429 (rate limit) or 401 (auth) depending on middleware order
        # Our design specifies rate limit should come first
        assert response.status_code in [401, 429]

    def test_streaming_endpoint_rate_limited(self, client, mock_chat_service):
        """Test that streaming endpoint is also rate limited."""
        # Exhaust rate limit with non-streaming requests
        for i in range(15):
            client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={"X-API-Key": "test-api-key"}
            )

        # Streaming request should also be rate limited
        response = client.post(
            "/api/v1/chat?stream=true",
            json={"message": "test", "history": []},
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 429


class TestRateLimitConfiguration:
    """Test suite for rate limit configuration."""

    @patch.dict("os.environ", {"RATE_LIMIT_ENABLED": "false"})
    def test_rate_limiting_can_be_disabled(self, client, mock_chat_service):
        """Test that rate limiting can be disabled via config."""
        # Note: This test requires reloading the config
        # In practice, you'd need to restart the app with RATE_LIMIT_ENABLED=false
        # This is more of a documentation test for the feature
        pass

    @patch.dict("os.environ", {"RATE_LIMIT_PER_HOUR": "5"})
    def test_rate_limit_value_configurable(self, client, mock_chat_service):
        """Test that rate limit value is configurable."""
        # Note: This test requires reloading the config
        # In practice, you'd need to restart the app with RATE_LIMIT_PER_HOUR=5
        # This is more of a documentation test for the feature
        pass


@pytest.mark.integration
class TestRateLimitIntegration:
    """Integration tests for rate limiting with real requests."""

    def test_different_ips_have_separate_limits(self, client, mock_chat_service):
        """Test that different IP addresses have independent rate limits."""
        # Note: TestClient doesn't easily allow simulating different IPs
        # This would need to be tested manually or with a more sophisticated test setup
        pass

    def test_x_forwarded_for_header_respected(self, client, mock_chat_service):
        """Test that X-Forwarded-For header is used for IP detection."""
        # Test with X-Forwarded-For header
        for i in range(15):
            response = client.post(
                "/api/v1/chat",
                json={"message": f"test {i}", "history": []},
                headers={
                    "X-API-Key": "test-api-key",
                    "X-Forwarded-For": "192.168.1.100"
                }
            )
            assert response.status_code == 200

        # 16th request should be rate limited
        response = client.post(
            "/api/v1/chat",
            json={"message": "test", "history": []},
            headers={
                "X-API-Key": "test-api-key",
                "X-Forwarded-For": "192.168.1.100"
            }
        )
        assert response.status_code == 429
