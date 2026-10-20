from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from api.middleware.rate_limit import _build_increment_statement


@pytest.fixture(autouse=True)
def limit_of_fifteen(app_settings):
    app_settings._values["rate_limit_per_hour"] = 15
    return app_settings


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

    def test_rate_limiting_disabled(self, client, mock_chat_service, app_settings):
        app_settings._values["rate_limit_enabled"] = False

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

        with patch(
            "api.middleware.rate_limit._increment_counter", side_effect=mock_increment_error
        ):
            response = client.post(
                "/api/v1/chat",
                json={"message": "test", "history": []},
                headers={"X-API-Key": "test-api-key"},
            )
            assert response.status_code == 200

    def test_rate_limit_uses_forwarded_client_ip(self, client, mock_chat_service):
        captured_keys = []

        async def mock_increment(key: str) -> int:
            captured_keys.append(key)
            return 1

        with patch("api.middleware.rate_limit._increment_counter", side_effect=mock_increment):
            response = client.post(
                "/api/v1/chat",
                json={"message": "test", "history": []},
                headers={
                    "X-API-Key": "test-api-key",
                    "X-Forwarded-For": "203.0.113.10, 10.0.0.1",
                },
            )

        assert response.status_code == 200
        assert captured_keys == ["rate_limit:203.0.113.10"]

    def test_rate_limit_prefers_fly_client_ip(self, client, mock_chat_service):
        captured_keys = []

        async def mock_increment(key: str) -> int:
            captured_keys.append(key)
            return 1

        with patch("api.middleware.rate_limit._increment_counter", side_effect=mock_increment):
            response = client.post(
                "/api/v1/chat",
                json={"message": "test", "history": []},
                headers={
                    "X-API-Key": "test-api-key",
                    "Fly-Client-IP": "198.51.100.20",
                    "X-Forwarded-For": "203.0.113.10, 10.0.0.1",
                },
            )

        assert response.status_code == 200
        assert captured_keys == ["rate_limit:198.51.100.20"]

    def test_expired_rate_limit_window_resets_counter(self):
        stmt = _build_increment_statement(
            key="rate_limit:203.0.113.10", now_timestamp=100, expiry_timestamp=3700
        )

        compiled = stmt.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
        compiled_sql = str(compiled)

        assert "count = CASE WHEN (rate_limit.expiry <=" in compiled_sql
        assert "THEN" in compiled_sql
        assert "ELSE rate_limit.count +" in compiled_sql
        assert "expiry = CASE WHEN (rate_limit.expiry <=" in compiled_sql
        assert "ELSE rate_limit.expiry END" in compiled_sql
        assert "RETURNING rate_limit.count" in compiled_sql
        assert compiled.params["expiry_1"] == 100
        assert compiled.params["param_1"] == 1
        assert compiled.params["param_2"] == 3700
