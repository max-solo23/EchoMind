from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_db_session
from api.main import app
from api.middleware.auth import verify_api_key
from api.middleware.rate_limit_state import rate_limit_state
from api.routes import admin


NOW = datetime(2026, 1, 1, 12, 0, 0)
STATS = {"total_questions": 2, "total_variations": 4, "avg_variations_per_question": 2.0}
SESSION_SUMMARY = {
    "id": 1,
    "session_id": "sess_1",
    "user_ip": "127.0.0.1",
    "created_at": NOW,
    "last_activity": NOW,
    "message_count": 3,
}
CACHE_ENTRY = {
    "id": 5,
    "question": "What is Python?",
    "variations": ["A language"],
    "variation_index": 0,
    "hit_count": 1,
    "created_at": NOW,
    "last_used": NOW,
}


@asynccontextmanager
async def fake_session(_config):
    yield object()


@pytest.fixture
def logger(monkeypatch):
    fake = AsyncMock()

    async def fake_logger(session):
        return fake

    monkeypatch.setattr(admin, "get_conversation_logger", fake_logger)
    monkeypatch.setattr(admin, "is_database_configured", lambda: True)
    return fake


@pytest.fixture
def client(logger):
    overrides = {
        verify_api_key: lambda: None,
        admin.require_database: lambda: None,
        get_db_session: lambda: object(),
    }
    app.dependency_overrides.update(overrides)
    yield TestClient(app)
    for key in overrides:
        app.dependency_overrides.pop(key, None)


class TestAdminAccess:
    def test_requires_valid_api_key(self, client, monkeypatch):
        app.dependency_overrides.pop(verify_api_key)
        monkeypatch.setattr(
            "api.middleware.auth.get_config", lambda: SimpleNamespace(api_key="secret")
        )

        assert client.get("/api/v1/admin/health").status_code == 422
        assert client.get("/api/v1/admin/health", headers={"X-API-Key": "nope"}).status_code == 401
        assert client.get("/api/v1/admin/health", headers={"X-API-Key": "secret"}).status_code == 200

    def test_database_endpoints_return_503_without_database(self, client, monkeypatch):
        app.dependency_overrides.pop(admin.require_database)
        monkeypatch.setattr(admin, "is_database_configured", lambda: False)

        response = client.get("/api/v1/admin/cache/stats")

        assert response.status_code == 503
        assert "Database not configured" in response.json()["detail"]


class TestAdminHealth:
    def test_degraded_without_database(self, client, monkeypatch):
        monkeypatch.setattr(admin, "is_database_configured", lambda: False)

        body = client.get("/api/v1/admin/health").json()

        assert body["status"] == "degraded"
        assert body["database"] == {"configured": False, "connected": False, "error": None}
        assert body["cache"] is None

    def test_healthy_with_database(self, client, logger, monkeypatch):
        logger.get_cache_stats.return_value = STATS
        monkeypatch.setattr("api.dependencies.get_config", lambda: object())
        monkeypatch.setattr("repositories.connection.get_session", fake_session)

        body = client.get("/api/v1/admin/health").json()

        assert body["status"] == "healthy"
        assert body["database"]["connected"] is True
        assert body["cache"]["total_questions"] == 2

    def test_reports_database_error(self, client, monkeypatch):
        def broken(_config):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("api.dependencies.get_config", lambda: object())
        monkeypatch.setattr("repositories.connection.get_session", broken)

        body = client.get("/api/v1/admin/health").json()

        assert body["status"] == "degraded"
        assert body["database"]["error"] == "connection refused"


class TestCacheEndpoints:
    def test_stats(self, client, logger):
        logger.get_cache_stats.return_value = STATS

        response = client.get("/api/v1/admin/cache/stats")

        assert response.status_code == 200
        assert response.json()["avg_variations_per_question"] == 2.0

    def test_clear(self, client, logger):
        logger.clear_cache.return_value = 7

        assert client.delete("/api/v1/admin/cache").json() == {"success": True, "deleted_count": 7}

    def test_list_passes_pagination_and_sorting(self, client, logger):
        logger.list_cache_entries.return_value = {
            "entries": [CACHE_ENTRY],
            "total": 1,
            "page": 2,
            "limit": 5,
            "total_pages": 1,
        }

        response = client.get(
            "/api/v1/admin/cache/entries",
            params={"page": 2, "limit": 5, "sort_by": "hit_count", "order": "asc"},
        )

        assert response.status_code == 200
        assert response.json()["entries"][0]["question"] == "What is Python?"
        logger.list_cache_entries.assert_awaited_once_with(2, 5, "hit_count", "asc")

    def test_list_rejects_unknown_sort(self, client):
        response = client.get("/api/v1/admin/cache/entries", params={"sort_by": "bogus"})

        assert response.status_code == 422

    def test_search(self, client, logger):
        logger.search_cache.return_value = [
            {"id": 1, "question": "Python?", "hit_count": 1, "last_used": None}
        ]

        response = client.get("/api/v1/admin/cache/search", params={"q": "python", "limit": 5})

        assert response.status_code == 200
        assert response.json()["total"] == 1
        logger.search_cache.assert_awaited_once_with("python", 5)

    def test_search_requires_query(self, client):
        assert client.get("/api/v1/admin/cache/search").status_code == 422

    def test_get_entry(self, client, logger):
        logger.get_cache_entry.return_value = {**CACHE_ENTRY, "tfidf_vector": "[0.1]"}

        response = client.get("/api/v1/admin/cache/5")

        assert response.status_code == 200
        assert response.json()["tfidf_vector"] == "[0.1]"

    def test_get_missing_entry(self, client, logger):
        logger.get_cache_entry.return_value = None

        assert client.get("/api/v1/admin/cache/5").status_code == 404

    def test_update_entry(self, client, logger):
        logger.update_cache_entry.return_value = True

        response = client.put("/api/v1/admin/cache/5", json={"variations": ["new"]})

        assert response.status_code == 200
        assert response.json()["success"] is True
        logger.update_cache_entry.assert_awaited_once_with(5, ["new"])

    def test_update_missing_entry(self, client, logger):
        logger.update_cache_entry.return_value = False

        response = client.put("/api/v1/admin/cache/5", json={"variations": ["new"]})

        assert response.status_code == 404

    def test_update_rejects_empty_variations(self, client):
        assert client.put("/api/v1/admin/cache/5", json={"variations": []}).status_code == 422

    def test_delete_entry(self, client, logger):
        logger.delete_cache_entry.return_value = True

        assert client.delete("/api/v1/admin/cache/5").json() == {"success": True, "deleted_id": 5}

    def test_delete_missing_entry(self, client, logger):
        logger.delete_cache_entry.return_value = False

        assert client.delete("/api/v1/admin/cache/5").status_code == 404

    def test_cleanup_expired(self, client, logger):
        logger.cleanup_expired_cache.return_value = 3

        assert client.post("/api/v1/admin/cache/cleanup").json() == {
            "success": True,
            "deleted_count": 3,
        }


class TestSessionEndpoints:
    def test_history(self, client, logger):
        logger.get_session_history.return_value = {
            "id": 1,
            "session_id": "sess_1",
            "user_ip": None,
            "created_at": NOW,
            "conversations": [{"user_message": "hi", "bot_response": "hello", "timestamp": NOW}],
        }

        response = client.get("/api/v1/admin/sessions/sess_1")

        assert response.status_code == 200
        assert response.json()["conversations"][0]["bot_response"] == "hello"

    def test_history_missing(self, client, logger):
        logger.get_session_history.return_value = None

        assert client.get("/api/v1/admin/sessions/nope").status_code == 404

    def test_list_passes_pagination_and_sorting(self, client, logger):
        logger.list_sessions.return_value = {
            "sessions": [SESSION_SUMMARY],
            "total": 1,
            "page": 2,
            "limit": 5,
            "total_pages": 1,
        }

        response = client.get(
            "/api/v1/admin/sessions",
            params={"page": 2, "limit": 5, "sort_by": "last_activity", "order": "asc"},
        )

        assert response.status_code == 200
        assert response.json()["sessions"][0]["session_id"] == "sess_1"
        logger.list_sessions.assert_awaited_once_with(2, 5, "last_activity", "asc")

    def test_list_rejects_unknown_sort(self, client):
        response = client.get("/api/v1/admin/sessions", params={"sort_by": "bogus"})

        assert response.status_code == 422

    def test_delete(self, client, logger):
        logger.delete_session.return_value = True

        assert client.delete("/api/v1/admin/sessions/sess_1").json() == {
            "success": True,
            "session_id": "sess_1",
        }

    def test_delete_missing(self, client, logger):
        logger.delete_session.return_value = False

        assert client.delete("/api/v1/admin/sessions/nope").status_code == 404

    def test_clear_all(self, client, logger):
        logger.clear_all_sessions.return_value = 4

        assert client.delete("/api/v1/admin/sessions").json() == {
            "success": True,
            "deleted_count": 4,
        }


class TestRateLimitSettings:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        rate_limit_state.update_settings(enabled=True, rate_per_hour=10)
        yield
        rate_limit_state.update_settings(enabled=True, rate_per_hour=10)

    def test_get_without_database_has_no_buckets(self, client, monkeypatch):
        monkeypatch.setattr(admin, "is_database_configured", lambda: False)

        assert client.get("/api/v1/admin/rate-limit").json() == {
            "enabled": True,
            "rate_per_hour": 10,
            "active_limits": [],
        }

    def test_update_changes_live_settings(self, client):
        response = client.post(
            "/api/v1/admin/rate-limit", json={"enabled": False, "rate_per_hour": 99}
        )

        assert response.status_code == 200
        assert response.json() == {"enabled": False, "rate_per_hour": 99, "active_limits": []}
        assert rate_limit_state.get_settings() == {"enabled": False, "rate_per_hour": 99}

    def test_update_rejects_zero_rate(self, client):
        response = client.post("/api/v1/admin/rate-limit", json={"rate_per_hour": 0})

        assert response.status_code == 422
        assert rate_limit_state.rate_per_hour == 10
