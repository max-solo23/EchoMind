from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from api.main import app
from api.middleware import auth, cors, rate_limit
from api.middleware.rate_limit_state import RateLimitState, rate_limit_state


def _request(headers: dict[str, str], client=("192.0.2.1", 1234)) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": client,
    }
    return Request(scope)


class TestVerifyApiKey:
    async def test_accepts_matching_key(self):
        with patch.object(auth, "get_config", return_value=SimpleNamespace(api_key="secret")):
            assert await auth.verify_api_key("secret") is None

    async def test_rejects_wrong_key(self):
        with (
            patch.object(auth, "get_config", return_value=SimpleNamespace(api_key="secret")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await auth.verify_api_key("nope")

        assert exc_info.value.status_code == 401

    async def test_fails_when_server_has_no_key(self):
        with (
            patch.object(auth, "get_config", return_value=SimpleNamespace(api_key="")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await auth.verify_api_key("anything")

        assert exc_info.value.status_code == 500


class TestSetupCors:
    def test_blocks_everything_without_origins(self, capsys):
        target = FastAPI()

        with patch.object(cors, "get_config", return_value=SimpleNamespace(allowed_origins=[])):
            cors.setup_cors(target)

        assert target.user_middleware[0].kwargs["allow_origins"] == []
        assert "No ALLOWED_ORIGINS" in capsys.readouterr().out

    def test_passes_configured_origins(self):
        target = FastAPI()
        config = SimpleNamespace(allowed_origins=["http://a"])

        with patch.object(cors, "get_config", return_value=config):
            cors.setup_cors(target)

        kwargs = target.user_middleware[0].kwargs
        assert kwargs["allow_origins"] == ["http://a"]
        assert kwargs["allow_methods"] == ["GET", "POST"]


class TestApp:
    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()

        assert body["name"] == "EchoMind API"
        assert body["endpoints"]["chat"] == "POST /api/v1/chat"

    def test_lifespan_applies_rate_limit_config_and_closes_database(self, test_config):
        config = replace(test_config, rate_limit_enabled=False, rate_limit_per_hour=42)
        close_database = AsyncMock()

        with (
            patch("api.main.get_config", return_value=config),
            patch("api.main.close_database", close_database),
            TestClient(app),
        ):
            assert rate_limit_state.get_settings() == {"enabled": False, "rate_per_hour": 42}

        close_database.assert_awaited_once()
        rate_limit_state.update_settings(enabled=True, rate_per_hour=10)


class TestIncrementCounter:
    async def test_commits_and_returns_count(self):
        result = MagicMock()
        result.scalar_one.return_value = 7
        session = AsyncMock()
        session.execute.return_value = result

        @asynccontextmanager
        async def fake_session(_config):
            yield session

        with (
            patch.object(rate_limit, "get_config", return_value=object()),
            patch.object(rate_limit, "get_session", fake_session),
        ):
            assert await rate_limit._increment_counter("rate_limit:1.2.3.4") == 7

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


class TestGetClientIp:
    def test_ignores_blank_proxy_headers(self):
        request = _request({"fly-client-ip": "  ", "x-forwarded-for": " , 10.0.0.1"})

        assert rate_limit._get_client_ip(request) == "192.0.2.1"

    def test_unknown_without_client(self):
        assert rate_limit._get_client_ip(_request({}, client=None)) == "unknown"


class TestRateLimitState:
    def test_rejects_rate_below_one(self):
        state = RateLimitState()

        with pytest.raises(ValueError):
            state.update_settings(rate_per_hour=0)

        assert state.rate_per_hour == 10

    def test_partial_update_keeps_other_setting(self):
        state = RateLimitState(enabled=True, rate_per_hour=10)

        state.update_settings(enabled=False)
        assert state.get_settings() == {"enabled": False, "rate_per_hour": 10}

        state.update_settings(rate_per_hour=3)
        assert state.get_settings() == {"enabled": False, "rate_per_hour": 3}
