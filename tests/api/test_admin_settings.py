from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api import dependencies
from api.routes import admin
from api.routes.chat import _chat_with_logging, _Turn


def fake_session():
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session.merge = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def config():
    with patch("api.routes.admin.get_config", return_value=SimpleNamespace(llm_model="gpt-6-luna")):
        yield


async def test_get_settings_reports_model_effort_and_cache(config, app_settings):
    response = await admin.get_app_settings_endpoint()

    assert response.model == "gpt-6-luna"
    assert response.reasoning_effort == "none"
    assert response.cache_enabled is True
    assert response.reasoning_efforts == ["none", "low", "medium", "high", "xhigh", "max"]


async def test_put_settings_applies_and_persists(config, app_settings):
    session = fake_session()
    request = admin.UpdateAppSettingsRequest(reasoning_effort="medium", cache_enabled=False)

    response = await admin.update_app_settings(request, session)

    assert (response.reasoning_effort, response.cache_enabled) == ("medium", False)
    assert app_settings.reasoning_effort == "medium"
    session.commit.assert_awaited_once()


async def test_put_partial_update_keeps_other_setting(config, app_settings):
    request = admin.UpdateAppSettingsRequest(reasoning_effort="low")

    response = await admin.update_app_settings(request, fake_session())

    assert (response.reasoning_effort, response.cache_enabled) == ("low", True)


async def test_put_rejects_unknown_effort(config, app_settings):
    request = admin.UpdateAppSettingsRequest(reasoning_effort="extreme")

    with pytest.raises(HTTPException) as error:
        await admin.update_app_settings(request, fake_session())

    assert error.value.status_code == 400
    assert app_settings.reasoning_effort == "none"


def test_put_without_database_returns_503(client):
    response = client.put(
        "/api/v1/admin/settings",
        json={"cache_enabled": True},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 503


def test_settings_require_api_key(client):
    assert client.get("/api/v1/admin/settings").status_code in (401, 422)


async def test_disabled_cache_skips_lookup_and_calls_llm(app_settings):
    app_settings._values["cache_enabled"] = False
    chat_service = AsyncMock()
    chat_service.chat.return_value = "Fresh reply"

    with (
        patch("api.routes.chat.is_database_configured", return_value=True),
        patch("api.routes.chat.get_session", side_effect=AssertionError("no lookup when off")),
        patch("api.routes.chat._persist_in_background", AsyncMock()),
    ):
        reply = await _chat_with_logging(chat_service, _Turn("Question here", [], "s", None))

    assert reply == "Fresh reply"


async def test_refresh_reads_database_once_per_ttl(app_settings):
    session = fake_session()

    @asynccontextmanager
    async def get_session(_config):
        yield session

    with (
        patch("api.dependencies.is_database_configured", return_value=True),
        patch("api.dependencies.get_session", get_session),
        patch("api.dependencies.get_config", return_value=object()),
    ):
        await dependencies.refresh_app_settings()
        await dependencies.refresh_app_settings()

    session.execute.assert_awaited_once()
