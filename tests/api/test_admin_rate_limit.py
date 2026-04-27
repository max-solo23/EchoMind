from types import SimpleNamespace

import pytest

from api.middleware.rate_limit_state import rate_limit_state
from api.routes import admin


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeExecuteResult:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return FakeScalarResult(self.rows)


class FakeSession:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, statement):
        return FakeExecuteResult(self.rows)


class FakeSessionContext:
    def __init__(self, rows):
        self.rows = rows

    async def __aenter__(self):
        return FakeSession(self.rows)

    async def __aexit__(self, exc_type, exc, tb):
        return None


@pytest.fixture(autouse=True)
def reset_rate_limit_state():
    rate_limit_state.update_settings(enabled=True, rate_per_hour=10)
    yield


async def test_rate_limit_settings_return_empty_active_limits_without_database(monkeypatch):
    monkeypatch.setattr(admin, "is_database_configured", lambda: False)

    response = await admin.get_rate_limit_settings()

    assert response.enabled is True
    assert response.rate_per_hour == 10
    assert response.active_limits == []


async def test_rate_limit_settings_include_active_buckets(monkeypatch):
    rows = [SimpleNamespace(key="rate_limit:203.0.113.10", count=3, expiry=1700000300)]

    monkeypatch.setattr(admin, "is_database_configured", lambda: True)
    monkeypatch.setattr(admin, "get_config", lambda: SimpleNamespace(database_url="postgresql://test"))
    monkeypatch.setattr(admin.time, "time", lambda: 1700000000)
    monkeypatch.setattr(admin, "get_session", lambda config: FakeSessionContext(rows))

    response = await admin.get_rate_limit_settings()

    assert response.active_limits[0].ip == "203.0.113.10"
    assert response.active_limits[0].used == 3
    assert response.active_limits[0].limit == 10
    assert response.active_limits[0].ttl_seconds == 300
