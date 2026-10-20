from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from services import app_settings as module
from services.app_settings import AppSettings


def session_with_rows(*rows):
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session.merge = AsyncMock()
    session.commit = AsyncMock()
    return session


def test_defaults_keep_cache_off():
    settings = AppSettings()

    assert settings.snapshot() == {"reasoning_effort": "none", "cache_enabled": False}
    assert settings.is_stale()


async def test_load_applies_known_rows_and_ignores_unknown():
    settings = AppSettings()

    await settings.load(
        session_with_rows(
            SimpleNamespace(key="cache_enabled", value=True),
            SimpleNamespace(key="reasoning_effort", value="medium"),
            SimpleNamespace(key="legacy", value=1),
        )
    )

    assert settings.snapshot() == {"reasoning_effort": "medium", "cache_enabled": True}
    assert not settings.is_stale()


async def test_stale_again_after_ttl(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    settings = AppSettings()
    await settings.load(session_with_rows())

    clock["now"] += module.SETTINGS_TTL_SECONDS - 1
    assert not settings.is_stale()
    clock["now"] += 1
    assert settings.is_stale()


async def test_failed_load_keeps_values_and_waits_a_ttl():
    settings = AppSettings(reasoning_effort="low")
    session = session_with_rows()
    session.execute.side_effect = RuntimeError("db down")

    await settings.load(session)

    assert settings.reasoning_effort == "low"
    assert not settings.is_stale()


async def test_update_persists_and_applies_immediately():
    settings = AppSettings()
    session = session_with_rows()

    result = await settings.update(session, {"cache_enabled": True, "reasoning_effort": "high"})

    assert result == {"reasoning_effort": "high", "cache_enabled": True}
    assert session.merge.await_count == 2
    session.commit.assert_awaited_once()


@pytest.mark.parametrize("changes", [{"reasoning_effort": "extreme"}, {"temperature": 1}])
async def test_update_rejects_invalid_changes_without_writing(changes):
    settings = AppSettings()
    session = session_with_rows()

    with pytest.raises(ValueError):
        await settings.update(session, changes)

    session.commit.assert_not_awaited()
    assert settings.snapshot() == AppSettings().snapshot()
