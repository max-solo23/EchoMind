import logging
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import AppSetting


logger = logging.getLogger(__name__)

# ponytail: each machine re-reads the table at most this often, so an admin change reaches
# the other Fly machine within this window; push invalidation if seconds start to matter.
SETTINGS_TTL_SECONDS = 30.0

REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


class AppSettings:
    """Admin-controlled runtime settings, stored in app_settings and cached in memory."""

    def __init__(
        self,
        reasoning_effort: str = "none",
        cache_enabled: bool = False,
        rate_limit_enabled: bool = True,
        rate_limit_per_hour: int = 10,
    ):
        self._values: dict[str, Any] = {
            "reasoning_effort": reasoning_effort,
            "cache_enabled": cache_enabled,
            "rate_limit_enabled": rate_limit_enabled,
            "rate_limit_per_hour": rate_limit_per_hour,
        }
        self._loaded_at: float | None = None

    @property
    def reasoning_effort(self) -> str:
        return str(self._values["reasoning_effort"])

    @property
    def cache_enabled(self) -> bool:
        return bool(self._values["cache_enabled"])

    @property
    def rate_limit_enabled(self) -> bool:
        return bool(self._values["rate_limit_enabled"])

    @property
    def rate_limit_per_hour(self) -> int:
        return int(self._values["rate_limit_per_hour"])

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)

    def is_stale(self) -> bool:
        return self._loaded_at is None or time.monotonic() - self._loaded_at >= SETTINGS_TTL_SECONDS

    async def load(self, session: AsyncSession) -> None:
        # A failed read keeps the last known values and waits a full TTL before retrying,
        # so a database hiccup can neither break chat nor add a query to every request.
        self._loaded_at = time.monotonic()
        try:
            rows = (await session.execute(select(AppSetting))).scalars().all()
        except Exception:
            logger.exception("Failed to load app settings; keeping previous values")
            return
        for row in rows:
            if row.key in self._values:
                self._values[row.key] = row.value

    async def update(self, session: AsyncSession, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = set(changes) - set(self._values)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        effort = changes.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        rate = changes.get("rate_limit_per_hour")
        if rate is not None and (not isinstance(rate, int) or rate < 1):
            raise ValueError("rate_limit_per_hour must be at least 1")

        for key, value in changes.items():
            await session.merge(AppSetting(key=key, value=value))
        await session.commit()
        self._values.update(changes)
        self._loaded_at = time.monotonic()
        return self.snapshot()
