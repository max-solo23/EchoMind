from threading import Lock
from typing import NamedTuple


class RateLimitSettings(NamedTuple):
    enabled: bool
    rate_per_hour: int


class RateLimitState:
    def __init__(self, enabled: bool = True, rate_per_hour: int = 10):
        self._lock = Lock()
        self._enabled = enabled
        self._rate_per_hour = rate_per_hour

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def rate_per_hour(self) -> int:
        with self._lock:
            return self._rate_per_hour

    def get_settings(self) -> dict:
        with self._lock:
            return {"enabled": self._enabled, "rate_per_hour": self._rate_per_hour}

    def update_settings(self, enabled: bool | None = None, rate_per_hour: int | None = None):
        with self._lock:
            if enabled is not None:
                self._enabled = enabled
            if rate_per_hour is not None:
                if rate_per_hour < 1:
                    raise ValueError("Rate limit must be at least 1 request per hour")
                self._rate_per_hour = rate_per_hour


rate_limit_state = RateLimitState()
