"""
Dynamic rate limiting state manager.

Provides thread-safe global state to enable/disable rate limiting at runtime
without server restart.
"""

from threading import Lock
from typing import NamedTuple


class RateLimitSettings(NamedTuple):
    """Immutable snapshot of rate limit settings."""

    enabled: bool
    rate_per_hour: int


class RateLimitState:
    """
    Thread-safe global state manager for rate limiting configuration.

    Allows dynamic toggling of rate limiting via admin endpoints.
    Uses locks to prevent race conditions when reading/writing state.
    """

    def __init__(self, enabled: bool = True, rate_per_hour: int = 10):
        self._lock = Lock()
        self._enabled = enabled
        self._rate_per_hour = rate_per_hour

    @property
    def enabled(self) -> bool:
        """Check if rate limiting is currently enabled."""
        with self._lock:
            return self._enabled

    @property
    def rate_per_hour(self) -> int:
        """Get current rate limit per hour."""
        with self._lock:
            return self._rate_per_hour

    def get_settings(self) -> dict:
        """
        Get atomic snapshot of current rate limit settings.

        Returns:
            Dictionary with enabled and rate_per_hour keys
        """
        with self._lock:
            return {"enabled": self._enabled, "rate_per_hour": self._rate_per_hour}

    def update_settings(self, enabled: bool | None = None, rate_per_hour: int | None = None):
        """
        Atomically update rate limit settings.

        Args:
            enabled: Enable or disable rate limiting (optional)
            rate_per_hour: New rate limit per hour (optional, must be >= 1)

        Raises:
            ValueError: If rate_per_hour is less than 1
        """
        with self._lock:
            if enabled is not None:
                self._enabled = enabled
            if rate_per_hour is not None:
                if rate_per_hour < 1:
                    raise ValueError("Rate limit must be at least 1 request per hour")
                self._rate_per_hour = rate_per_hour


rate_limit_state = RateLimitState()
