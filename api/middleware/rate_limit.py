import logging
import time
from typing import cast

from fastapi import HTTPException, Request
from sqlalchemy import case
from sqlalchemy.dialects.postgresql import insert

from api.dependencies import get_config
from api.middleware.rate_limit_state import rate_limit_state
from models.models import RateLimit
from repositories.connection import get_session


logger = logging.getLogger(__name__)

EXPIRY_SECONDS = 3600


async def check_rate_limit(request: Request) -> None:
    if not rate_limit_state.enabled:
        return

    ip = _get_client_ip(request)
    key = f"rate_limit:{ip}"

    try:
        count = await _increment_counter(key)
        if count > rate_limit_state.rate_per_hour:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(EXPIRY_SECONDS)},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limit check failed, allowing request: {e}")


async def _increment_counter(key: str) -> int:
    config = get_config()
    now_timestamp = int(time.time())
    expiry_timestamp = now_timestamp + EXPIRY_SECONDS

    async with get_session(config) as session:
        stmt = _build_increment_statement(key, now_timestamp, expiry_timestamp)
        result = await session.execute(stmt)
        await session.commit()
        return cast("int", result.scalar_one())


def _build_increment_statement(key: str, now_timestamp: int, expiry_timestamp: int):
    stmt = insert(RateLimit).values(key=key, count=1, expiry=expiry_timestamp)
    is_expired = RateLimit.expiry <= now_timestamp
    return stmt.on_conflict_do_update(
        index_elements=["key"],
        set_={
            "count": case((is_expired, 1), else_=RateLimit.count + 1),
            "expiry": case((is_expired, expiry_timestamp), else_=RateLimit.expiry),
        },
    ).returning(RateLimit.count)


def _get_client_ip(request: Request) -> str:
    fly_client_ip = request.headers.get("fly-client-ip")
    if fly_client_ip and fly_client_ip.strip():
        return fly_client_ip.strip()

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for and forwarded_for.split(",", 1)[0].strip():
        return forwarded_for.split(",", 1)[0].strip()

    return request.client.host if request.client else "unknown"
