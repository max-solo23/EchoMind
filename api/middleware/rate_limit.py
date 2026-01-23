import logging
import time

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from api.dependencies import get_config
from api.middleware.rate_limit_state import rate_limit_state
from database import get_session
from models.models import RateLimit

logger = logging.getLogger(__name__)

EXPIRY_SECONDS = 3600


async def check_rate_limit(request: Request) -> None:
    if not rate_limit_state.enabled:
        return

    ip = request.client.host if request.client else "unknown"
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
    expiry_timestamp = int(time.time()) + EXPIRY_SECONDS

    async with get_session(config) as session:
        stmt = insert(RateLimit).values(key=key, count=1, expiry=expiry_timestamp)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"],
            set_={"count": RateLimit.count + 1, "expiry": expiry_timestamp},
        )
        await session.execute(stmt)
        await session.commit()

        result = await session.execute(select(RateLimit.count).where(RateLimit.key == key))
        return result.scalar_one()
