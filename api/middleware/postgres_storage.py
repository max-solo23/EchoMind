"""PostgreSQL async storage backend for SlowAPI rate limiting."""

import logging
import time

from limits.aio.storage import Storage
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from api.dependencies import get_config
from database import get_session
from models.models import RateLimit


logger = logging.getLogger(__name__)


class PostgresStorage(Storage):
    STORAGE_SCHEME = ["async+postgresql"]

    def __init__(self, uri=None, wrap_exceptions=False, **options):
        super().__init__(uri, wrap_exceptions, **options)

    @property
    def base_exceptions(self) -> tuple[type[Exception], ...]:
        return (SQLAlchemyError,)

    async def incr(self, key: str, expiry: int, amount: int = 1) -> int:
        try:
            expiry_timestamp = int(time.time()) + expiry

            upsert_query = insert(RateLimit).values(key=key, count=amount, expiry=expiry_timestamp)
            upsert_query = upsert_query.on_conflict_do_update(
                index_elements=["key"],
                set_={"count": RateLimit.count + amount, "expiry": expiry_timestamp},
            )

            async with get_session(get_config()) as session:
                await session.execute(upsert_query)
                await session.commit()

                result = await session.execute(select(RateLimit.count).where(RateLimit.key == key))
                new_count = result.scalar_one()
                return new_count
        except Exception as error:
            logger.warning(f"Rate limit DB failed, allowing request: {error}")
            return 1

    async def get(self, key: str) -> int:
        try:
            current_time = int(time.time())

            async with get_session(get_config()) as database:
                query = select(RateLimit).where(RateLimit.key == key)
                result = await database.execute(query)
                rate_limit = result.scalar_one_or_none()

                if rate_limit is None:
                    return 0

                if rate_limit.expiry < current_time:
                    return 0

                return rate_limit.count
        except Exception as error:
            logger.warning(f"Rate limit get failed: {error}")
            return 0

    async def get_expiry(self, key: str) -> int:
        try:
            async with get_session(get_config()) as database:
                query = select(RateLimit.expiry).where(RateLimit.key == key)
                result = await database.execute(query)
                expiry = result.scalar_one_or_none()

                if expiry is None:
                    return -1

                return expiry
        except Exception as error:
            logger.warning(f"Rate limit get_expiry failed: {error}")
            return -1

    async def check(self) -> bool:
        try:
            async with get_session(get_config()) as database:
                await database.execute(select(1))
                return True
        except Exception as error:
            logger.warning(f"Rate limit storage check failed: {error}")
            return False

    async def reset(self) -> int:
        try:
            async with get_session(get_config()) as database:
                result = await database.execute(delete(RateLimit))
                await database.commit()
                return result.rowcount  # type: ignore[no-any-return, attr-defined]
        except Exception as error:
            logger.warning(f"Rate limit reset failed: {error}")
            return 0

    async def clear(self, key: str) -> None:
        try:
            async with get_session(get_config()) as database:
                await database.execute(delete(RateLimit).where(RateLimit.key == key))
                await database.commit()
        except Exception as error:
            logger.warning(f"Rate limit clear failed: {error}")
