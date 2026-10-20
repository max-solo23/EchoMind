import json
from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, delete, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import CachedAnswer


FULL_CACHE_FIELDS = (
    "id",
    "cache_key",
    "question",
    "context_preview",
    "variations",
    "variation_index",
    "cache_type",
    "expires_at",
    "hit_count",
    "created_at",
    "last_used",
)
QUESTION_CACHE_FIELDS = FULL_CACHE_FIELDS[:-2]
ALL_QUESTIONS_CACHE_FIELDS = (
    "id",
    "cache_key",
    "question",
    "variations",
    "variation_index",
    "cache_type",
    "expires_at",
)
LIST_ENTRY_CACHE_FIELDS = FULL_CACHE_FIELDS
SEARCH_CACHE_FIELDS = (
    "id",
    "cache_key",
    "question",
    "context_preview",
    "cache_type",
    "expires_at",
    "hit_count",
    "last_used",
)


class SQLAlchemyCacheRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _cache_to_dict(cache: CachedAnswer, fields: tuple[str, ...]) -> dict:
        return {
            field: json.loads(cache.variations) if field == "variations" else getattr(cache, field)
            for field in fields
        }
    async def get_cache_by_key(self, cache_key: str) -> dict | None:
        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.cache_key == cache_key)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return self._cache_to_dict(cache, FULL_CACHE_FIELDS)

    async def get_cache_by_question(self, question: str) -> dict | None:
        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.question == question)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return self._cache_to_dict(cache, QUESTION_CACHE_FIELDS)

    async def get_all_cached_questions(self) -> list[dict]:
        result = await self.session.execute(select(CachedAnswer))
        caches = result.scalars().all()

        return [self._cache_to_dict(cache, ALL_QUESTIONS_CACHE_FIELDS) for cache in caches]

    async def create_cache(
        self,
        cache_key: str,
        question: str,
        persona_hash: str,
        answer: str,
        cache_type: str = "knowledge",
        expires_at: datetime | None = None,
        context_preview: str | None = None,
    ) -> int:
        cache = CachedAnswer(
            cache_key=cache_key,
            question=question,
            context_preview=context_preview,
            persona_hash=persona_hash,
            variations=json.dumps([answer]),
            variation_index=0,
            cache_type=cache_type,
            expires_at=expires_at,
            hit_count=0,
        )

        self.session.add(cache)
        await self.session.commit()
        await self.session.refresh(cache)

        return cache.id

    async def add_variation(self, cache_id: int, answer: str) -> None:
        result = await self.session.execute(select(CachedAnswer).where(CachedAnswer.id == cache_id))
        cache = result.scalar_one_or_none()

        if not cache:
            return

        variations = json.loads(cache.variations)

        if len(variations) < 3:
            variations.append(answer)
            cache.variations = json.dumps(variations)
            await self.session.commit()

    async def record_hit(self, cache_id: int, variation_count: int) -> None:
        """Advance the rotation and count the hit in one atomic UPDATE (safe across machines)."""
        await self.session.execute(
            update(CachedAnswer)
            .where(CachedAnswer.id == cache_id)
            .values(
                variation_index=(CachedAnswer.variation_index + 1) % variation_count,
                hit_count=CachedAnswer.hit_count + 1,
                last_used=datetime.utcnow(),
            )
        )
        await self.session.commit()

    async def get_knowledge_questions(self, persona_hash: str) -> list[tuple[int, str]]:
        """Live first-turn questions answered under the current persona: the similarity corpus."""
        result = await self.session.execute(
            select(CachedAnswer.id, CachedAnswer.question).where(
                CachedAnswer.cache_type == "knowledge",
                CachedAnswer.persona_hash == persona_hash,
                or_(CachedAnswer.expires_at.is_(None), CachedAnswer.expires_at > datetime.utcnow()),
            )
        )
        return [(row.id, row.question) for row in result]

    async def delete_expired(self) -> int:
        result = cast(
            "CursorResult[tuple[()]]",
            await self.session.execute(
                delete(CachedAnswer).where(CachedAnswer.expires_at < datetime.utcnow())
            ),
        )
        await self.session.commit()
        return result.rowcount or 0

    async def clear_all_cache(self) -> int:
        result = cast(
            "CursorResult[tuple[()]]",
            await self.session.execute(delete(CachedAnswer)),
        )
        await self.session.commit()
        return result.rowcount or 0

    async def list_cache_entries(
        self, page: int = 1, limit: int = 20, sort_by: str = "last_used", order: str = "desc"
    ) -> dict:
        offset = (page - 1) * limit

        count_result = await self.session.execute(select(func.count(CachedAnswer.id)))
        total = count_result.scalar()

        query = select(CachedAnswer)

        sort_columns = {
            "hit_count": CachedAnswer.hit_count,
            "created_at": CachedAnswer.created_at,
            "last_used": CachedAnswer.last_used,
            "expires_at": CachedAnswer.expires_at,
            "cache_type": CachedAnswer.cache_type,
        }
        sort_col = sort_columns.get(sort_by, CachedAnswer.last_used)

        if order == "desc":
            query = query.order_by(desc(sort_col).nulls_last())
        else:
            query = query.order_by(sort_col.nulls_last())

        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        caches = result.scalars().all()

        return {
            "entries": [
                self._cache_to_dict(cache, LIST_ENTRY_CACHE_FIELDS)
                for cache in caches
            ],
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit if total else 0,
        }

    async def get_cache_by_id(self, cache_id: int) -> dict | None:
        result = await self.session.execute(select(CachedAnswer).where(CachedAnswer.id == cache_id))
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return self._cache_to_dict(cache, FULL_CACHE_FIELDS)

    async def delete_cache_by_id(self, cache_id: int) -> bool:
        result = cast(
            "CursorResult[tuple[()]]",
            await self.session.execute(delete(CachedAnswer).where(CachedAnswer.id == cache_id)),
        )
        await self.session.commit()
        return (result.rowcount or 0) > 0

    async def update_cache_variations(self, cache_id: int, variations: list[str]) -> bool:
        result = await self.session.execute(select(CachedAnswer).where(CachedAnswer.id == cache_id))
        cache = result.scalar_one_or_none()

        if not cache:
            return False

        variations = variations[:3]
        cache.variations = json.dumps(variations)
        cache.variation_index = 0

        await self.session.commit()
        return True

    async def search_cache(self, query: str, limit: int = 20) -> list[dict]:
        result = await self.session.execute(
            select(CachedAnswer)
            .where(CachedAnswer.question.ilike(f"%{query}%"))
            .order_by(desc(CachedAnswer.hit_count))
            .limit(limit)
        )
        caches = result.scalars().all()

        return [self._cache_to_dict(cache, SEARCH_CACHE_FIELDS) for cache in caches]
