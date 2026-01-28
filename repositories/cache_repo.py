import json
from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import CachedAnswer


class SQLAlchemyCacheRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_cache_by_key(self, cache_key: str) -> dict | None:
        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.cache_key == cache_key)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return {
            "id": cache.id,
            "cache_key": cache.cache_key,
            "question": cache.question,
            "context_preview": cache.context_preview,
            "tfidf_vector": cache.tfidf_vector,
            "variations": json.loads(cache.variations),
            "variation_index": cache.variation_index,
            "cache_type": cache.cache_type,
            "expires_at": cache.expires_at,
            "hit_count": cache.hit_count,
            "created_at": cache.created_at,
            "last_used": cache.last_used,
        }

    async def get_cache_by_question(self, question: str) -> dict | None:
        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.question == question)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return {
            "id": cache.id,
            "cache_key": cache.cache_key,
            "question": cache.question,
            "context_preview": cache.context_preview,
            "tfidf_vector": cache.tfidf_vector,
            "variations": json.loads(cache.variations),
            "variation_index": cache.variation_index,
            "cache_type": cache.cache_type,
            "expires_at": cache.expires_at,
            "hit_count": cache.hit_count,
        }

    async def get_all_cached_questions(self) -> list[dict]:
        result = await self.session.execute(select(CachedAnswer))
        caches = result.scalars().all()

        return [
            {
                "id": cache.id,
                "cache_key": cache.cache_key,
                "question": cache.question,
                "tfidf_vector": cache.tfidf_vector,
                "variations": json.loads(cache.variations),
                "variation_index": cache.variation_index,
                "cache_type": cache.cache_type,
                "expires_at": cache.expires_at,
            }
            for cache in caches
        ]

    async def create_cache(
        self,
        cache_key: str,
        question: str,
        tfidf_vector: str,
        answer: str,
        cache_type: str = "knowledge",
        expires_at: datetime | None = None,
        context_preview: str | None = None,
    ) -> int:
        cache = CachedAnswer(
            cache_key=cache_key,
            question=question,
            context_preview=context_preview,
            tfidf_vector=tfidf_vector,
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

    async def get_next_variation(self, cache_id: int) -> str:
        result = await self.session.execute(select(CachedAnswer).where(CachedAnswer.id == cache_id))
        cache = result.scalar_one_or_none()

        if not cache:
            return ""

        variations: list[str] = json.loads(cache.variations)
        current_index = cache.variation_index

        answer = variations[current_index]

        cache.variation_index = (current_index + 1) % len(variations)
        cache.hit_count += 1
        cache.last_used = datetime.utcnow()

        await self.session.commit()

        return answer

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
                {
                    "id": c.id,
                    "cache_key": c.cache_key,
                    "question": c.question,
                    "context_preview": c.context_preview,
                    "variations": json.loads(c.variations),
                    "variation_index": c.variation_index,
                    "cache_type": c.cache_type,
                    "expires_at": c.expires_at,
                    "hit_count": c.hit_count,
                    "created_at": c.created_at,
                    "last_used": c.last_used,
                }
                for c in caches
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

        return {
            "id": cache.id,
            "cache_key": cache.cache_key,
            "question": cache.question,
            "context_preview": cache.context_preview,
            "tfidf_vector": cache.tfidf_vector,
            "variations": json.loads(cache.variations),
            "variation_index": cache.variation_index,
            "cache_type": cache.cache_type,
            "expires_at": cache.expires_at,
            "hit_count": cache.hit_count,
            "created_at": cache.created_at,
            "last_used": cache.last_used,
        }

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

        return [
            {
                "id": c.id,
                "cache_key": c.cache_key,
                "question": c.question,
                "context_preview": c.context_preview,
                "cache_type": c.cache_type,
                "expires_at": c.expires_at,
                "hit_count": c.hit_count,
                "last_used": c.last_used,
            }
            for c in caches
        ]
