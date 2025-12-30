"""SQLAlchemy implementation of CacheRepository."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, desc
from models.models import CachedAnswer
import json
from datetime import datetime


class SQLAlchemyCacheRepository:
    """Concrete implementation for cache operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_cache_by_question(self, question: str) -> Optional[dict]:
        """Get exact match for question."""

        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.question == question)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return {
            "id": cache.id,
            "question": cache.question,
            "tfidf_vector": cache.tfidf_vector,
            "variations": json.loads(cache.variations),
            "variation_index": cache.variation_index,
            "hit_count": cache.hit_count
        }

    async def get_all_cached_questions(self) -> list[dict]:
        """Get all for similarity comparison."""

        result = await self.session.execute(select(CachedAnswer))
        caches = result.scalars().all()

        return [
            {
                "id": cache.id,
                "question": cache.question,
                "tfidf_vector": cache.tfidf_vector,
                "variations": json.loads(cache.variations),
                "variation_index": cache.variation_index
            }
            for cache in caches
        ]

    async def create_cache(self, question: str, tfidf_vector: str, answer: str) -> int:
        """Create new cache with first variation."""

        cache = CachedAnswer(
            question=question,
            tfidf_vector=tfidf_vector,
            variations=json.dumps([answer]),  # First variation
            variation_index=0,
            hit_count=0
        )

        self.session.add(cache)
        await self.session.commit()
        await self.session.refresh(cache)

        return cache.id

    async def add_variation(self, cache_id: int, answer: str) -> None:
        """Add variation (max 3)."""

        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.id == cache_id)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return

        variations = json.loads(cache.variations)

        if len(variations) < 3:
            variations.append(answer)
            cache.variations = json.dumps(variations)
            await self.session.commit()

    async def get_next_variation(self, cache_id: int) -> str:
        """Get next variation and rotate index."""

        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.id == cache_id)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return ""

        variations = json.loads(cache.variations)
        current_index = cache.variation_index

        # Get answer at current index
        answer = variations[current_index]

        # Rotate to next index (0 -> 1 -> 2 -> 0)
        cache.variation_index = (current_index + 1) % len(variations)
        cache.hit_count += 1
        cache.last_used = datetime.utcnow()

        await self.session.commit()

        return answer

    async def clear_all_cache(self) -> int:
        """Delete all cached answers."""

        result = await self.session.execute(delete(CachedAnswer))
        await self.session.commit()

        return result.rowcount

    async def list_cache_entries(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "last_used",
        order: str = "desc"
    ) -> dict:
        """List cache entries with pagination."""

        offset = (page - 1) * limit

        # Get total count
        count_result = await self.session.execute(select(func.count(CachedAnswer.id)))
        total = count_result.scalar()

        # Build query with sorting
        query = select(CachedAnswer)

        if sort_by == "hit_count":
            sort_col = CachedAnswer.hit_count
        elif sort_by == "created_at":
            sort_col = CachedAnswer.created_at
        elif sort_by == "last_used":
            sort_col = CachedAnswer.last_used
        else:
            sort_col = CachedAnswer.last_used

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
                    "question": c.question,
                    "variations": json.loads(c.variations),
                    "variation_index": c.variation_index,
                    "hit_count": c.hit_count,
                    "created_at": c.created_at,
                    "last_used": c.last_used
                }
                for c in caches
            ],
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit if total else 0
        }

    async def get_cache_by_id(self, cache_id: int) -> Optional[dict]:
        """Get single cache entry by ID."""

        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.id == cache_id)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return None

        return {
            "id": cache.id,
            "question": cache.question,
            "tfidf_vector": cache.tfidf_vector,
            "variations": json.loads(cache.variations),
            "variation_index": cache.variation_index,
            "hit_count": cache.hit_count,
            "created_at": cache.created_at,
            "last_used": cache.last_used
        }

    async def delete_cache_by_id(self, cache_id: int) -> bool:
        """Delete single cache entry by ID."""

        result = await self.session.execute(
            delete(CachedAnswer).where(CachedAnswer.id == cache_id)
        )
        await self.session.commit()

        return result.rowcount > 0

    async def update_cache_variations(self, cache_id: int, variations: list[str]) -> bool:
        """Update cache variations (max 3)."""

        result = await self.session.execute(
            select(CachedAnswer).where(CachedAnswer.id == cache_id)
        )
        cache = result.scalar_one_or_none()

        if not cache:
            return False

        # Enforce max 3 variations
        variations = variations[:3]
        cache.variations = json.dumps(variations)
        cache.variation_index = 0  # Reset rotation

        await self.session.commit()
        return True

    async def search_cache(self, query: str, limit: int = 20) -> list[dict]:
        """Search cache entries by question text (case-insensitive)."""

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
                "question": c.question,
                "hit_count": c.hit_count,
                "last_used": c.last_used
            }
            for c in caches
        ]
