"""SQLAlchemy implementation of CacheRepository."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
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
