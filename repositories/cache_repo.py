"""SQLAlchemy implementation of CacheRepository with context-aware caching."""

from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, desc
from models.models import CachedAnswer
import json


class SQLAlchemyCacheRepository:
    """
    Repository for cache operations with context-aware keys and TTL support.

    Key changes from previous version:
    - Uses cache_key (SHA256 hash) for lookups instead of raw question
    - Supports TTL with expires_at field
    - Tracks cache_type (knowledge vs conversational)
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_cache_by_key(self, cache_key: str) -> Optional[dict]:
        """
        Get cache entry by context-aware key (SHA256 hash).

        This is the primary lookup method for context-aware caching.

        Args:
            cache_key: SHA256 hash of (context || question)

        Returns:
            Cache entry dict or None
        """
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

    async def get_cache_by_question(self, question: str) -> Optional[dict]:
        """
        Get exact match for question text.

        Note: This is now secondary to get_cache_by_key for context-aware caching.
        Kept for backwards compatibility and similarity matching.

        Args:
            question: Original question text

        Returns:
            Cache entry dict or None
        """
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
        """
        Get all cache entries for similarity comparison.

        Returns:
            List of cache entry dicts with fields needed for similarity matching
        """
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
        expires_at: Optional[datetime] = None,
        context_preview: Optional[str] = None
    ) -> int:
        """
        Create new cache entry with context-aware key and TTL.

        Args:
            cache_key: SHA256 hash of (context || question)
            question: Original question text
            tfidf_vector: Serialized TF-IDF vector
            answer: First answer variation
            cache_type: "knowledge" or "conversational"
            expires_at: When this entry expires
            context_preview: Truncated context for debugging

        Returns:
            Created cache entry ID
        """
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
        """
        Add answer variation to existing cache entry (max 3).

        Args:
            cache_id: Cache entry ID
            answer: New answer variation
        """
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
        """
        Get next answer variation and rotate index.

        Also updates hit_count and last_used timestamp.

        Args:
            cache_id: Cache entry ID

        Returns:
            Answer variation string
        """
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

    async def delete_expired(self) -> int:
        """
        Delete all expired cache entries.

        Returns:
            Number of entries deleted
        """
        result = await self.session.execute(
            delete(CachedAnswer).where(
                CachedAnswer.expires_at < datetime.utcnow()
            )
        )
        await self.session.commit()
        return result.rowcount

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
        """
        List cache entries with pagination.

        Args:
            page: Page number (1-indexed)
            limit: Entries per page
            sort_by: Field to sort by (hit_count, created_at, last_used, expires_at)
            order: Sort order (asc, desc)

        Returns:
            Dict with entries, total count, and pagination info
        """
        offset = (page - 1) * limit

        # Get total count
        count_result = await self.session.execute(select(func.count(CachedAnswer.id)))
        total = count_result.scalar()

        # Build query with sorting
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
