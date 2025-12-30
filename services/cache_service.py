"""
Cache Service for intelligent answer caching.

This service orchestrates:
- TF-IDF similarity matching
- Cache repository operations
- Answer variation rotation

Design notes:
- Uses 90% similarity threshold
- Stores up to 3 answer variations per question
- Rotates through variations to avoid repetition
"""

from typing import Optional
from .similarity_service import SimilarityService
from repositories.cache_repo import SQLAlchemyCacheRepository


class CacheService:
    """
    Orchestrates caching logic with similarity matching.

    Flow:
    1. Check for similar question in cache
    2. If found (>90% match): return cached answer variation
    3. If not found: store question + answer for future use
    """

    def __init__(
        self,
        cache_repo: SQLAlchemyCacheRepository,
        similarity_service: SimilarityService
    ):
        """
        Initialize with dependencies.

        Args:
            cache_repo: Repository for cache operations
            similarity_service: Service for TF-IDF matching
        """
        self.cache_repo = cache_repo
        self.similarity = similarity_service

    async def get_cached_answer(self, question: str) -> Optional[str]:
        """
        Try to find a cached answer for the question.

        First checks for exact match, then uses similarity matching.

        Args:
            question: User's question

        Returns:
            Cached answer if found, None otherwise
        """
        # First try exact match (fast path)
        exact_match = await self.cache_repo.get_cache_by_question(question)
        if exact_match:
            return await self.cache_repo.get_next_variation(exact_match["id"])

        # Try similarity matching
        all_cached = await self.cache_repo.get_all_cached_questions()
        if not all_cached:
            return None

        # Fit vectorizer on existing questions for consistency
        questions = [c["question"] for c in all_cached]
        self.similarity.fit_on_corpus(questions + [question])

        # Find best match
        best_match = self.similarity.find_best_match(question, all_cached)
        if best_match:
            return await self.cache_repo.get_next_variation(best_match["id"])

        return None

    async def cache_answer(self, question: str, answer: str) -> int:
        """
        Cache a new question-answer pair.

        If question already exists (exact match), adds as variation.
        Otherwise creates new cache entry.

        Args:
            question: User's question
            answer: Bot's answer

        Returns:
            Cache entry ID
        """
        # Check if question already cached
        existing = await self.cache_repo.get_cache_by_question(question)

        if existing:
            # Add as variation (if under 3)
            await self.cache_repo.add_variation(existing["id"], answer)
            return existing["id"]

        # Create new cache entry
        tfidf_vector = self.similarity.vectorize(question)
        return await self.cache_repo.create_cache(question, tfidf_vector, answer)

    async def should_cache(self, question: str) -> bool:
        """
        Determine if a question should be cached.

        Returns False if a similar question already exists.
        Use this to avoid duplicate caching.

        Args:
            question: Question to check

        Returns:
            True if question should be cached, False otherwise
        """
        all_cached = await self.cache_repo.get_all_cached_questions()
        if not all_cached:
            return True

        # Fit vectorizer
        questions = [c["question"] for c in all_cached]
        self.similarity.fit_on_corpus(questions + [question])

        # Check for similar question
        best_match = self.similarity.find_best_match(question, all_cached)
        return best_match is None

    async def clear_cache(self) -> int:
        """
        Clear all cached answers.

        Returns:
            Number of entries deleted
        """
        return await self.cache_repo.clear_all_cache()

    async def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        all_cached = await self.cache_repo.get_all_cached_questions()

        total_questions = len(all_cached)
        total_variations = sum(len(c.get("variations", [])) for c in all_cached)

        return {
            "total_questions": total_questions,
            "total_variations": total_variations,
            "avg_variations_per_question": total_variations / total_questions if total_questions > 0 else 0
        }

    # New admin methods

    async def list_cache_entries(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "last_used",
        order: str = "desc"
    ) -> dict:
        """List cache entries with pagination."""
        return await self.cache_repo.list_cache_entries(page, limit, sort_by, order)

    async def get_cache_by_id(self, cache_id: int) -> Optional[dict]:
        """Get single cache entry by ID."""
        return await self.cache_repo.get_cache_by_id(cache_id)

    async def delete_cache_by_id(self, cache_id: int) -> bool:
        """Delete single cache entry by ID."""
        return await self.cache_repo.delete_cache_by_id(cache_id)

    async def update_cache_variations(self, cache_id: int, variations: list[str]) -> bool:
        """Update cache entry variations."""
        return await self.cache_repo.update_cache_variations(cache_id, variations)

    async def search_cache(self, query: str, limit: int = 20) -> list[dict]:
        """Search cache entries by question text."""
        return await self.cache_repo.search_cache(query, limit)
