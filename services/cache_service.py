"""
Cache Service for intelligent answer caching with context awareness.

This service orchestrates:
- Role-aware input filtering (denylist + min tokens)
- Context-aware cache keys (hash of context + question)
- TF-IDF similarity matching
- TTL-based expiration (knowledge: 30 days, conversational: 24 hours)
- Answer variation rotation

Design decisions:
- Short/filler inputs are never cached (prevents cross-context contamination)
- Cache key includes last assistant message for context sensitivity
- Different TTLs for knowledge vs conversational cache types
"""

import hashlib
from datetime import datetime, timedelta
from enum import Enum

from repositories.cache_repo import SQLAlchemyCacheRepository

from .similarity_service import SimilarityService


class CacheType(str, Enum):
    """Cache entry classification for TTL purposes."""
    KNOWLEDGE = "knowledge"          # Standalone questions (30 day TTL)
    CONVERSATIONAL = "conversational"  # Context-dependent replies (24 hour TTL)


# TTL configuration
CACHE_TTL = {
    CacheType.KNOWLEDGE: timedelta(days=30),
    CacheType.CONVERSATIONAL: timedelta(hours=24),
}

# Low-information inputs that should never be cached when in a conversation
# These are only skipped when is_continuation=True (not first turn)
CACHE_DENYLIST = {
    # Acknowledgements
    "ok", "okay", "yes", "no", "yeah", "yep", "nope", "yea", "nah",
    # Thanks
    "thanks", "thank you", "thx", "ty", "thank", "thankyou",
    # Continuations
    "continue", "go on", "go ahead", "more", "next",
    # Confirmations
    "sure", "alright", "right", "got it", "understood", "i understand",
    # Reactions
    "cool", "nice", "great", "awesome", "perfect", "good", "fine",
    # Fillers
    "hmm", "hm", "ah", "oh", "i see", "uh", "um", "wow",
    # Short responses
    "k", "kk", "ya", "ye", "na", "lol", "haha",
}

# Minimum token count for caching (unless it's a question)
MIN_TOKENS_FOR_CACHE = 4


class CacheService:
    """
    Orchestrates caching logic with context awareness and TTL.

    Flow:
    1. Check if input should be cached (filter by denylist/tokens)
    2. Build context-aware cache key (hash of context + question)
    3. Check for cache hit (exact key match, then similarity)
    4. Respect TTL - delete expired entries on access
    5. Store new entries with appropriate TTL
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

    def should_skip_cache(
        self,
        message: str,
        is_continuation: bool = False
    ) -> bool:
        """
        Determine if message should bypass cache entirely.

        Args:
            message: User's message
            is_continuation: True if this follows a previous exchange
                            (i.e., conversation history is not empty)

        Returns:
            True if message should skip cache (no lookup, no storage)

        Rules:
        - Denylist only applies to continuation turns (not first message)
        - Questions (contains ?) are always allowed regardless of length
        - Very short messages (< 2 tokens) are always skipped
        - Messages < MIN_TOKENS are skipped unless they're questions
        """
        normalized = message.lower().strip()
        tokens = normalized.split()
        token_count = len(tokens)

        # Rule 1: Questions are always cacheable (contains ?)
        if "?" in message:
            return False

        # Rule 2: Very short messages are always skipped
        if token_count < 2:
            return True

        # Rule 3: Denylist check (only for continuations)
        if is_continuation and normalized in CACHE_DENYLIST:
            return True

        # Rule 4: Short messages below threshold
        return token_count < MIN_TOKENS_FOR_CACHE

    def get_cache_type(self, is_continuation: bool) -> CacheType:
        """
        Determine cache type based on conversation state.

        Args:
            is_continuation: True if conversation has history

        Returns:
            CacheType.CONVERSATIONAL for continuations
            CacheType.KNOWLEDGE for first-turn/standalone questions
        """
        if is_continuation:
            return CacheType.CONVERSATIONAL
        return CacheType.KNOWLEDGE

    def calculate_expiry(self, cache_type: CacheType) -> datetime:
        """
        Calculate expiration timestamp for cache entry.

        Args:
            cache_type: Type of cache entry

        Returns:
            datetime when the entry should expire
        """
        return datetime.utcnow() + CACHE_TTL[cache_type]

    def build_cache_key(
        self,
        message: str,
        last_assistant_message: str | None = None
    ) -> str:
        """
        Build context-aware cache key.

        Key = SHA256(last_assistant_message || user_message)

        This ensures the same user message after different assistant
        responses produces different cache keys.

        Args:
            message: User's message
            last_assistant_message: Previous assistant response (if any)

        Returns:
            64-character hex SHA256 hash
        """
        context = last_assistant_message or ""
        combined = f"{context}||{message}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get_cached_answer(
        self,
        message: str,
        last_assistant_message: str | None = None,
        is_continuation: bool = False
    ) -> str | None:
        """
        Try to find a cached answer for the message with context awareness.

        Args:
            message: User's message
            last_assistant_message: Previous assistant response for context
            is_continuation: True if conversation has history

        Returns:
            Cached answer if found and not expired, None otherwise
        """
        # Gate 1: Skip cache for low-information inputs
        if self.should_skip_cache(message, is_continuation):
            return None

        # Gate 2: Build context-aware key
        cache_key = self.build_cache_key(message, last_assistant_message)

        # Try exact key match first (fast path)
        exact_match = await self.cache_repo.get_cache_by_key(cache_key)
        if exact_match:
            # Check expiration
            if exact_match.get("expires_at") and exact_match["expires_at"] < datetime.utcnow():
                # Expired - delete and continue to similarity check
                await self.cache_repo.delete_cache_by_id(exact_match["id"])
            else:
                return await self.cache_repo.get_next_variation(exact_match["id"])

        # Try similarity matching (slower path)
        # Only match on question text, but still requires same context for storage
        all_cached = await self.cache_repo.get_all_cached_questions()
        if not all_cached:
            return None

        # Filter out expired entries
        valid_cached = [
            c for c in all_cached
            if not c.get("expires_at") or c["expires_at"] >= datetime.utcnow()
        ]

        if not valid_cached:
            return None

        # Fit vectorizer on existing questions for consistency
        questions = [c["question"] for c in valid_cached]
        self.similarity.fit_on_corpus(questions + [message])

        # Find best match
        best_match = self.similarity.find_best_match(message, valid_cached)
        if best_match:
            return await self.cache_repo.get_next_variation(best_match["id"])

        return None

    async def cache_answer(
        self,
        message: str,
        answer: str,
        last_assistant_message: str | None = None,
        is_continuation: bool = False
    ) -> int | None:
        """
        Cache a new question-answer pair with context awareness and TTL.

        Args:
            message: User's message
            answer: Bot's answer
            last_assistant_message: Previous assistant response for context
            is_continuation: True if conversation has history

        Returns:
            Cache entry ID if cached, None if skipped
        """
        # Don't cache low-information inputs
        if self.should_skip_cache(message, is_continuation):
            return None

        cache_key = self.build_cache_key(message, last_assistant_message)
        cache_type = self.get_cache_type(is_continuation)
        expires_at = self.calculate_expiry(cache_type)

        # Truncate context for debugging (first 200 chars)
        context_preview = None
        if last_assistant_message:
            context_preview = last_assistant_message[:200]

        # Check if this exact cache key already exists
        existing = await self.cache_repo.get_cache_by_key(cache_key)

        if existing:
            # Add as variation (if under 3)
            await self.cache_repo.add_variation(existing["id"], answer)
            return existing["id"]

        # Create new cache entry
        tfidf_vector = self.similarity.vectorize(message)
        return await self.cache_repo.create_cache(
            cache_key=cache_key,
            question=message,
            tfidf_vector=tfidf_vector,
            answer=answer,
            cache_type=cache_type.value,
            expires_at=expires_at,
            context_preview=context_preview
        )

    async def should_cache(
        self,
        message: str,
        is_continuation: bool = False
    ) -> bool:
        """
        Determine if a question should be cached.

        Returns False if:
        - Input is in denylist (for continuations)
        - Input is too short
        - A similar question already exists

        Args:
            message: Question to check
            is_continuation: True if conversation has history

        Returns:
            True if question should be cached, False otherwise
        """
        # First check filter rules
        if self.should_skip_cache(message, is_continuation):
            return False

        # Then check similarity
        all_cached = await self.cache_repo.get_all_cached_questions()
        if not all_cached:
            return True

        # Fit vectorizer
        questions = [c["question"] for c in all_cached]
        self.similarity.fit_on_corpus(questions + [message])

        # Check for similar question
        best_match = self.similarity.find_best_match(message, all_cached)
        return best_match is None

    async def clear_cache(self) -> int:
        """
        Clear all cached answers.

        Returns:
            Number of entries deleted
        """
        return await self.cache_repo.clear_all_cache()

    async def cleanup_expired(self) -> int:
        """
        Delete all expired cache entries.

        Returns:
            Number of entries deleted
        """
        return await self.cache_repo.delete_expired()

    async def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics including TTL breakdown
        """
        all_cached = await self.cache_repo.get_all_cached_questions()

        total_questions = len(all_cached)
        total_variations = sum(len(c.get("variations", [])) for c in all_cached)

        # Count by type
        knowledge_count = sum(1 for c in all_cached if c.get("cache_type") == "knowledge")
        conversational_count = sum(1 for c in all_cached if c.get("cache_type") == "conversational")

        # Count expired
        now = datetime.utcnow()
        expired_count = sum(
            1 for c in all_cached
            if c.get("expires_at") and c["expires_at"] < now
        )

        return {
            "total_questions": total_questions,
            "total_variations": total_variations,
            "avg_variations_per_question": total_variations / total_questions if total_questions > 0 else 0,
            "knowledge_entries": knowledge_count,
            "conversational_entries": conversational_count,
            "expired_entries": expired_count,
        }

    # Admin methods

    async def list_cache_entries(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "last_used",
        order: str = "desc"
    ) -> dict:
        """List cache entries with pagination."""
        return await self.cache_repo.list_cache_entries(page, limit, sort_by, order)

    async def get_cache_by_id(self, cache_id: int) -> dict | None:
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
