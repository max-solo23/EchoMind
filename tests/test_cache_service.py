"""Tests for CacheService - intelligent answer caching."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.cache_service import (
    CACHE_DENYLIST,
    CACHE_TTL,
    MIN_TOKENS_FOR_CACHE,
    CacheService,
    CacheType,
)


class TestCacheType:
    """Test CacheType enum and TTL configuration."""

    def test_cache_types_exist(self):
        """
        Verify both cache types are defined.

        Why: System needs two TTL strategies - long for knowledge,
        short for context-dependent responses.
        """
        assert CacheType.KNOWLEDGE.value == "knowledge"
        assert CacheType.CONVERSATIONAL.value == "conversational"

    def test_ttl_configuration(self):
        """
        Verify TTL values are sensible.

        Why: Knowledge (facts) should last longer than conversational
        (context-dependent) responses which may become stale quickly.
        """
        assert CACHE_TTL[CacheType.KNOWLEDGE] == timedelta(days=30)
        assert CACHE_TTL[CacheType.CONVERSATIONAL] == timedelta(hours=24)


class TestShouldSkipCache:
    """Test cache filtering logic."""

    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies."""
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    def test_questions_always_allowed(self, service):
        """
        Verify questions (with ?) are never skipped.

        Why: Questions are high-value cache candidates regardless
        of length. "Why?" is short but worth caching.
        """
        assert service.should_skip_cache("Why?") is False
        assert service.should_skip_cache("?") is False
        assert service.should_skip_cache("ok?", is_continuation=True) is False

    def test_very_short_messages_skipped(self, service):
        """
        Verify single-word non-questions are skipped.

        Why: "hi" or "a" alone provides no meaningful context
        to cache. Would create garbage entries.
        """
        assert service.should_skip_cache("hi") is True
        assert service.should_skip_cache("a") is True

    def test_denylist_only_applies_to_continuations(self, service):
        """
        Verify denylist words pass on first message.

        Why: "ok tell me about Python" is valid first message.
        But "ok" alone after assistant spoke is just acknowledgment.
        """
        # First message - denylist doesn't apply
        assert service.should_skip_cache("ok", is_continuation=False) is True  # Too short
        assert service.should_skip_cache("thanks for that info", is_continuation=False) is False

        # Continuation - denylist applies
        assert service.should_skip_cache("thanks", is_continuation=True) is True
        assert service.should_skip_cache("ok", is_continuation=True) is True

    def test_denylist_words_blocked_in_continuation(self, service):
        """
        Verify all denylist words are blocked in continuations.

        Why: These are low-information acknowledgments that would
        pollute the cache with context-dependent noise.
        """
        for word in ["ok", "thanks", "cool", "nice", "got it", "yes", "no"]:
            if word in CACHE_DENYLIST:
                assert service.should_skip_cache(word, is_continuation=True) is True

    def test_short_messages_below_threshold_skipped(self, service):
        """
        Verify messages below MIN_TOKENS are skipped (unless questions).

        Why: "do it" or "yes please" are too short to be meaningful
        standalone cache entries.
        """
        # Below threshold (4 tokens)
        assert service.should_skip_cache("do it now") is True  # 3 tokens

        # At or above threshold
        assert service.should_skip_cache("tell me about Python programming") is False

    def test_normal_messages_allowed(self, service):
        """
        Verify normal messages pass filtering.

        Why: Most user messages should be cacheable. Only filter
        out the obvious noise.
        """
        assert service.should_skip_cache("What is your experience with Python?") is False
        assert service.should_skip_cache("Tell me about your backend skills") is False


class TestGetCacheType:
    """Test cache type determination."""

    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    def test_first_message_is_knowledge(self, service):
        """
        Verify first messages get KNOWLEDGE type (30 day TTL).

        Why: Standalone questions like "What do you do?" are
        factual and don't depend on prior context.
        """
        assert service.get_cache_type(is_continuation=False) == CacheType.KNOWLEDGE

    def test_continuation_is_conversational(self, service):
        """
        Verify continuations get CONVERSATIONAL type (24 hour TTL).

        Why: Context-dependent responses may become stale as
        conversation patterns change.
        """
        assert service.get_cache_type(is_continuation=True) == CacheType.CONVERSATIONAL


class TestCalculateExpiry:
    """Test expiration calculation."""

    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    def test_knowledge_expires_in_30_days(self, service):
        """
        Verify KNOWLEDGE entries expire in ~30 days.

        Why: Factual responses about persona don't change often.
        30 days balances freshness with cache efficiency.
        """
        before = datetime.utcnow()
        expiry = service.calculate_expiry(CacheType.KNOWLEDGE)
        after = datetime.utcnow()

        # Should be ~30 days from now
        assert expiry >= before + timedelta(days=29, hours=23)
        assert expiry <= after + timedelta(days=30, hours=1)

    def test_conversational_expires_in_24_hours(self, service):
        """
        Verify CONVERSATIONAL entries expire in ~24 hours.

        Why: Context-dependent responses may become stale quickly.
        24 hours is short enough to stay fresh.
        """
        before = datetime.utcnow()
        expiry = service.calculate_expiry(CacheType.CONVERSATIONAL)
        after = datetime.utcnow()

        # Should be ~24 hours from now
        assert expiry >= before + timedelta(hours=23)
        assert expiry <= after + timedelta(hours=25)


class TestBuildCacheKey:
    """Test context-aware cache key generation."""

    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    def test_key_is_sha256_hex(self, service):
        """
        Verify key is 64-character hex string (SHA256).

        Why: SHA256 produces fixed-length, collision-resistant keys
        suitable for database indexing.
        """
        key = service.build_cache_key("What is Python?")

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_message_same_key(self, service):
        """
        Verify deterministic key generation.

        Why: Same input must produce same key for cache hits to work.
        """
        key1 = service.build_cache_key("What is Python?")
        key2 = service.build_cache_key("What is Python?")

        assert key1 == key2

    def test_different_context_different_key(self, service):
        """
        Verify context changes the key.

        Why: "yes" after "Do you know Python?" should have different
        cache key than "yes" after "Are you available?". Prevents
        cross-context contamination.
        """
        key1 = service.build_cache_key("yes", last_assistant_message="Do you know Python?")
        key2 = service.build_cache_key("yes", last_assistant_message="Are you available?")

        assert key1 != key2

    def test_no_context_uses_empty_string(self, service):
        """
        Verify None context is handled (uses empty string).

        Why: First message has no prior context. Should still
        produce valid, consistent key.
        """
        key1 = service.build_cache_key("Hello", last_assistant_message=None)
        key2 = service.build_cache_key("Hello", last_assistant_message="")

        # Both should produce same key (empty context)
        assert key1 == key2


class TestGetCachedAnswer:
    """Test cache lookup with async mocking."""

    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    @pytest.mark.asyncio
    async def test_skips_cache_for_denylist(self, service):
        """
        Verify denylist messages don't hit cache.

        Why: No point querying DB for "ok" or "thanks" -
        we know we won't cache the response anyway.
        """
        result = await service.get_cached_answer("thanks", is_continuation=True)

        assert result is None
        service.cache_repo.get_cache_by_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_cached_answer_on_exact_match(self, service):
        """
        Verify exact key match returns cached answer.

        Why: Fast path - if we have exact key, return immediately
        without expensive similarity search.
        """
        service.cache_repo.get_cache_by_key.return_value = {
            "id": 1,
            "expires_at": datetime.utcnow() + timedelta(days=1),
        }
        service.cache_repo.get_next_variation.return_value = "Cached response"

        result = await service.get_cached_answer("What is Python?")

        assert result == "Cached response"
        service.cache_repo.get_next_variation.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_deletes_expired_exact_match(self, service):
        """
        Verify expired entries are deleted on access.

        Why: Lazy expiration - delete when accessed rather than
        background job. Keeps cache fresh without extra processes.
        """
        service.cache_repo.get_cache_by_key.return_value = {
            "id": 1,
            "expires_at": datetime.utcnow() - timedelta(days=1),  # Expired
        }
        service.cache_repo.get_all_cached_questions.return_value = []

        result = await service.get_cached_answer("What is Python?")

        service.cache_repo.delete_cache_by_id.assert_called_once_with(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cache(self, service):
        """
        Verify None returned when cache is empty.

        Why: No cached questions means no possible match.
        Should handle gracefully.
        """
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = []

        result = await service.get_cached_answer("What is Python?")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_similar_match_when_no_exact(self, service):
        """
        Verify falls back to similarity matching when no exact key match.

        Why: User might rephrase question. Similarity matching finds
        semantically equivalent cached answers.
        """
        from datetime import timedelta

        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = [
            {
                "id": 1,
                "question": "What is Python programming?",
                "tfidf_vector": "[0.5]",
                "expires_at": datetime.utcnow() + timedelta(days=1),
            }
        ]
        service.similarity.find_best_match.return_value = {"id": 1}
        service.cache_repo.get_next_variation.return_value = "Similar answer"

        result = await service.get_cached_answer("What is Python?")

        assert result == "Similar answer"

    @pytest.mark.asyncio
    async def test_filters_expired_in_similarity_match(self, service):
        """
        Verify expired entries are filtered during similarity matching.

        Why: Don't match against expired entries - they're stale.
        """
        from datetime import timedelta

        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = [
            {
                "id": 1,
                "question": "Expired question",
                "tfidf_vector": "[0.5]",
                "expires_at": datetime.utcnow() - timedelta(days=1),  # Expired
            }
        ]

        result = await service.get_cached_answer("What is Python?")

        # No valid entries after filtering expired
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_similar_match(self, service):
        """
        Verify None returned when no similarity match found.

        Why: Question is new and different from all cached questions.
        Should return None so LLM generates fresh response.
        """
        from datetime import timedelta

        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = [
            {
                "id": 1,
                "question": "What is Python?",
                "tfidf_vector": "[0.5]",
                "expires_at": datetime.utcnow() + timedelta(days=1),
            }
        ]
        service.similarity.find_best_match.return_value = None  # No match

        result = await service.get_cached_answer("How do I cook pasta?")

        assert result is None


class TestShouldCache:
    """Test should_cache decision logic."""

    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    @pytest.mark.asyncio
    async def test_returns_false_for_denylist(self, service):
        """
        Verify denylist messages return False.

        Why: should_cache wraps should_skip_cache for API convenience.
        """
        result = await service.should_cache("ok", is_continuation=True)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_new_question(self, service):
        """
        Verify new question returns True when no cache.

        Why: Empty cache means nothing to match against.
        """
        service.cache_repo.get_all_cached_questions.return_value = []

        result = await service.should_cache("What is Python?")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_similar_existing(self, service):
        """
        Verify similar question returns False.

        Why: Don't cache duplicates - use existing cache entry.
        """
        service.cache_repo.get_all_cached_questions.return_value = [
            {"id": 1, "question": "What is Python?", "tfidf_vector": "[0.5]"}
        ]
        service.similarity.find_best_match.return_value = {"id": 1}

        result = await service.should_cache("What is Python programming?")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_different_question(self, service):
        """
        Verify different question returns True.

        Why: New unique question should be cached.
        """
        service.cache_repo.get_all_cached_questions.return_value = [
            {"id": 1, "question": "What is Python?", "tfidf_vector": "[0.5]"}
        ]
        service.similarity.find_best_match.return_value = None

        result = await service.should_cache("How do I learn JavaScript?")

        assert result is True


class TestCacheAnswer:
    """Test storing answers in cache."""

    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        mock_similarity.vectorize.return_value = "[0.5, 0.3, 0.2]"
        return CacheService(mock_repo, mock_similarity)

    @pytest.mark.asyncio
    async def test_skips_denylist_messages(self, service):
        """
        Verify denylist messages aren't cached.

        Why: "ok" or "thanks" responses would pollute cache
        with context-dependent noise.
        """
        result = await service.cache_answer("thanks", "You're welcome!", is_continuation=True)

        assert result is None
        service.cache_repo.create_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_adds_variation_to_existing(self, service):
        """
        Verify duplicate questions add variations (up to 3).

        Why: Same question can have multiple valid answers.
        Rotating variations makes bot feel less robotic.
        """
        service.cache_repo.get_cache_by_key.return_value = {"id": 1}

        result = await service.cache_answer("What is Python?", "A programming language")

        service.cache_repo.add_variation.assert_called_once_with(1, "A programming language")
        assert result == 1

    @pytest.mark.asyncio
    async def test_creates_new_cache_entry(self, service):
        """
        Verify new questions create new cache entries.

        Why: First time seeing this question - store it with
        TTL, type, and TF-IDF vector for similarity matching.
        """
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.create_cache.return_value = 42

        result = await service.cache_answer("What is Python?", "A programming language")

        assert result == 42
        service.cache_repo.create_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_cache_with_context_preview(self, service):
        """
        Verify context_preview is truncated to 200 chars.

        Why: Store context for debugging but don't bloat database
        with full conversation history.
        """
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.create_cache.return_value = 1

        long_context = "A" * 300  # Longer than 200

        await service.cache_answer(
            "Tell me more about Python programming please",  # 6 words, passes MIN_TOKENS
            "Response",
            last_assistant_message=long_context,
        )

        # Verify create_cache was called with truncated context
        call_kwargs = service.cache_repo.create_cache.call_args[1]
        assert len(call_kwargs["context_preview"]) == 200


class TestCacheStats:
    """Test cache statistics."""

    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    @pytest.mark.asyncio
    async def test_returns_stats_dict(self, service):
        """
        Verify stats include all expected fields.

        Why: Admin dashboard needs visibility into cache health -
        how many entries, what types, how many expired.
        """
        service.cache_repo.get_all_cached_questions.return_value = [
            {
                "cache_type": "knowledge",
                "expires_at": datetime.utcnow() + timedelta(days=1),
                "variations": ["a", "b"],
            },
            {
                "cache_type": "conversational",
                "expires_at": datetime.utcnow() - timedelta(days=1),
                "variations": ["c"],
            },
        ]

        stats = await service.get_cache_stats()

        assert stats["total_questions"] == 2
        assert stats["total_variations"] == 3
        assert stats["knowledge_entries"] == 1
        assert stats["conversational_entries"] == 1
        assert stats["expired_entries"] == 1

    @pytest.mark.asyncio
    async def test_handles_empty_cache(self, service):
        """
        Verify stats work with empty cache.

        Why: Fresh system has no cache entries. Should return
        zeros without division errors.
        """
        service.cache_repo.get_all_cached_questions.return_value = []

        stats = await service.get_cache_stats()

        assert stats["total_questions"] == 0
        assert stats["avg_variations_per_question"] == 0


class TestAdminMethods:
    """Test admin/passthrough methods."""

    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity)

    @pytest.mark.asyncio
    async def test_clear_cache_delegates(self, service):
        """Verify clear_cache calls repo."""
        service.cache_repo.clear_all_cache.return_value = 5

        result = await service.clear_cache()

        assert result == 5
        service.cache_repo.clear_all_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_delegates(self, service):
        """Verify cleanup_expired calls repo."""
        service.cache_repo.delete_expired.return_value = 3

        result = await service.cleanup_expired()

        assert result == 3
        service.cache_repo.delete_expired.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_cache_entries_delegates(self, service):
        """Verify list_cache_entries calls repo."""
        service.cache_repo.list_cache_entries.return_value = {"entries": [], "total": 0}

        result = await service.list_cache_entries(page=2, limit=15)

        assert result == {"entries": [], "total": 0}
        service.cache_repo.list_cache_entries.assert_called_once_with(2, 15, "last_used", "desc")

    @pytest.mark.asyncio
    async def test_get_cache_by_id_delegates(self, service):
        """Verify get_cache_by_id calls repo."""
        service.cache_repo.get_cache_by_id.return_value = {"id": 1, "question": "Test"}

        result = await service.get_cache_by_id(1)

        assert result == {"id": 1, "question": "Test"}

    @pytest.mark.asyncio
    async def test_delete_cache_by_id_delegates(self, service):
        """Verify delete_cache_by_id calls repo."""
        service.cache_repo.delete_cache_by_id.return_value = True

        result = await service.delete_cache_by_id(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_update_cache_variations_delegates(self, service):
        """Verify update_cache_variations calls repo."""
        service.cache_repo.update_cache_variations.return_value = True

        result = await service.update_cache_variations(1, ["v1", "v2"])

        assert result is True
        service.cache_repo.update_cache_variations.assert_called_once_with(1, ["v1", "v2"])

    @pytest.mark.asyncio
    async def test_search_cache_delegates(self, service):
        """Verify search_cache calls repo."""
        service.cache_repo.search_cache.return_value = [{"question": "Python"}]

        result = await service.search_cache("python", limit=10)

        assert result == [{"question": "Python"}]
        service.cache_repo.search_cache.assert_called_once_with("python", 10)
