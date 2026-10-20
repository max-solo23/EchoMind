from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.cache_service import (
    CACHE_DENYLIST,
    CACHE_TTL,
    MAX_VARIATIONS,
    CacheLookup,
    CacheService,
    CacheType,
)
from services.similarity_service import SimilarityIndex


def entry(entry_id=1, variations=("a", "b", "c"), variation_index=0, expires_in_days=1):
    return {
        "id": entry_id,
        "variations": list(variations),
        "variation_index": variation_index,
        "expires_at": datetime.utcnow() + timedelta(days=expires_in_days),
    }


class TestCacheType:
    def test_cache_types_exist(self):
        assert CacheType.KNOWLEDGE.value == "knowledge"
        assert CacheType.CONVERSATIONAL.value == "conversational"

    def test_ttl_configuration(self):
        assert CACHE_TTL[CacheType.KNOWLEDGE] == timedelta(days=30)
        assert CACHE_TTL[CacheType.CONVERSATIONAL] == timedelta(hours=24)


class TestShouldSkipCache:
    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    def test_questions_always_allowed(self, service):
        assert service.should_skip_cache("Why?") is False

    def test_low_signal_questions_are_skipped(self, service):
        assert service.should_skip_cache("?") is True
        assert service.should_skip_cache("ok?", is_continuation=True) is True

    def test_very_short_messages_skipped(self, service):
        assert service.should_skip_cache("hi") is True
        assert service.should_skip_cache("a") is True

    def test_denylist_only_applies_to_continuations(self, service):
        assert service.should_skip_cache("ok", is_continuation=False) is True
        assert service.should_skip_cache("thanks for that info", is_continuation=False) is False
        assert service.should_skip_cache("thanks", is_continuation=True) is True
        assert service.should_skip_cache("ok", is_continuation=True) is True

    def test_denylist_words_blocked_in_continuation(self, service):
        for word in ["ok", "thanks", "cool", "nice", "got it", "yes", "no"]:
            if word in CACHE_DENYLIST:
                assert service.should_skip_cache(word, is_continuation=True) is True

    def test_short_messages_below_threshold_skipped(self, service):
        assert service.should_skip_cache("do it now") is True
        assert service.should_skip_cache("tell me about Python programming") is False

    def test_normal_messages_allowed(self, service):
        assert service.should_skip_cache("What is your experience with Python?") is False
        assert service.should_skip_cache("Tell me about your backend skills") is False


class TestGetCacheType:
    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    def test_first_message_is_knowledge(self, service):
        assert service.get_cache_type(is_continuation=False) == CacheType.KNOWLEDGE

    def test_continuation_is_conversational(self, service):
        assert service.get_cache_type(is_continuation=True) == CacheType.CONVERSATIONAL


class TestCalculateExpiry:
    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    def test_knowledge_expires_in_30_days(self, service):
        before = datetime.utcnow()
        expiry = service.calculate_expiry(CacheType.KNOWLEDGE)
        after = datetime.utcnow()

        assert expiry >= before + timedelta(days=29, hours=23)
        assert expiry <= after + timedelta(days=30, hours=1)

    def test_conversational_expires_in_24_hours(self, service):
        before = datetime.utcnow()
        expiry = service.calculate_expiry(CacheType.CONVERSATIONAL)
        after = datetime.utcnow()

        assert expiry >= before + timedelta(hours=23)
        assert expiry <= after + timedelta(hours=25)


class TestBuildCacheKey:
    @pytest.fixture
    def service(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    def test_key_is_sha256_hex(self, service):
        key = service.build_cache_key("What is Python?")

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_message_same_key(self, service):
        key1 = service.build_cache_key("What is Python?")
        key2 = service.build_cache_key("What is Python?")

        assert key1 == key2

    def test_different_context_different_key(self, service):
        key1 = service.build_cache_key("yes", last_assistant_message="Do you know Python?")
        key2 = service.build_cache_key("yes", last_assistant_message="Are you available?")

        assert key1 != key2

    def test_no_context_uses_empty_string(self, service):
        key1 = service.build_cache_key("Hello", last_assistant_message=None)
        key2 = service.build_cache_key("Hello", last_assistant_message="")

        assert key1 == key2

    def test_different_persona_hash_different_key(self):
        mock_repo = MagicMock()
        mock_similarity = MagicMock()
        service1 = CacheService(mock_repo, mock_similarity, "hash_v1")
        service2 = CacheService(mock_repo, mock_similarity, "hash_v2")

        key1 = service1.build_cache_key("What is Python?")
        key2 = service2.build_cache_key("What is Python?")

        assert key1 != key2


class TestGetCachedAnswer:
    @pytest.fixture
    def service(self):
        repo = AsyncMock()
        repo.get_cache_by_key.return_value = None
        repo.get_knowledge_questions.return_value = []
        return CacheService(repo, SimilarityIndex(), "test_hash")

    async def test_skips_cache_for_denylist(self, service):
        result = await service.get_cached_answer("thanks", is_continuation=True)

        assert result == CacheLookup()
        service.cache_repo.get_cache_by_key.assert_not_called()

    async def test_serves_rotating_variation_once_three_are_collected(self, service):
        service.cache_repo.get_cache_by_key.return_value = entry(variation_index=4)

        result = await service.get_cached_answer("What is Python?")

        assert result == CacheLookup(entry_id=1, answer="b")  # 4 % 3

    async def test_entry_still_collecting_variations_is_a_miss_that_names_the_entry(self, service):
        service.cache_repo.get_cache_by_key.return_value = entry(variations=("a", "b"))

        result = await service.get_cached_answer("What is Python?")

        assert result == CacheLookup(entry_id=1)

    async def test_deletes_expired_exact_match(self, service):
        service.cache_repo.get_cache_by_key.return_value = entry(expires_in_days=-1)

        result = await service.get_cached_answer("What is Python?")

        service.cache_repo.delete_cache_by_id.assert_called_once_with(1)
        assert result == CacheLookup()

    async def test_returns_empty_lookup_when_nothing_cached(self, service):
        result = await service.get_cached_answer("What is Python?")

        assert result == CacheLookup()
        service.cache_repo.get_knowledge_questions.assert_awaited_once_with("test_hash")

    async def test_similar_first_turn_question_hits(self, service):
        service.cache_repo.get_knowledge_questions.return_value = [(7, "What is your tech stack?")]
        service.cache_repo.get_cache_by_id.return_value = entry(entry_id=7)

        result = await service.get_cached_answer("whats your tech-stack?")

        assert result == CacheLookup(entry_id=7, answer="a")
        service.cache_repo.get_cache_by_id.assert_awaited_once_with(7)

    async def test_different_question_with_shared_words_misses(self, service):
        service.cache_repo.get_knowledge_questions.return_value = [
            (7, "Tell me about your AI projects")
        ]

        result = await service.get_cached_answer("Tell me about your C# projects")

        assert result == CacheLookup()
        service.cache_repo.get_cache_by_id.assert_not_called()

    async def test_follow_ups_use_the_exact_key_only(self, service):
        result = await service.get_cached_answer(
            "Tell me more about that project", "Earlier answer", is_continuation=True
        )

        assert result == CacheLookup()
        service.cache_repo.get_knowledge_questions.assert_not_called()

    async def test_index_is_loaded_once_per_ttl(self, service):
        await service.get_cached_answer("What is Python?")
        await service.get_cached_answer("What is Rust?")

        service.cache_repo.get_knowledge_questions.assert_awaited_once()

    async def test_key_ignores_case_and_punctuation(self, service):
        assert service.build_cache_key("What is your stack?") == service.build_cache_key(
            "what is your stack"
        )

    async def test_record_hit_rotates_over_max_variations(self, service):
        await service.record_hit(5)

        service.cache_repo.record_hit.assert_awaited_once_with(5, MAX_VARIATIONS)


class TestShouldCache:
    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    @pytest.mark.asyncio
    async def test_returns_false_for_denylist(self, service):
        result = await service.should_cache("ok", is_continuation=True)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_new_question(self, service):
        service.cache_repo.get_cache_by_key.return_value = None

        result = await service.should_cache("What is Python?")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_exact_existing(self, service):
        service.cache_repo.get_cache_by_key.return_value = {"id": 1}

        result = await service.should_cache("What is Python?")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_different_question(self, service):
        service.cache_repo.get_cache_by_key.return_value = None

        result = await service.should_cache("How do I learn JavaScript?")

        assert result is True


class TestCacheAnswer:
    @pytest.fixture
    def service(self):
        repo = AsyncMock()
        repo.get_cache_by_key.return_value = None
        index = SimilarityIndex()
        index.rebuild([])
        return CacheService(repo, index, "test_hash")

    async def test_skips_denylist_messages(self, service):
        result = await service.cache_answer("thanks", "You're welcome!", is_continuation=True)

        assert result is None
        service.cache_repo.create_cache.assert_not_called()

    async def test_adds_variation_to_existing_exact_key(self, service):
        service.cache_repo.get_cache_by_key.return_value = {"id": 1}

        result = await service.cache_answer("What is Python?", "A programming language")

        service.cache_repo.add_variation.assert_called_once_with(1, "A programming language")
        assert result == 1

    async def test_adds_variation_to_the_matched_similar_entry(self, service):
        result = await service.cache_answer("whats your stack?", "C# and Python", entry_id=9)

        service.cache_repo.add_variation.assert_called_once_with(9, "C# and Python")
        service.cache_repo.get_cache_by_key.assert_not_called()
        assert result == 9

    async def test_creates_new_entry_with_persona_and_invalidates_index(self, service):
        service.cache_repo.create_cache.return_value = 42

        result = await service.cache_answer("What is Python?", "A programming language")

        assert result == 42
        kwargs = service.cache_repo.create_cache.call_args.kwargs
        assert kwargs["persona_hash"] == "test_hash"
        assert kwargs["cache_type"] == "knowledge"
        assert service.similarity.is_stale()

    async def test_follow_up_entry_keeps_index_fresh(self, service):
        service.cache_repo.create_cache.return_value = 3

        await service.cache_answer(
            "Tell me more about Python programming please", "Response", "Earlier", True
        )

        assert service.cache_repo.create_cache.call_args.kwargs["cache_type"] == "conversational"
        assert not service.similarity.is_stale()

    async def test_creates_cache_with_context_preview(self, service):
        service.cache_repo.create_cache.return_value = 1

        await service.cache_answer(
            "Tell me more about Python programming please",
            "Response",
            last_assistant_message="A" * 300,
        )

        assert len(service.cache_repo.create_cache.call_args.kwargs["context_preview"]) == 200


class TestCacheStats:
    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    @pytest.mark.asyncio
    async def test_returns_stats_dict(self, service):
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
        service.cache_repo.get_all_cached_questions.return_value = []

        stats = await service.get_cache_stats()

        assert stats["total_questions"] == 0
        assert stats["avg_variations_per_question"] == 0


class TestAdminMethods:
    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    @pytest.mark.asyncio
    async def test_clear_cache_delegates(self, service):
        service.cache_repo.clear_all_cache.return_value = 5

        result = await service.clear_cache()

        assert result == 5
        service.cache_repo.clear_all_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_delegates(self, service):
        service.cache_repo.delete_expired.return_value = 3

        result = await service.cleanup_expired()

        assert result == 3
        service.cache_repo.delete_expired.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_cache_entries_delegates(self, service):
        service.cache_repo.list_cache_entries.return_value = {"entries": [], "total": 0}

        result = await service.list_cache_entries(page=2, limit=15)

        assert result == {"entries": [], "total": 0}
        service.cache_repo.list_cache_entries.assert_called_once_with(2, 15, "last_used", "desc")

    @pytest.mark.asyncio
    async def test_get_cache_by_id_delegates(self, service):
        service.cache_repo.get_cache_by_id.return_value = {"id": 1, "question": "Test"}

        result = await service.get_cache_by_id(1)

        assert result == {"id": 1, "question": "Test"}

    @pytest.mark.asyncio
    async def test_delete_cache_by_id_delegates(self, service):
        service.cache_repo.delete_cache_by_id.return_value = True

        result = await service.delete_cache_by_id(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_update_cache_variations_delegates(self, service):
        service.cache_repo.update_cache_variations.return_value = True

        result = await service.update_cache_variations(1, ["v1", "v2"])

        assert result is True
        service.cache_repo.update_cache_variations.assert_called_once_with(1, ["v1", "v2"])

    @pytest.mark.asyncio
    async def test_search_cache_delegates(self, service):
        service.cache_repo.search_cache.return_value = [{"question": "Python"}]

        result = await service.search_cache("python", limit=10)

        assert result == [{"question": "Python"}]
        service.cache_repo.search_cache.assert_called_once_with("python", 10)
