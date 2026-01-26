from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.cache_service import (
    CACHE_DENYLIST,
    CACHE_TTL,
    CacheService,
    CacheType,
)


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
        assert service.should_skip_cache("?") is False
        assert service.should_skip_cache("ok?", is_continuation=True) is False

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
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        return CacheService(mock_repo, mock_similarity, "test_hash")

    @pytest.mark.asyncio
    async def test_skips_cache_for_denylist(self, service):
        result = await service.get_cached_answer("thanks", is_continuation=True)

        assert result is None
        service.cache_repo.get_cache_by_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_cached_answer_on_exact_match(self, service):
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
        service.cache_repo.get_cache_by_key.return_value = {
            "id": 1,
            "expires_at": datetime.utcnow() - timedelta(days=1),
        }
        service.cache_repo.get_all_cached_questions.return_value = []

        result = await service.get_cached_answer("What is Python?")

        service.cache_repo.delete_cache_by_id.assert_called_once_with(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cache(self, service):
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = []

        result = await service.get_cached_answer("What is Python?")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_similar_match_when_no_exact(self, service):
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
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = [
            {
                "id": 1,
                "question": "Expired question",
                "tfidf_vector": "[0.5]",
                "expires_at": datetime.utcnow() - timedelta(days=1),
            }
        ]

        result = await service.get_cached_answer("What is Python?")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_similar_match(self, service):
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.get_all_cached_questions.return_value = [
            {
                "id": 1,
                "question": "What is Python?",
                "tfidf_vector": "[0.5]",
                "expires_at": datetime.utcnow() + timedelta(days=1),
            }
        ]
        service.similarity.find_best_match.return_value = None

        result = await service.get_cached_answer("How do I cook pasta?")

        assert result is None


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
        service.cache_repo.get_all_cached_questions.return_value = []

        result = await service.should_cache("What is Python?")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_similar_existing(self, service):
        service.cache_repo.get_all_cached_questions.return_value = [
            {"id": 1, "question": "What is Python?", "tfidf_vector": "[0.5]"}
        ]
        service.similarity.find_best_match.return_value = {"id": 1}

        result = await service.should_cache("What is Python programming?")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_different_question(self, service):
        service.cache_repo.get_all_cached_questions.return_value = [
            {"id": 1, "question": "What is Python?", "tfidf_vector": "[0.5]"}
        ]
        service.similarity.find_best_match.return_value = None

        result = await service.should_cache("How do I learn JavaScript?")

        assert result is True


class TestCacheAnswer:
    @pytest.fixture
    def service(self):
        mock_repo = AsyncMock()
        mock_similarity = MagicMock()
        mock_similarity.vectorize.return_value = "[0.5, 0.3, 0.2]"
        return CacheService(mock_repo, mock_similarity, "test_hash")

    @pytest.mark.asyncio
    async def test_skips_denylist_messages(self, service):
        result = await service.cache_answer("thanks", "You're welcome!", is_continuation=True)

        assert result is None
        service.cache_repo.create_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_adds_variation_to_existing(self, service):
        service.cache_repo.get_cache_by_key.return_value = {"id": 1}

        result = await service.cache_answer("What is Python?", "A programming language")

        service.cache_repo.add_variation.assert_called_once_with(1, "A programming language")
        assert result == 1

    @pytest.mark.asyncio
    async def test_creates_new_cache_entry(self, service):
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.create_cache.return_value = 42

        result = await service.cache_answer("What is Python?", "A programming language")

        assert result == 42
        service.cache_repo.create_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_cache_with_context_preview(self, service):
        service.cache_repo.get_cache_by_key.return_value = None
        service.cache_repo.create_cache.return_value = 1

        long_context = "A" * 300

        await service.cache_answer(
            "Tell me more about Python programming please",
            "Response",
            last_assistant_message=long_context,
        )

        call_kwargs = service.cache_repo.create_cache.call_args[1]
        assert len(call_kwargs["context_preview"]) == 200


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
