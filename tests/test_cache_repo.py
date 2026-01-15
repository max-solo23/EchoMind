"""Tests for SQLAlchemyCacheRepository - database cache operations."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from repositories.cache_repo import SQLAlchemyCacheRepository


class MockCachedAnswer:
    """Mock SQLAlchemy model for testing."""

    def __init__(
        self,
        id: int = 1,
        cache_key: str = "abc123",
        question: str = "What is Python?",
        context_preview: str | None = None,
        tfidf_vector: str = "[0.5, 0.3]",
        variations: str = '["Answer 1"]',
        variation_index: int = 0,
        cache_type: str = "knowledge",
        expires_at: datetime | None = None,
        hit_count: int = 0,
        created_at: datetime | None = None,
        last_used: datetime | None = None,
    ):
        self.id = id
        self.cache_key = cache_key
        self.question = question
        self.context_preview = context_preview
        self.tfidf_vector = tfidf_vector
        self.variations = variations
        self.variation_index = variation_index
        self.cache_type = cache_type
        self.expires_at = expires_at
        self.hit_count = hit_count
        self.created_at = created_at or datetime.utcnow()
        self.last_used = last_used


@pytest.fixture
def mock_session():
    """Create mocked AsyncSession."""
    session = AsyncMock()
    return session


@pytest.fixture
def repo(mock_session):
    """Create repository with mocked session."""
    return SQLAlchemyCacheRepository(mock_session)


class TestGetCacheByKey:
    """Test cache lookup by SHA256 key."""

    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self, repo, mock_session):
        """
        Verify cache entry is returned as dict when found.

        Why: Service layer expects dict, not SQLAlchemy model.
        Decouples service from ORM implementation.
        """
        mock_cache = MockCachedAnswer(id=1, cache_key="abc123")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_key("abc123")

        assert result is not None
        assert result["id"] == 1
        assert result["cache_key"] == "abc123"
        assert "variations" in result
        assert isinstance(result["variations"], list)

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_session):
        """
        Verify None returned when key doesn't exist.

        Why: Caller needs to distinguish "not found" from "found empty".
        """
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_key("nonexistent")

        assert result is None


class TestGetCacheByQuestion:
    """Test cache lookup by question text."""

    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self, repo, mock_session):
        """
        Verify question lookup returns dict.

        Why: Used for backwards compatibility and similarity matching.
        """
        mock_cache = MockCachedAnswer(question="What is Python?")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_question("What is Python?")

        assert result is not None
        assert result["question"] == "What is Python?"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_session):
        """Verify None when question not cached."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_question("Unknown question")

        assert result is None


class TestGetAllCachedQuestions:
    """Test bulk retrieval for similarity matching."""

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self, repo, mock_session):
        """
        Verify all entries returned as list of dicts.

        Why: Similarity service needs all questions to compare against.
        """
        mock_caches = [
            MockCachedAnswer(id=1, question="Q1"),
            MockCachedAnswer(id=2, question="Q2"),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_caches
        mock_session.execute.return_value = mock_result

        result = await repo.get_all_cached_questions()

        assert len(result) == 2
        assert result[0]["question"] == "Q1"
        assert result[1]["question"] == "Q2"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_cache(self, repo, mock_session):
        """Verify empty list when cache is empty."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        result = await repo.get_all_cached_questions()

        assert result == []


class TestCreateCache:
    """Test cache entry creation."""

    @pytest.mark.asyncio
    async def test_creates_and_returns_id(self, repo, mock_session):
        """
        Verify new entry is created and ID returned.

        Why: Caller needs ID to add variations later.
        """

        # Mock the refresh to set ID
        async def mock_refresh(obj):
            obj.id = 42

        mock_session.refresh = mock_refresh

        result = await repo.create_cache(
            cache_key="abc123",
            question="What is Python?",
            tfidf_vector="[0.5]",
            answer="A programming language",
            cache_type="knowledge",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )

        assert result == 42
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


class TestAddVariation:
    """Test adding answer variations."""

    @pytest.mark.asyncio
    async def test_adds_variation_under_limit(self, repo, mock_session):
        """
        Verify variation added when under 3.

        Why: Multiple variations make bot feel less robotic.
        """
        mock_cache = MockCachedAnswer(variations='["Answer 1"]')
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        await repo.add_variation(1, "Answer 2")

        # Check variations was updated
        assert json.loads(mock_cache.variations) == ["Answer 1", "Answer 2"]
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_at_limit(self, repo, mock_session):
        """
        Verify no addition when already at 3 variations.

        Why: Limit prevents unbounded growth of cache entries.
        """
        mock_cache = MockCachedAnswer(variations='["A1", "A2", "A3"]')
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        await repo.add_variation(1, "Answer 4")

        # Variations should still be 3
        assert len(json.loads(mock_cache.variations)) == 3

    @pytest.mark.asyncio
    async def test_handles_missing_cache(self, repo, mock_session):
        """Verify no error when cache entry doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Should not raise
        await repo.add_variation(999, "Answer")


class TestGetNextVariation:
    """Test answer rotation."""

    @pytest.mark.asyncio
    async def test_returns_current_and_rotates(self, repo, mock_session):
        """
        Verify returns answer at current index and rotates.

        Why: Rotating answers prevents repetitive responses.
        """
        mock_cache = MockCachedAnswer(variations='["A", "B", "C"]', variation_index=0, hit_count=5)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.get_next_variation(1)

        assert result == "A"
        assert mock_cache.variation_index == 1
        assert mock_cache.hit_count == 6
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_wraps_around_at_end(self, repo, mock_session):
        """
        Verify index wraps from last to first.

        Why: After showing all variations, start over.
        """
        mock_cache = MockCachedAnswer(
            variations='["A", "B", "C"]',
            variation_index=2,  # Last index
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.get_next_variation(1)

        assert result == "C"
        assert mock_cache.variation_index == 0  # Wrapped to start

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_found(self, repo, mock_session):
        """Verify empty string when cache doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_next_variation(999)

        assert result == ""


class TestDeleteExpired:
    """Test expired entry cleanup."""

    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, repo, mock_session):
        """
        Verify returns number of deleted entries.

        Why: Admin dashboard shows cleanup results.
        """
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result

        result = await repo.delete_expired()

        assert result == 5
        mock_session.commit.assert_called_once()


class TestClearAllCache:
    """Test full cache clear."""

    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, repo, mock_session):
        """Verify returns total deleted count."""
        mock_result = MagicMock()
        mock_result.rowcount = 100
        mock_session.execute.return_value = mock_result

        result = await repo.clear_all_cache()

        assert result == 100
        mock_session.commit.assert_called_once()


class TestDeleteCacheById:
    """Test single entry deletion."""

    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self, repo, mock_session):
        """Verify True when entry was deleted."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repo.delete_cache_by_id(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, repo, mock_session):
        """Verify False when entry didn't exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await repo.delete_cache_by_id(999)

        assert result is False


class TestUpdateCacheVariations:
    """Test variation updates."""

    @pytest.mark.asyncio
    async def test_updates_and_resets_index(self, repo, mock_session):
        """
        Verify variations updated and index reset.

        Why: New variations mean old rotation position is invalid.
        """
        mock_cache = MockCachedAnswer(variations='["old"]', variation_index=2)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.update_cache_variations(1, ["new1", "new2"])

        assert result is True
        assert json.loads(mock_cache.variations) == ["new1", "new2"]
        assert mock_cache.variation_index == 0

    @pytest.mark.asyncio
    async def test_enforces_max_three(self, repo, mock_session):
        """
        Verify max 3 variations enforced.

        Why: Prevent unbounded growth of cache entries.
        """
        mock_cache = MockCachedAnswer()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        await repo.update_cache_variations(1, ["a", "b", "c", "d", "e"])

        # Should be truncated to 3
        assert len(json.loads(mock_cache.variations)) == 3

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, repo, mock_session):
        """Verify False when cache doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.update_cache_variations(999, ["new"])

        assert result is False


class TestListCacheEntries:
    """Test paginated cache entry listing."""

    @pytest.mark.asyncio
    async def test_returns_paginated_results(self, repo, mock_session):
        """Verify pagination metadata included."""
        mock_caches = [
            MockCachedAnswer(id=1, question="Q1"),
            MockCachedAnswer(id=2, question="Q2"),
        ]

        count_result = MagicMock()
        count_result.scalar.return_value = 50

        entries_result = MagicMock()
        entries_result.scalars.return_value.all.return_value = mock_caches

        mock_session.execute.side_effect = [count_result, entries_result]

        result = await repo.list_cache_entries(page=1, limit=20)

        assert result["total"] == 50
        assert result["page"] == 1
        assert len(result["entries"]) == 2


class TestGetCacheById:
    """Test single cache entry retrieval."""

    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self, repo, mock_session):
        """Verify cache entry returned as dict."""
        mock_cache = MockCachedAnswer(id=1, question="Test?")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_id(1)

        assert result is not None
        assert result["id"] == 1
        assert result["question"] == "Test?"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_session):
        """Verify None when ID doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_id(999)

        assert result is None


class TestSearchCache:
    """Test cache search."""

    @pytest.mark.asyncio
    async def test_returns_matching_entries(self, repo, mock_session):
        """Verify search returns matching entries."""
        mock_caches = [MockCachedAnswer(id=1, question="Python question")]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_caches
        mock_session.execute.return_value = mock_result

        result = await repo.search_cache("python", limit=10)

        assert len(result) == 1
        assert result[0]["question"] == "Python question"
