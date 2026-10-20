import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from repositories.cache_repo import SQLAlchemyCacheRepository


class MockCachedAnswer:
    def __init__(
        self,
        id: int = 1,
        cache_key: str = "abc123",
        question: str = "What is Python?",
        context_preview: str | None = None,
        persona_hash: str | None = "persona",
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
        self.persona_hash = persona_hash
        self.variations = variations
        self.variation_index = variation_index
        self.cache_type = cache_type
        self.expires_at = expires_at
        self.hit_count = hit_count
        self.created_at = created_at or datetime.utcnow()
        self.last_used = last_used


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = MagicMock()  # session.add is synchronous in SQLAlchemy
    return session


@pytest.fixture
def repo(mock_session):
    return SQLAlchemyCacheRepository(mock_session)


class TestGetCacheByKey:
    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self, repo, mock_session):
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
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_key("nonexistent")

        assert result is None


class TestGetCacheByQuestion:
    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self, repo, mock_session):
        mock_cache = MockCachedAnswer(question="What is Python?")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_question("What is Python?")

        assert result is not None
        assert result["question"] == "What is Python?"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_question("Unknown question")

        assert result is None


class TestGetAllCachedQuestions:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self, repo, mock_session):
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
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        result = await repo.get_all_cached_questions()

        assert result == []


class TestCreateCache:
    @pytest.mark.asyncio
    async def test_creates_and_returns_id(self, repo, mock_session):
        async def mock_refresh(obj):
            obj.id = 42

        mock_session.refresh = mock_refresh

        result = await repo.create_cache(
            cache_key="abc123",
            question="What is Python?",
            persona_hash="persona",
            answer="A programming language",
            cache_type="knowledge",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )

        assert result == 42
        mock_session.add.assert_called_once()
        assert mock_session.add.call_args.args[0].persona_hash == "persona"
        mock_session.commit.assert_called_once()


class TestAddVariation:
    @pytest.mark.asyncio
    async def test_adds_variation_under_limit(self, repo, mock_session):
        mock_cache = MockCachedAnswer(variations='["Answer 1"]')
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        await repo.add_variation(1, "Answer 2")

        assert json.loads(mock_cache.variations) == ["Answer 1", "Answer 2"]
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_at_limit(self, repo, mock_session):
        mock_cache = MockCachedAnswer(variations='["A1", "A2", "A3"]')
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        await repo.add_variation(1, "Answer 4")

        assert len(json.loads(mock_cache.variations)) == 3

    @pytest.mark.asyncio
    async def test_handles_missing_cache(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        await repo.add_variation(999, "Answer")


class TestRecordHit:
    @pytest.mark.asyncio
    async def test_rotates_and_counts_in_one_update(self, repo, mock_session):
        await repo.record_hit(7, 3)

        (statement,), _ = mock_session.execute.call_args
        sql = str(statement.compile(dialect=postgresql.dialect()))
        assert sql.startswith("UPDATE cached_answers SET variation_index=")
        assert "(cached_answers.variation_index + %(variation_index_1)s) %% " in sql
        assert "hit_count=(cached_answers.hit_count + " in sql
        mock_session.commit.assert_awaited_once()


class TestGetKnowledgeQuestions:
    @pytest.mark.asyncio
    async def test_returns_id_question_pairs_for_live_entries_of_the_persona(
        self, repo, mock_session
    ):
        rows = [MagicMock(id=1, question="Q1"), MagicMock(id=2, question="Q2")]
        mock_session.execute.return_value = rows

        result = await repo.get_knowledge_questions("persona")

        assert result == [(1, "Q1"), (2, "Q2")]
        (statement,), _ = mock_session.execute.call_args
        sql = str(statement.compile(dialect=postgresql.dialect()))
        assert "cached_answers.cache_type = " in sql
        assert "cached_answers.persona_hash = " in sql
        assert "cached_answers.expires_at IS NULL OR cached_answers.expires_at > " in sql


class TestDeleteExpired:
    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result

        result = await repo.delete_expired()

        assert result == 5
        mock_session.commit.assert_called_once()


class TestClearAllCache:
    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 100
        mock_session.execute.return_value = mock_result

        result = await repo.clear_all_cache()

        assert result == 100
        mock_session.commit.assert_called_once()


class TestDeleteCacheById:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repo.delete_cache_by_id(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await repo.delete_cache_by_id(999)

        assert result is False


class TestUpdateCacheVariations:
    @pytest.mark.asyncio
    async def test_updates_and_resets_index(self, repo, mock_session):
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
        mock_cache = MockCachedAnswer()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache
        mock_session.execute.return_value = mock_result

        await repo.update_cache_variations(1, ["a", "b", "c", "d", "e"])

        assert len(json.loads(mock_cache.variations)) == 3

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.update_cache_variations(999, ["new"])

        assert result is False


class TestListCacheEntries:
    @pytest.mark.asyncio
    async def test_returns_paginated_results(self, repo, mock_session):
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
    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self, repo, mock_session):
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
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_cache_by_id(999)

        assert result is None


class TestSearchCache:
    @pytest.mark.asyncio
    async def test_returns_matching_entries(self, repo, mock_session):
        mock_caches = [MockCachedAnswer(id=1, question="Python question")]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_caches
        mock_session.execute.return_value = mock_result

        result = await repo.search_cache("python", limit=10)

        assert len(result) == 1
        assert result[0]["question"] == "Python question"


class TestCacheEntryShapes:
    @pytest.mark.asyncio
    async def test_preserves_all_public_mapping_shapes(self, repo, mock_session):
        cache = MockCachedAnswer()
        single_result = MagicMock()
        single_result.scalar_one_or_none.return_value = cache
        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [cache]
        count_result = MagicMock()
        count_result.scalar.return_value = 1

        mock_session.execute.side_effect = [
            single_result,
            single_result,
            list_result,
            count_result,
            list_result,
            single_result,
            list_result,
        ]

        by_key = await repo.get_cache_by_key(cache.cache_key)
        by_question = await repo.get_cache_by_question(cache.question)
        all_questions = await repo.get_all_cached_questions()
        listed = await repo.list_cache_entries()
        by_id = await repo.get_cache_by_id(cache.id)
        searched = await repo.search_cache(cache.question)

        full_detail_fields = {
            "id",
            "cache_key",
            "question",
            "context_preview",
            "variations",
            "variation_index",
            "cache_type",
            "expires_at",
            "hit_count",
            "created_at",
            "last_used",
        }
        assert by_key is not None and set(by_key) == full_detail_fields
        assert by_id is not None and set(by_id) == full_detail_fields
        assert set(by_question) == full_detail_fields - {"created_at", "last_used"}
        assert set(all_questions[0]) == {
            "id",
            "cache_key",
            "question",
            "variations",
            "variation_index",
            "cache_type",
            "expires_at",
        }
        assert set(listed["entries"][0]) == full_detail_fields
        assert set(searched[0]) == {
            "id",
            "cache_key",
            "question",
            "context_preview",
            "cache_type",
            "expires_at",
            "hit_count",
            "last_used",
        }


class TestListCacheEntriesSorting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("order", ["asc", "desc"])
    async def test_applies_sort_direction_with_nulls_last(self, repo, mock_session, order):
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        entries_result = MagicMock()
        entries_result.scalars.return_value.all.return_value = []
        mock_session.execute.side_effect = [count_result, entries_result]

        await repo.list_cache_entries(sort_by="hit_count", order=order)

        query = str(mock_session.execute.await_args_list[1].args[0])
        assert "NULLS LAST" in query
        assert ("DESC" in query) is (order == "desc")
