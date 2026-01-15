"""Tests for SimilarityService - TF-IDF question matching."""

import json

import numpy as np
import pytest

from services.similarity_service import SimilarityService


class TestSimilarityServiceInit:
    """Test SimilarityService initialization."""

    def test_init_default_threshold(self):
        """
        Verify default threshold is 0.90 (90% similarity).

        Why: 90% is strict enough to avoid false matches but lenient
        enough to catch rephrased questions like "What do you do?"
        vs "What's your job?". Default should be sensible out of box.
        """
        service = SimilarityService()
        assert service.threshold == 0.90

    def test_init_custom_threshold(self):
        """
        Verify custom threshold is stored.

        Why: Different use cases need different thresholds. FAQ matching
        might use 0.80, while deduplication might need 0.95.
        """
        service = SimilarityService(threshold=0.75)
        assert service.threshold == 0.75

    def test_init_vectorizer_not_fitted(self):
        """
        Verify vectorizer starts unfitted.

        Why: The vectorizer needs training data before it can transform.
        _is_fitted flag tracks this state to handle first-call differently.
        """
        service = SimilarityService()
        assert service._is_fitted is False


class TestVectorize:
    """Test vectorization of questions."""

    def test_vectorize_returns_json_string(self):
        """
        Verify vectorize returns JSON string (not numpy array).

        Why: Vectors are stored in database as JSON strings. Numpy arrays
        can't be stored directly in PostgreSQL text columns.
        """
        service = SimilarityService()
        result = service.vectorize("What is Python?")

        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_vectorize_fits_on_first_call(self):
        """
        Verify first vectorize call fits the vectorizer.

        Why: TfidfVectorizer needs to learn vocabulary before transforming.
        First call does fit_transform, subsequent calls do transform only.
        """
        service = SimilarityService()
        assert service._is_fitted is False

        service.vectorize("What is Python?")

        assert service._is_fitted is True

    def test_vectorize_subsequent_calls_use_transform(self):
        """
        Verify subsequent calls don't re-fit (use existing vocabulary).

        Why: Re-fitting would change vector dimensions, making cached
        vectors incompatible. Once fitted, vocabulary stays fixed.
        """
        service = SimilarityService()
        service.vectorize("What is Python?")

        # Second call should still work (transform, not fit)
        result = service.vectorize("How do I learn Python?")
        assert isinstance(result, str)
        assert service._is_fitted is True


class TestDeserializeVector:
    """Test JSON to numpy conversion."""

    def test_deserialize_returns_numpy_array(self):
        """
        Verify deserialize converts JSON back to numpy array.

        Why: Cosine similarity calculation needs numpy arrays, not lists.
        This reverses what vectorize() does for storage.
        """
        service = SimilarityService()
        json_vector = json.dumps([0.5, 0.3, 0.2])

        result = service.deserialize_vector(json_vector)

        assert isinstance(result, np.ndarray)
        assert list(result) == [0.5, 0.3, 0.2]


class TestCalculateSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors_return_one(self):
        """
        Verify identical vectors have similarity 1.0.

        Why: Cosine similarity of a vector with itself is always 1.0.
        This is the maximum possible similarity.
        """
        service = SimilarityService()
        vector = np.array([0.5, 0.3, 0.2])

        result = service.calculate_similarity(vector, vector)

        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        """
        Verify orthogonal vectors have similarity 0.0.

        Why: Vectors at 90 degrees have zero cosine similarity.
        They share no common direction.
        """
        service = SimilarityService()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        result = service.calculate_similarity(v1, v2)

        assert result == pytest.approx(0.0)

    def test_different_length_vectors_are_padded(self):
        """
        Verify vectors of different lengths are padded and compared.

        Why: Cached vectors from different vocabulary sizes need comparison.
        Shorter vector gets zero-padded to match longer one.
        """
        service = SimilarityService()
        v1 = np.array([0.5, 0.5])  # Length 2
        v2 = np.array([0.5, 0.5, 0.0, 0.0])  # Length 4

        result = service.calculate_similarity(v1, v2)

        # After padding v1 to [0.5, 0.5, 0.0, 0.0], should be identical
        assert result == pytest.approx(1.0)


class TestFindBestMatch:
    """Test finding best matching cached question."""

    def test_empty_cache_returns_none(self):
        """
        Verify empty cache returns None.

        Why: No cached questions means no possible match.
        Should handle gracefully without error.
        """
        service = SimilarityService()

        result = service.find_best_match("What is Python?", [])

        assert result is None

    def test_match_above_threshold_returned(self):
        """
        Verify match above threshold is returned with score.

        Why: Core functionality - find similar cached questions.
        Result should include original data plus similarity_score.
        """
        service = SimilarityService(threshold=0.5)  # Low threshold for test

        # Vectorize a question
        question = "What is Python programming?"
        vector_json = service.vectorize(question)

        cached_questions = [{"id": 1, "question": question, "tfidf_vector": vector_json}]

        # Same question should match perfectly
        result = service.find_best_match(question, cached_questions)

        assert result is not None
        assert result["id"] == 1
        assert "similarity_score" in result
        assert result["similarity_score"] >= 0.5

    def test_match_below_threshold_returns_none(self):
        """
        Verify match below threshold returns None.

        Why: Low similarity means questions are too different.
        Returning a bad match would give wrong cached answer.
        """
        service = SimilarityService(threshold=0.99)  # Very high threshold

        q1 = "What is Python?"
        q2 = "How do I cook pasta?"

        v1 = service.vectorize(q1)

        cached_questions = [{"id": 1, "question": q1, "tfidf_vector": v1}]

        # Different question should not match at 99% threshold
        result = service.find_best_match(q2, cached_questions)

        assert result is None

    def test_returns_best_match_when_multiple_above_threshold(self):
        """
        Verify best match is returned when multiple qualify.

        Why: Multiple cached questions might be similar. We want
        the MOST similar one, not just any above threshold.
        """
        service = SimilarityService(threshold=0.3)  # Low threshold

        base_q = "What programming language should I learn?"
        similar_q = "What programming language is best to learn?"
        different_q = "How do I make coffee?"

        # Fit on all questions for consistent vocabulary
        service.fit_on_corpus([base_q, similar_q, different_q])

        cached_questions = [
            {"id": 1, "question": similar_q, "tfidf_vector": service.vectorize(similar_q)},
            {"id": 2, "question": different_q, "tfidf_vector": service.vectorize(different_q)},
        ]

        result = service.find_best_match(base_q, cached_questions)

        # Should match similar_q, not different_q
        assert result is not None
        assert result["id"] == 1


class TestFitOnCorpus:
    """Test fitting vectorizer on corpus."""

    def test_fit_on_corpus_sets_fitted_flag(self):
        """
        Verify fit_on_corpus marks vectorizer as fitted.

        Why: After fitting on corpus, subsequent vectorize calls
        should use transform (not fit_transform) for consistency.
        """
        service = SimilarityService()
        assert service._is_fitted is False

        service.fit_on_corpus(["Question one", "Question two"])

        assert service._is_fitted is True

    def test_fit_on_empty_corpus_does_nothing(self):
        """
        Verify empty corpus doesn't fit (avoids sklearn error).

        Why: sklearn raises error on empty fit. Code checks for
        empty list and skips fitting gracefully.
        """
        service = SimilarityService()

        service.fit_on_corpus([])

        assert service._is_fitted is False
