import json

import numpy as np
import pytest

from services.similarity_service import SimilarityService


class TestSimilarityServiceInit:

    def test_init_default_threshold(self):
        service = SimilarityService()
        assert service.threshold == 0.80

    def test_init_custom_threshold(self):
        service = SimilarityService(threshold=0.75)
        assert service.threshold == 0.75

    def test_init_vectorizer_not_fitted(self):
        service = SimilarityService()
        assert service._is_fitted is False


class TestVectorize:

    def test_vectorize_returns_json_string(self):
        service = SimilarityService()
        result = service.vectorize("What is Python?")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_vectorize_fits_on_first_call(self):
        service = SimilarityService()
        assert service._is_fitted is False

        service.vectorize("What is Python?")

        assert service._is_fitted is True

    def test_vectorize_subsequent_calls_use_transform(self):
        service = SimilarityService()
        service.vectorize("What is Python?")

        result = service.vectorize("How do I learn Python?")
        assert isinstance(result, str)
        assert service._is_fitted is True


class TestDeserializeVector:

    def test_deserialize_returns_numpy_array(self):
        service = SimilarityService()
        json_vector = json.dumps([0.5, 0.3, 0.2])

        result = service.deserialize_vector(json_vector)

        assert isinstance(result, np.ndarray)
        assert list(result) == [0.5, 0.3, 0.2]


class TestCalculateSimilarity:

    def test_identical_vectors_return_one(self):
        service = SimilarityService()
        vector = np.array([0.5, 0.3, 0.2])

        result = service.calculate_similarity(vector, vector)

        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        service = SimilarityService()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        result = service.calculate_similarity(v1, v2)

        assert result == pytest.approx(0.0)

    def test_different_length_vectors_are_padded(self):
        service = SimilarityService()
        v1 = np.array([0.5, 0.5])
        v2 = np.array([0.5, 0.5, 0.0, 0.0])

        result = service.calculate_similarity(v1, v2)

        assert result == pytest.approx(1.0)


class TestFindBestMatch:

    def test_empty_cache_returns_none(self):
        service = SimilarityService()

        result = service.find_best_match("What is Python?", [])

        assert result is None

    def test_match_above_threshold_returned(self):
        service = SimilarityService(threshold=0.5)

        question = "What is Python programming?"
        vector_json = service.vectorize(question)

        cached_questions = [{"id": 1, "question": question, "tfidf_vector": vector_json}]

        result = service.find_best_match(question, cached_questions)

        assert result is not None
        assert result["id"] == 1
        assert "similarity_score" in result
        assert result["similarity_score"] >= 0.5

    def test_match_below_threshold_returns_none(self):
        service = SimilarityService(threshold=0.99)

        q1 = "What is Python?"
        q2 = "How do I cook pasta?"

        v1 = service.vectorize(q1)

        cached_questions = [{"id": 1, "question": q1, "tfidf_vector": v1}]

        result = service.find_best_match(q2, cached_questions)

        assert result is None

    def test_returns_best_match_when_multiple_above_threshold(self):
        service = SimilarityService(threshold=0.3)

        base_q = "What programming language should I learn?"
        similar_q = "What programming language is best to learn?"
        different_q = "How do I make coffee?"

        service.fit_on_corpus([base_q, similar_q, different_q])

        cached_questions = [
            {"id": 1, "question": similar_q, "tfidf_vector": service.vectorize(similar_q)},
            {"id": 2, "question": different_q, "tfidf_vector": service.vectorize(different_q)},
        ]

        result = service.find_best_match(base_q, cached_questions)

        assert result is not None
        assert result["id"] == 1


class TestFitOnCorpus:

    def test_fit_on_corpus_sets_fitted_flag(self):
        service = SimilarityService()
        assert service._is_fitted is False

        service.fit_on_corpus(["Question one", "Question two"])

        assert service._is_fitted is True

    def test_fit_on_empty_corpus_does_nothing(self):
        service = SimilarityService()

        service.fit_on_corpus([])

        assert service._is_fitted is False
