"""
TF-IDF Similarity Service for question matching.

This service handles:
- TF-IDF vectorization of questions
- Cosine similarity comparison
- Threshold-based matching (90%)

Design notes:
- Uses scikit-learn's TfidfVectorizer
- Vectors are serialized as JSON for database storage
- Similarity threshold is configurable
"""

import json
import numpy as np
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityService:
    """
    Handles TF-IDF vectorization and similarity matching.

    Why TF-IDF?
    - Term Frequency-Inverse Document Frequency
    - Weighs words by importance (common words get lower weight)
    - Good for short text like questions
    """

    def __init__(self, threshold: float = 0.90):
        """
        Initialize with similarity threshold.

        Args:
            threshold: Minimum cosine similarity to consider a match (0.0-1.0)
        """
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000
        )
        self._is_fitted = False

    def vectorize(self, question: str) -> str:
        """
        Convert question to TF-IDF vector and serialize to JSON.

        Args:
            question: The question text to vectorize

        Returns:
            JSON string of the TF-IDF vector
        """
        if not self._is_fitted:
            # Fit on single question if not fitted
            vector = self.vectorizer.fit_transform([question])
            self._is_fitted = True
        else:
            vector = self.vectorizer.transform([question])

        # Convert sparse matrix to dense array, then to list for JSON
        vector_array = vector.toarray()[0].tolist()
        return json.dumps(vector_array)

    def deserialize_vector(self, vector_json: str) -> np.ndarray:
        """
        Deserialize JSON vector back to numpy array.

        Args:
            vector_json: JSON string of the vector

        Returns:
            Numpy array of the vector
        """
        return np.array(json.loads(vector_json))

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First TF-IDF vector
            vector2: Second TF-IDF vector

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        # Handle different length vectors by padding
        max_len = max(len(vector1), len(vector2))
        v1 = np.pad(vector1, (0, max_len - len(vector1)))
        v2 = np.pad(vector2, (0, max_len - len(vector2)))

        # Reshape for cosine_similarity function
        v1 = v1.reshape(1, -1)
        v2 = v2.reshape(1, -1)

        similarity = cosine_similarity(v1, v2)[0][0]
        return float(similarity)

    def find_best_match(
        self,
        question: str,
        cached_questions: list[dict]
    ) -> Optional[dict]:
        """
        Find the best matching cached question above threshold.

        Args:
            question: The new question to match
            cached_questions: List of cached questions with their vectors
                Each dict should have: id, question, tfidf_vector

        Returns:
            Best matching cache entry or None if no match above threshold
        """
        if not cached_questions:
            return None

        # Vectorize the new question
        new_vector = self.deserialize_vector(self.vectorize(question))

        best_match = None
        best_score = 0.0

        for cached in cached_questions:
            cached_vector = self.deserialize_vector(cached["tfidf_vector"])
            score = self.calculate_similarity(new_vector, cached_vector)

            if score >= self.threshold and score > best_score:
                best_score = score
                best_match = {
                    **cached,
                    "similarity_score": score
                }

        return best_match

    def fit_on_corpus(self, questions: list[str]) -> None:
        """
        Fit the vectorizer on a corpus of questions.

        Call this with all existing cached questions to ensure
        consistent vectorization.

        Args:
            questions: List of question strings to fit on
        """
        if questions:
            self.vectorizer.fit(questions)
            self._is_fitted = True
