import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityService:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000,
        )
        self._is_fitted = False

    def vectorize(self, question: str) -> str:
        if not self._is_fitted:
            vector = self.vectorizer.fit_transform([question])
            self._is_fitted = True
        else:
            vector = self.vectorizer.transform([question])

        vector_array = vector.toarray()[0].tolist()
        return json.dumps(vector_array)

    def deserialize_vector(self, vector_json: str) -> np.ndarray:
        return np.array(json.loads(vector_json))

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        max_len = max(len(vector1), len(vector2))
        v1 = np.pad(vector1, (0, max_len - len(vector1)))
        v2 = np.pad(vector2, (0, max_len - len(vector2)))

        v1 = v1.reshape(1, -1)
        v2 = v2.reshape(1, -1)

        similarity = cosine_similarity(v1, v2)[0][0]
        return float(similarity)

    def find_best_match(self, question: str, cached_questions: list[dict]) -> dict | None:
        if not cached_questions:
            return None

        new_vector = self.deserialize_vector(self.vectorize(question))

        best_match = None
        best_score = 0.0

        for cached in cached_questions:
            cached_vector = self.deserialize_vector(cached["tfidf_vector"])
            score = self.calculate_similarity(new_vector, cached_vector)

            if score >= self.threshold and score > best_score:
                best_score = score
                best_match = {**cached, "similarity_score": score}

        return best_match

    def fit_on_corpus(self, questions: list[str]) -> None:
        if questions:
            self.vectorizer.fit(questions)
            self._is_fitted = True
