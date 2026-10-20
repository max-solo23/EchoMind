from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


# Chosen on tests/data/similarity_pairs.json (precision 1.00, recall 0.81) and confirmed on the
# untouched similarity_pairs_holdout.json (precision 1.00, recall 0.80); see test_similarity_service.
SIMILARITY_THRESHOLD = 0.60
# ponytail: a machine sees questions cached by the other machine after this delay; fine for
# a cache, push invalidation if it ever matters.
INDEX_TTL_SECONDS = 60.0


def normalize_question(text: str) -> str:
    """Lowercase, drop punctuation (keeping # and + for C#, C++), collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s#+]", " ", text.lower()).split())


# Function words that may differ between two phrasings of the same question (EN + IT).
# Everything else is a content word and has to appear on both sides.
STOP_WORDS = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "does", "did", "s", "what", "whats", "which", "who", "how", "where", "when", "why", "can", "could", "would", "should", "will", "i", "you", "your", "yours", "me", "my", "max", "maxs", "he", "his", "him", "it", "its", "this", "that", "of", "to", "in", "into", "on", "at", "for", "with", "and", "or", "about", "tell", "please", "has", "have", "had", "any", "now", "right", "currently", "get", "some", "there", "il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "da", "del", "della", "dei", "con", "su", "per", "tra", "fra", "e", "è", "che", "cosa", "come", "dove", "quale", "quali", "chi", "hai", "ha", "ho", "sono", "sei", "ti", "mi", "tu", "tuo", "tua", "suo", "sua", "adesso", "ora", "attualmente", "faccio", "posso", "puoi", "può", "mi", "ci", "si"]
)


def _same_word(a: str, b: str) -> bool:
    """Equal, an inflection (one is a prefix of the other), or a one-letter typo/transposition."""
    if a == b:
        return True
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    if len(a) == len(b) >= 4:
        return sorted(a) == sorted(b) or sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    return False


def same_content_words(a: str, b: str) -> bool:
    """True when every content word of each question has a counterpart in the other.

    Questions made only of function words ("who are you?") never match fuzzily: there is
    nothing to compare, so they rely on the exact key.
    """
    words_a = [w for w in normalize_question(a).split() if w not in STOP_WORDS]
    words_b = [w for w in normalize_question(b).split() if w not in STOP_WORDS]
    if not words_a or not words_b:
        return False
    return all(any(_same_word(x, y) for y in words_b) for x in words_a) and all(
        any(_same_word(y, x) for x in words_a) for y in words_b
    )


class SimilarityIndex:
    """In-memory TF-IDF over character n-grams of cached first-turn questions.

    Character n-grams work for English and Italian alike and tolerate typos and inflection.
    Hashing (instead of a fitted vocabulary) keeps every n-gram of the incoming question, so
    extra words lower the score instead of silently disappearing from the comparison.
    Methods are CPU-bound and synchronous: call them through asyncio.to_thread.
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._ids: list[int] = []
        self._questions: list[str] = []
        self._hasher: HashingVectorizer | None = None
        self._tfidf: TfidfTransformer | None = None
        self._matrix: Any = None  # scipy sparse matrix; scipy ships no type stubs
        self._loaded_at: float | None = None

    def is_stale(self) -> bool:
        return self._loaded_at is None or time.monotonic() - self._loaded_at >= INDEX_TTL_SECONDS

    def invalidate(self) -> None:
        self._loaded_at = None

    def rebuild(self, entries: list[tuple[int, str]]) -> None:
        # sklearn/scipy cost ~1 s to import; they load here, in a worker thread, only once the
        # answer cache is switched on.
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

        self._loaded_at = time.monotonic()
        self._ids = [entry_id for entry_id, _ in entries]
        self._questions = [question for _, question in entries]
        if not entries:
            self._matrix = None
            return
        self._hasher = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2**18,
            alternate_sign=False,
            norm=None,
        )
        counts = self._hasher.transform([normalize_question(q) for _, q in entries])
        self._tfidf = TfidfTransformer(sublinear_tf=True).fit(counts)
        self._matrix = self._tfidf.transform(counts)

    def score(self, question: str) -> list[float]:
        """Cosine similarity of the question to every indexed entry, in index order."""
        if self._matrix is None or self._hasher is None or self._tfidf is None:
            return []
        query = self._tfidf.transform(self._hasher.transform([normalize_question(question)]))
        return [float(x) for x in (self._matrix @ query.T).toarray().ravel()]

    def best_match(self, question: str) -> int | None:
        """Most similar entry above the threshold whose content words also match."""
        scores = self.score(question)
        for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=True):
            if scores[i] < self.threshold:
                break
            if same_content_words(question, self._questions[i]):
                return self._ids[i]
        return None
