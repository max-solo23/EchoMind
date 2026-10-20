import json
from pathlib import Path

import pytest

from services import similarity_service as module
from services.similarity_service import (
    SimilarityIndex,
    normalize_question,
    same_content_words,
)


DATA = Path(__file__).parent / "data"


def load_pairs(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))["pairs"]


def matches(a, b, threshold=module.SIMILARITY_THRESHOLD):
    index = SimilarityIndex(threshold=threshold)
    index.rebuild([(1, a)])
    return index.best_match(b) == 1


def test_normalize_keeps_language_symbols_and_drops_punctuation():
    assert normalize_question("  What's your C# / C++  stack?? ") == "what s your c# c++ stack"
    assert normalize_question("Max è disponibile?") == "max è disponibile"


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("What is Max's strongest stack?", "what is maxs stronegst stack", True),  # transposition
        ("Tell me about your C# project", "Tell me about your C# projects", True),  # inflection
        ("Quali progetto hai fatto?", "Quali progetti hai fatto?", True),  # Italian ending
        ("Does Max know Docker?", "Does Max know Dockerxyz?", False),  # bare prefix
        ("Does Max know Java?", "Does Max know lava?", False),  # one-letter change
        ("Tell me about your C# projects", "Tell me about your AI projects", False),
        ("What is your tech stack?", "What is your tech stack for mobile apps?", False),
        ("Who are you?", "What is it?", False),  # no content words: exact key only
    ],
)
def test_same_content_words(a, b, same):
    assert same_content_words(a, b) is same


@pytest.mark.parametrize(
    ("name", "min_recall"), [("similarity_pairs", 0.65), ("similarity_pairs_holdout", 0.70)]
)
def test_threshold_has_full_precision_on_labeled_pairs(name, min_recall):
    """A false match serves another question's answer, so precision must stay at 1.0."""
    pairs = load_pairs(name)
    results = [(matches(p["a"], p["b"]), p["same"]) for p in pairs]

    false_matches = [p for p, (hit, same) in zip(pairs, results, strict=True) if hit and not same]
    true_matches = sum(1 for hit, same in results if hit and same)
    recall = true_matches / sum(1 for _, same in results if same)

    assert false_matches == []
    assert recall >= min_recall


def test_extra_words_lower_the_score_instead_of_vanishing():
    index = SimilarityIndex()
    index.rebuild([(1, "What is your tech stack?")])

    (same,) = index.score("what is your tech stack")
    (longer,) = index.score("what is your tech stack and what food do you like to cook")

    assert same > 0.99
    assert longer < 0.6


def test_best_match_skips_high_score_with_different_content_words():
    index = SimilarityIndex()
    index.rebuild([(1, "Tell me about your AI projects"), (2, "Tell me about your C# projects")])

    assert index.best_match("tell me about your c# project") == 2
    assert index.best_match("Tell me about your Rust projects") is None


def test_empty_index_matches_nothing():
    index = SimilarityIndex()
    index.rebuild([])

    assert index.score("anything") == []
    assert index.best_match("anything") is None


def test_staleness_follows_ttl_and_invalidate(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    index = SimilarityIndex()
    assert index.is_stale()

    index.rebuild([(1, "What is your stack?")])
    assert not index.is_stale()
    clock["now"] += module.INDEX_TTL_SECONDS
    assert index.is_stale()

    index.rebuild([(1, "What is your stack?")])
    index.invalidate()
    assert index.is_stale()
