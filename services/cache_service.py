import hashlib
from datetime import datetime, timedelta
from enum import Enum

from repositories.cache_repo import SQLAlchemyCacheRepository

from .similarity_service import SimilarityService


class CacheType(str, Enum):
    KNOWLEDGE = "knowledge"
    CONVERSATIONAL = "conversational"

CACHE_TTL = {
    CacheType.KNOWLEDGE: timedelta(days=30),
    CacheType.CONVERSATIONAL: timedelta(hours=24),
}

CACHE_DENYLIST = {
    "ok",
    "okay",
    "yes",
    "no",
    "yeah",
    "yep",
    "nope",
    "yea",
    "nah",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "thank",
    "thankyou",
    "continue",
    "go on",
    "go ahead",
    "more",
    "next",
    "sure",
    "alright",
    "right",
    "got it",
    "understood",
    "i understand",
    "cool",
    "nice",
    "great",
    "awesome",
    "perfect",
    "good",
    "fine",
    "hmm",
    "hm",
    "ah",
    "oh",
    "i see",
    "uh",
    "um",
    "wow",
    "k",
    "kk",
    "ya",
    "ye",
    "na",
    "lol",
    "haha",
}

MIN_TOKENS_FOR_CACHE = 4


class CacheService:
    def __init__(
        self,
        cache_repo: SQLAlchemyCacheRepository,
        similarity_service: SimilarityService,
        persona_hash: str
    ):
        self.cache_repo = cache_repo
        self.similarity = similarity_service
        self.persona_hash = persona_hash

    def should_skip_cache(self, message: str, is_continuation: bool = False) -> bool:
        normalized = message.lower().strip()
        tokens = normalized.split()
        token_count = len(tokens)

        if "?" in message:
            return False

        if token_count < 2:
            return True

        if is_continuation and normalized in CACHE_DENYLIST:
            return True

        return token_count < MIN_TOKENS_FOR_CACHE

    def get_cache_type(self, is_continuation: bool) -> CacheType:
        if is_continuation:
            return CacheType.CONVERSATIONAL
        return CacheType.KNOWLEDGE

    def calculate_expiry(self, cache_type: CacheType) -> datetime:
        return datetime.utcnow() + CACHE_TTL[cache_type]

    def build_cache_key(self, message: str, last_assistant_message: str | None = None) -> str:
        context = last_assistant_message or ""
        combined = f"{self.persona_hash}||{context}||{message}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get_cached_answer(
        self, message: str, last_assistant_message: str | None = None, is_continuation: bool = False
    ) -> str | None:
        if self.should_skip_cache(message, is_continuation):
            return None

        cache_key = self.build_cache_key(message, last_assistant_message)

        exact_match = await self.cache_repo.get_cache_by_key(cache_key)
        if exact_match:
            if exact_match.get("expires_at") and exact_match["expires_at"] < datetime.utcnow():
                await self.cache_repo.delete_cache_by_id(exact_match["id"])
            else:
                return await self.cache_repo.get_next_variation(exact_match["id"])

        all_cached = await self.cache_repo.get_all_cached_questions()
        if not all_cached:
            return None

        valid_cached = [
            c for c in all_cached if not c.get("expires_at") or c["expires_at"] >= datetime.utcnow()
        ]

        if not valid_cached:
            return None

        questions = [c["question"] for c in valid_cached]
        self.similarity.fit_on_corpus(questions + [message])

        best_match = self.similarity.find_best_match(message, valid_cached)
        if best_match:
            return await self.cache_repo.get_next_variation(best_match["id"])

        return None

    async def cache_answer(
        self,
        message: str,
        answer: str,
        last_assistant_message: str | None = None,
        is_continuation: bool = False,
    ) -> int | None:
        if self.should_skip_cache(message, is_continuation):
            return None

        cache_key = self.build_cache_key(message, last_assistant_message)
        cache_type = self.get_cache_type(is_continuation)
        expires_at = self.calculate_expiry(cache_type)

        context_preview = None
        if last_assistant_message:
            context_preview = last_assistant_message[:200]

        existing = await self.cache_repo.get_cache_by_key(cache_key)

        if existing:
            cache_id: int = existing["id"]
            await self.cache_repo.add_variation(cache_id, answer)
            return cache_id

        tfidf_vector = self.similarity.vectorize(message)
        return await self.cache_repo.create_cache(
            cache_key=cache_key,
            question=message,
            tfidf_vector=tfidf_vector,
            answer=answer,
            cache_type=cache_type.value,
            expires_at=expires_at,
            context_preview=context_preview,
        )

    async def should_cache(self, message: str, is_continuation: bool = False) -> bool:
        if self.should_skip_cache(message, is_continuation):
            return False

        all_cached = await self.cache_repo.get_all_cached_questions()
        if not all_cached:
            return True

        questions = [c["question"] for c in all_cached]
        self.similarity.fit_on_corpus(questions + [message])

        best_match = self.similarity.find_best_match(message, all_cached)
        return best_match is None

    async def clear_cache(self) -> int:
        return await self.cache_repo.clear_all_cache()

    async def cleanup_expired(self) -> int:
        return await self.cache_repo.delete_expired()

    async def get_cache_stats(self) -> dict:
        all_cached = await self.cache_repo.get_all_cached_questions()

        total_questions = len(all_cached)
        total_variations = sum(len(c.get("variations", [])) for c in all_cached)

        knowledge_count = sum(1 for c in all_cached if c.get("cache_type") == "knowledge")
        conversational_count = sum(1 for c in all_cached if c.get("cache_type") == "conversational")

        now = datetime.utcnow()
        expired_count = sum(1 for c in all_cached if c.get("expires_at") and c["expires_at"] < now)

        return {
            "total_questions": total_questions,
            "total_variations": total_variations,
            "avg_variations_per_question": total_variations / total_questions
            if total_questions > 0
            else 0,
            "knowledge_entries": knowledge_count,
            "conversational_entries": conversational_count,
            "expired_entries": expired_count,
        }

    async def list_cache_entries(
        self, page: int = 1, limit: int = 20, sort_by: str = "last_used", order: str = "desc"
    ) -> dict:
        return await self.cache_repo.list_cache_entries(page, limit, sort_by, order)

    async def get_cache_by_id(self, cache_id: int) -> dict | None:
        return await self.cache_repo.get_cache_by_id(cache_id)

    async def delete_cache_by_id(self, cache_id: int) -> bool:
        return await self.cache_repo.delete_cache_by_id(cache_id)

    async def update_cache_variations(self, cache_id: int, variations: list[str]) -> bool:
        return await self.cache_repo.update_cache_variations(cache_id, variations)

    async def search_cache(self, query: str, limit: int = 20) -> list[dict]:
        return await self.cache_repo.search_cache(query, limit)
