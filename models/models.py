"""
SQLAlchemy database models for conversation logging and caching.

Design decisions:
1. conversations: Store every user-bot interaction
2. cached_answers: Store TF-IDF vectors and answer variations
3. session: Groups related conversations by user session
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Session(Base):
    """
    User session grouping related conversations.

    Why separate sessions?
    - Track conversation flow over time
    - Analytics: how many questions per session
    - Frontend: display conversation history by session
    """

    __tablename__ = "session"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    user_ip: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Conversation(Base):
    """
    Individual message exchange (user question + bot response).

    Why we store:
    - user_message: For similarity matching
    - bot_response: To return cached answers
    - timestamp: For analytics and sorting
    - session_id: To group related conversations
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), nullable=False, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    bot_response: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # Metadata for analytics
    tool_calls: Mapped[str | None] = mapped_column(JSON, nullable=True)
    evaluator_used: Mapped[bool] = mapped_column(default=False)
    evaluator_passed: Mapped[bool | None] = mapped_column(nullable=True)

    # Relationship back to Session
    session: Mapped["Session"] = relationship(back_populates="conversations")

    # Index for faster queries
    __table_args__ = (Index("ix_conversations_timestamp_desc", timestamp.desc()),)


class CachedAnswer(Base):
    """
    Cached answers with TF-IDF vectors for similarity matching.

    Design:
    - Context-aware caching: cache_key = hash(last_assistant_message + user_message)
    - One question can have up to 3 variations (stored as JSON array)
    - tfidf_vector: Serialized numpy array for similarity matching
    - TTL: knowledge cache (30 days) vs conversational cache (24 hours)

    Why context-aware keys?
    - Prevents "ok" after different conversations from returning wrong cached answers
    - Same question in different contexts produces different cache entries
    """

    __tablename__ = "cached_answers"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Context-aware cache key (SHA256 hash of context + question)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    # Original question text (for admin UI, debugging, similarity matching)
    question: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # Truncated context for debugging (first 200 chars of last assistant message)
    context_preview: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # TF-IDF vector for similarity matching
    tfidf_vector: Mapped[str] = mapped_column(Text, nullable=False)

    # Answer variations (JSON array, max 3 items)
    # Example: ["Answer 1", "Answer 2", "Answer 3"]
    variations: Mapped[str] = mapped_column(JSON, nullable=False)

    # Rotation tracking: which variation to show (0-2)
    variation_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Cache type and TTL
    # "knowledge" = standalone questions (30 day TTL)
    # "conversational" = context-dependent replies (24 hour TTL)
    cache_type: Mapped[str] = mapped_column(String(20), default="knowledge", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_used: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Indexes for efficient queries
    __table_args__ = (
        Index("ix_cached_answers_last_used", last_used.desc()),
        Index("ix_cached_answers_expires_at", expires_at),
        Index("ix_cached_answers_cache_type", cache_type),
    )
