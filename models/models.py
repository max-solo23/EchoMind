from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
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
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), nullable=False, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    bot_response: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    tool_calls: Mapped[str | None] = mapped_column(JSON, nullable=True)
    evaluator_used: Mapped[bool] = mapped_column(default=False)
    evaluator_passed: Mapped[bool | None] = mapped_column(nullable=True)

    session: Mapped["Session"] = relationship(back_populates="conversations")

    __table_args__ = (Index("ix_conversations_timestamp_desc", timestamp.desc()),)


class CachedAnswer(Base):
    __tablename__ = "cached_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    context_preview: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tfidf_vector: Mapped[str] = mapped_column(Text, nullable=False)
    variations: Mapped[str] = mapped_column(JSON, nullable=False)
    variation_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_type: Mapped[str] = mapped_column(String(20), default="knowledge", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_used: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("ix_cached_answers_last_used", last_used.desc()),
        Index("ix_cached_answers_expires_at", expires_at),
        Index("ix_cached_answers_cache_type", cache_type),
    )


class RateLimit(Base):
    __tablename__ = "rate_limit"

    key: Mapped[str] = mapped_column(String(500), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    expiry: Mapped[int] = mapped_column(Integer, nullable=False)
