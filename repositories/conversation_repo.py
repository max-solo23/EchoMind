"""
SQLAlchemy implementation of ConversationRepository.

Learning notes:
- Uses async sessions (AsyncSession)
- Each method is a transaction (commits automatically)
- Handles errors and rollbacks
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from models.models import Session, Conversation
import json


class SQLAlchemyConversationRepository:
    """Concrete implementation using SQLAlchemy."""

    def __init__(self, session: AsyncSession):
        """
        Inject database session.

        Why dependency injection?
        - Testing: can inject mock session
        - Flexibility: session lifecycle managed externally
        - Transaction control: caller decides when to commit
        """
        self.session = session

    async def create_session(self, session_id: str, user_ip: Optional[str]) -> int:
        """
        Create new session or return existing.

        Returns: Database ID (integer primary key)
        """
        # Check if session already exists
        result = await self.session.execute(
            select(Session).where(Session.session_id == session_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing.id

        # Create new session
        new_session = Session(session_id=session_id, user_ip=user_ip)
        self.session.add(new_session)
        await self.session.commit()
        await self.session.refresh(new_session)

        return new_session.id

    async def log_conversation(
        self,
        session_db_id: int,
        user_message: str,
        bot_response: str,
        tool_calls: Optional[list] = None,
        evaluator_used: bool = False,
        evaluator_passed: Optional[bool] = None
    ) -> int:
        """Log conversation linked to session."""

        conversation = Conversation(
            session_id=session_db_id,
            user_message=user_message,
            bot_response=bot_response,
            tool_calls=json.dumps(tool_calls) if tool_calls else None,
            evaluator_used=evaluator_used,
            evaluator_passed=evaluator_passed
        )

        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)

        return conversation.id

    async def get_session_by_id(self, session_id: str) -> Optional[dict]:
        """Get session with all conversations."""

        result = await self.session.execute(
            select(Session)
            .where(Session.session_id == session_id)
            .options(selectinload(Session.conversations))
        )
        session = result.scalar_one_or_none()

        if not session:
            return None

        return {
            "id": session.id,
            "session_id": session.session_id,
            "user_ip": session.user_ip,
            "created_at": session.created_at,
            "conversations": [
                {
                    "user_message": conv.user_message,
                    "bot_response": conv.bot_response,
                    "timestamp": conv.timestamp
                }
                for conv in session.conversations
            ]
        }
