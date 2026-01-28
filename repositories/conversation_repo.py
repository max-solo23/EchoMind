import json

from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.models import Conversation, Session


class SQLAlchemyConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(self, session_id: str, user_ip: str | None) -> int:
        result = await self.session.execute(select(Session).where(Session.session_id == session_id))
        existing = result.scalar_one_or_none()

        if existing:
            return existing.id

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
        tool_calls: list | None = None,
        evaluator_used: bool = False,
        evaluator_passed: bool | None = None,
    ) -> int:

        conversation = Conversation(
            session_id=session_db_id,
            user_message=user_message,
            bot_response=bot_response,
            tool_calls=json.dumps(tool_calls) if tool_calls else None,
            evaluator_used=evaluator_used,
            evaluator_passed=evaluator_passed,
        )

        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)

        return conversation.id

    async def get_session_by_id(self, session_id: str) -> dict | None:
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
                    "timestamp": conv.timestamp,
                }
                for conv in session.conversations
            ],
        }

    async def list_sessions(
        self, page: int = 1, limit: int = 20, sort_by: str = "created_at", order: str = "desc"
    ) -> dict:
        offset = (page - 1) * limit

        count_result = await self.session.execute(select(func.count(Session.id)))
        total = count_result.scalar()

        query = select(Session).options(selectinload(Session.conversations))

        if sort_by == "created_at":
            sort_col = Session.created_at
        elif sort_by == "last_activity":
            sort_col = Session.last_activity
        else:
            sort_col = Session.created_at

        query = query.order_by(desc(sort_col)) if order == "desc" else query.order_by(sort_col)

        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        sessions = result.scalars().all()

        return {
            "sessions": [
                {
                    "id": s.id,
                    "session_id": s.session_id,
                    "user_ip": s.user_ip,
                    "created_at": s.created_at,
                    "last_activity": s.last_activity,
                    "message_count": len(s.conversations),
                }
                for s in sessions
            ],
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit if total else 0,
        }

    async def delete_session(self, session_id: str) -> bool:
        result = await self.session.execute(select(Session).where(Session.session_id == session_id))
        session_obj = result.scalar_one_or_none()

        if not session_obj:
            return False

        await self.session.delete(session_obj)
        await self.session.commit()
        return True

    async def clear_all_sessions(self) -> int:
        result = await self.session.execute(select(func.count(Session.id)))
        count = result.scalar() or 0

        await self.session.execute(delete(Conversation))
        await self.session.execute(delete(Session))

        await self.session.commit()
        return count
