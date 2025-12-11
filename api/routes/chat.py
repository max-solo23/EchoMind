from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated

from models.requests import ChatRequest
from models.responses import ChatResponse
from api.middleware.auth import verify_api_key
from api.dependencies import get_chat_service
from Chat import Chat

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(
    request: ChatRequest,
    chat_service: Annotated[Chat, Depends(get_chat_service)]
) -> ChatResponse:
    try:
        reply = chat_service.chat(request.message, request.history)
        return ChatResponse(reply=reply)
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        )
    