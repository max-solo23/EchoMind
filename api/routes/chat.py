from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from typing import Annotated

from models.requests import ChatRequest
from models.responses import ChatResponse
from api.middleware.auth import verify_api_key
from api.dependencies import get_chat_service
from Chat import Chat

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat_endpoint(
    request: ChatRequest,
    chat_service: Annotated[Chat, Depends(get_chat_service)],
    stream: bool = Query(False, description="Enable streaming response (SSE)")
):
    """
    Chat endpoint with optional streaming support.

    - Without stream parameter: Returns complete response (with evaluator if enabled)
    - With stream=true: Returns SSE stream (no evaluator)
    """
    try:
        if stream:
            # Streaming mode
            return StreamingResponse(
                chat_service.chat_stream(request.message, request.history),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming mode (existing behavior)
            reply = chat_service.chat(request.message, request.history)
            return ChatResponse(reply=reply)
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        )
    